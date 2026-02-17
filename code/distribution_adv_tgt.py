import argparse
import csv
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

import lpips as LPIPS
import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms as T
from accelerate import Accelerator
from diffusers import AutoencoderKL, StableDiffusionPipeline
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import CLDataset, init_transform
from pgd import pgd
from style_transfer import AdaINStyleTransfer, ONNXMosaicStyleTransfer
from utils import StepLossCollector

to_pil = T.ToPILImage()

MANIFEST_FIELDS = [
    "exp_id",
    "method",
    "budget_l2",
    "budget_lpips",
    "dataset_type",
    "dataset_name",
    "task_type",
    "finetune_method",
    "src_model",
    "tgt_model",
    "lora_rank",
    "lora_steps",
    "dreambooth_steps",
    "perturb_budget_l2",
    "perturb_budget_lpips_target",
    "optimize_steps",
    "gpu_hours",
    "seed",
    "prompt_set_id",
    "edit_prompt_set_id",
    "metric_version",
]

PER_IMAGE_FIELDS = [
    "exp_id",
    "run_id",
    "method",
    "budget_l2",
    "budget_lpips",
    "image_name",
    "image_adv_path",
    "attack_score",
    "input_l2",
    "input_l2_normed",
    "input_linf",
    "input_lpips",
    "decoded_lpips",
    "decoded_l2",
    "latent_l2",
    "input_psnr",
    "input_ssim",
    "decoded_psnr",
    "decoded_ssim",
]


def parse_args():
    parser = argparse.ArgumentParser()

    # Data/model
    parser.add_argument("--image_root", type=str, default="init_images", help="Main directory path of images.")
    parser.add_argument("--image_dirname", type=str, default="lego-minifigure-faces", help="Sub-directory name.")
    parser.add_argument("--model_path", type=str, default="model/stable-diffusion-v1-5")
    parser.add_argument("--clip_model_path", type=str, default="model/clip-vit-base-patch32")

    # Experiment protocol metadata
    parser.add_argument("--exp_id", type=str, default=None, help="Experiment id for experiments/outputs/<exp_id>.")
    parser.add_argument("--exp_version", type=str, default=None, help="Legacy alias of exp id.")
    parser.add_argument("--output_root", type=str, default="experiments/outputs")
    parser.add_argument("--adv_output_root", type=str, default=None, help="Deprecated alias for --output_root.")
    parser.add_argument(
        "--method",
        type=str,
        default="chocolatent",
        choices=["chocolatent", "glaze", "photoguard", "robust-ldm", "mist"],
    )
    parser.add_argument("--stage", type=str, default="A")
    parser.add_argument("--dataset_type", type=str, default="D_UNKNOWN")
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--task_type", type=str, default="style_imitation")
    parser.add_argument("--finetune_method", type=str, default="LoRA")
    parser.add_argument("--src_model", type=str, default="SD1.5")
    parser.add_argument("--tgt_model", type=str, default="SD1.5")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_steps", type=int, default=1000)
    parser.add_argument("--dreambooth_steps", type=int, default=0)
    parser.add_argument("--prompt_set_id", type=str, default="unset")
    parser.add_argument("--edit_prompt_set_id", type=str, default="unset")
    parser.add_argument("--metric_version", type=str, default="latent-protocol-v0.2")
    parser.add_argument("--seed", type=int, default=42)

    # Budget configuration
    parser.add_argument("--budget_l2", type=str, default="8/255", help="Normalized L2 budget, supports fraction.")
    parser.add_argument("--budget_lpips", type=str, default="0.2")
    parser.add_argument("--budget_l2_grid", type=str, default=None, help="Comma list, e.g. 4/255,8/255,12/255.")
    parser.add_argument("--budget_lpips_grid", type=str, default=None, help="Comma list, e.g. 0.1,0.2,0.5.")
    parser.add_argument("--eps", type=float, default=0.1, help="Optional L-inf clamp. Use <=0 to disable.")
    parser.add_argument("--iters", type=int, default=600)
    parser.add_argument("--initial_lr", type=float, default=1.0)
    parser.add_argument("--strict_lpips_projection", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lpips_bisection_steps", type=int, default=6)
    parser.add_argument("--budget_penalty_l2", type=float, default=20.0)
    parser.add_argument("--budget_penalty_lpips", type=float, default=20.0)
    parser.add_argument("--nan_lr_decay", type=float, default=0.5)
    parser.add_argument("--nan_min_lr", type=float, default=1e-4)
    parser.add_argument("--nan_max_recoveries", type=int, default=8)

    # Method specific targets
    parser.add_argument("--target_image_path", type=str, default=None, help="Required by method=mist.")
    parser.add_argument(
        "--glaze_style_backend",
        type=str,
        default="onnx_mosaic",
        choices=["onnx_mosaic", "adain"],
        help="glaze style backend.",
    )
    parser.add_argument(
        "--glaze_style_onnx_path",
        type=str,
        default="model/style_transfer/mosaic-9.onnx",
        help="Used when --glaze_style_backend=onnx_mosaic.",
    )
    parser.add_argument(
        "--glaze_style_image_path",
        type=str,
        default=None,
        help="Used when --glaze_style_backend=adain.",
    )
    parser.add_argument(
        "--glaze_style_alpha",
        type=float,
        default=1.0,
        help="Style transfer strength in [0,1] for adain backend.",
    )
    parser.add_argument("--gray_value", type=float, default=0.0, help="Target gray value in [-1,1] for photoguard.")

    # Dataloader/runtime
    parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--persistent_workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="fp16")
    parser.add_argument("--show_progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--collect_metrics", action="store_true")
    return parser.parse_args()


def parse_float_or_fraction(text):
    value = str(text).strip()
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        return float(numerator) / float(denominator)
    return float(value)


def parse_budget_grid(single_value, grid_text):
    if grid_text is None:
        return [parse_float_or_fraction(single_value)]
    values = []
    for token in grid_text.split(","):
        token = token.strip()
        if token:
            values.append(parse_float_or_fraction(token))
    if not values:
        raise ValueError("Budget grid is empty.")
    return values


def select_model_dtype(mixed_precision, device):
    if device.type != "cuda":
        return torch.float32
    if mixed_precision == "bf16":
        return torch.bfloat16
    if mixed_precision == "fp16":
        return torch.float16
    return torch.float32


def build_default_exp_id(args):
    dataset_name = args.dataset_name or args.image_dirname
    time_tag = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{time_tag}-{args.method}-{dataset_name}"


def format_budget_tag(value):
    return f"{value:.6f}".rstrip("0").rstrip(".").replace(".", "p")


def load_vae(args, accelerator, model_dtype):
    try:
        vae = AutoencoderKL.from_pretrained(
            args.model_path,
            subfolder="vae",
            torch_dtype=model_dtype,
        )
        accelerator.print("[Info] Loaded VAE via AutoencoderKL.")
    except Exception as err:
        accelerator.print(f"[Warn] AutoencoderKL load failed, fallback to StableDiffusionPipeline. err={err}")
        pipeline = StableDiffusionPipeline.from_pretrained(
            args.model_path,
            torch_dtype=model_dtype,
        )
        vae = pipeline.vae
        del pipeline

    vae = vae.to(accelerator.device).eval()
    for param in vae.parameters():
        param.requires_grad_(False)
    return vae


def build_dataloader(args):
    image_dir = os.path.join(args.image_root, args.image_dirname)
    dataset = CLDataset(image_dir)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "drop_last": False,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = args.persistent_workers
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    dataloader = DataLoader(dataset, **loader_kwargs)
    return image_dir, dataloader


def prepare_protocol_dirs(exp_root):
    paths = {
        "root": exp_root,
        "protected_images": exp_root / "protected_images",
        "ft_clean": exp_root / "ft_clean",
        "ft_adv": exp_root / "ft_adv",
        "generated_clean": exp_root / "generated_clean",
        "generated_adv": exp_root / "generated_adv",
        "edited_clean": exp_root / "edited_clean",
        "edited_adv": exp_root / "edited_adv",
        "metrics": exp_root / "metrics",
        "logs": exp_root / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def write_tensorboard(step_collector, log_path):
    os.makedirs(log_path, exist_ok=True)
    tb_writer = SummaryWriter(log_path)
    for step, step_data in step_collector.step_metrics.items():
        for metric_name, metric_values in step_data.items():
            if not metric_values:
                continue
            metric_array = np.array(metric_values)
            tb_writer.add_scalar(metric_name, float(metric_array.mean()), step)
            tb_writer.add_histogram(f"{metric_name}_dist", metric_array, step)
    tb_writer.flush()
    tb_writer.close()


def load_target_tensor(args, device, dtype):
    if args.method != "mist":
        return None
    if not args.target_image_path:
        raise ValueError("method=mist requires --target_image_path.")
    image = Image.open(args.target_image_path).convert("RGB")
    target_tensor = init_transform(image).unsqueeze(0).to(device=device, dtype=dtype)
    return target_tensor.clamp(-1, 1)


def build_glaze_style_transfer(args, device, dtype):
    if args.method != "glaze":
        return None
    if args.glaze_style_backend == "adain":
        if not args.glaze_style_image_path:
            raise ValueError("method=glaze with backend=adain requires --glaze_style_image_path.")
        image = Image.open(args.glaze_style_image_path).convert("RGB")
        style_tensor = init_transform(image).unsqueeze(0).to(device=device, dtype=dtype).clamp(-1, 1)
        return AdaINStyleTransfer(style_tensor=style_tensor, alpha=args.glaze_style_alpha)

    if args.glaze_style_backend == "onnx_mosaic":
        return ONNXMosaicStyleTransfer(
            model_path=args.glaze_style_onnx_path,
            prefer_cuda=(device.type == "cuda"),
        )

    raise ValueError(f"Unknown glaze_style_backend: {args.glaze_style_backend}")


def gather_list_records(local_records, accelerator):
    if accelerator.num_processes == 1:
        return local_records
    if not dist.is_available() or not dist.is_initialized():
        return local_records

    gathered = [None for _ in range(accelerator.num_processes)]
    dist.all_gather_object(gathered, local_records)
    merged = []
    for records in gathered:
        if records:
            merged.extend(records)
    return merged


def write_config(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        # JSON syntax is valid YAML.
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_rows_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_rows_csv(path):
    if not Path(path).exists():
        return []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def merge_manifest_rows(existing_rows, new_rows):
    # Keep only one row for each logical experiment unit.
    def row_key(row):
        return (
            str(row.get("method", "")),
            str(row.get("budget_l2", "")),
            str(row.get("budget_lpips", "")),
            str(row.get("dataset_name", "")),
            str(row.get("task_type", "")),
            str(row.get("src_model", "")),
            str(row.get("tgt_model", "")),
            str(row.get("seed", "")),
        )

    merged = {}
    for row in existing_rows:
        merged[row_key(row)] = {field: row.get(field, "") for field in MANIFEST_FIELDS}
    for row in new_rows:
        merged[row_key(row)] = {field: row.get(field, "") for field in MANIFEST_FIELDS}
    return list(merged.values())


def summarize_metrics(records):
    summary = {
        "num_images": len(records),
    }
    metric_keys = [
        "attack_score",
        "input_l2",
        "input_l2_normed",
        "input_linf",
        "input_lpips",
        "decoded_lpips",
        "decoded_l2",
        "latent_l2",
        "input_psnr",
        "input_ssim",
        "decoded_psnr",
        "decoded_ssim",
    ]
    for key in metric_keys:
        values = np.array([record[key] for record in records], dtype=np.float64)
        if values.size == 0:
            summary[f"{key}_mean"] = None
            summary[f"{key}_std"] = None
            summary[f"{key}_ci95"] = None
            continue
        mean_value = float(values.mean())
        std_value = float(values.std(ddof=1)) if values.size > 1 else 0.0
        ci95 = 1.96 * std_value / np.sqrt(values.size) if values.size > 1 else 0.0
        summary[f"{key}_mean"] = mean_value
        summary[f"{key}_std"] = std_value
        summary[f"{key}_ci95"] = float(ci95)
    return summary


def format_metric_for_log(value):
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()

    if args.adv_output_root is not None and args.output_root == "experiments/outputs":
        args.output_root = args.adv_output_root

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    set_seed(args.seed + accelerator.process_index)
    model_dtype = select_model_dtype(args.mixed_precision, accelerator.device)

    budget_l2_list = parse_budget_grid(args.budget_l2, args.budget_l2_grid)
    budget_lpips_list = parse_budget_grid(args.budget_lpips, args.budget_lpips_grid)

    exp_id = args.exp_id or args.exp_version or build_default_exp_id(args)
    exp_root = Path(args.output_root) / exp_id
    protocol_dirs = prepare_protocol_dirs(exp_root)

    image_dir, dataloader = build_dataloader(args)
    dataloader = accelerator.prepare(dataloader)

    vae = load_vae(args, accelerator, model_dtype)
    lpips_model = LPIPS.LPIPS(net="alex").to(accelerator.device).eval()
    for param in lpips_model.parameters():
        param.requires_grad_(False)

    target_tensor = load_target_tensor(args, accelerator.device, model_dtype)
    glaze_style_transfer = build_glaze_style_transfer(args, accelerator.device, model_dtype)
    dataset_name = args.dataset_name or args.image_dirname

    if accelerator.is_main_process:
        global_config = vars(args).copy()
        global_config["exp_id"] = exp_id
        global_config["budget_l2_list"] = budget_l2_list
        global_config["budget_lpips_list"] = budget_lpips_list
        global_config["image_dir"] = image_dir
        global_config["created_at"] = datetime.now().isoformat(timespec="seconds")
        write_config(exp_root / "config.yaml", global_config)

    accelerator.print(
        f"[Config] exp_id={exp_id} method={args.method} stage={args.stage} "
        f"dataset={dataset_name} budgets={len(budget_l2_list)}x{len(budget_lpips_list)}"
    )

    manifest_rows = []
    for budget_l2 in budget_l2_list:
        for budget_lpips in budget_lpips_list:
            run_id = f"{args.method}-b2_{format_budget_tag(budget_l2)}-bp_{format_budget_tag(budget_lpips)}"
            protected_dir = protocol_dirs["protected_images"] / args.method / run_id / args.image_dirname
            run_metrics_dir = protocol_dirs["metrics"] / run_id
            run_logs_dir = protocol_dirs["logs"] / run_id
            protected_dir.mkdir(parents=True, exist_ok=True)
            run_metrics_dir.mkdir(parents=True, exist_ok=True)
            run_logs_dir.mkdir(parents=True, exist_ok=True)

            step_collector = StepLossCollector() if (args.collect_metrics and accelerator.is_main_process) else None

            local_generated = 0
            local_skipped = 0
            local_records = []
            run_start_time = time.time()

            for batch_idx, (init_images, image_args) in enumerate(dataloader):
                image_names = list(image_args["image_name"])
                selected_indices = []
                selected_paths = []
                selected_names = []

                for idx, image_name in enumerate(image_names):
                    image_adv_name = f"{Path(image_name).stem}_adv-{run_id}.png"
                    image_adv_path = protected_dir / image_adv_name
                    should_run = args.rewrite or (not image_adv_path.exists())
                    if should_run:
                        selected_indices.append(idx)
                        selected_paths.append(image_adv_path)
                        selected_names.append(image_name)

                if not selected_indices:
                    local_skipped += len(image_names)
                    continue

                batch_images = init_images[selected_indices].to(accelerator.device, non_blocking=True)
                if accelerator.is_local_main_process:
                    accelerator.print(
                        f"[Rank {accelerator.process_index}] run={run_id} batch={batch_idx} "
                        f"run={len(selected_indices)} skip={len(image_names) - len(selected_indices)}"
                    )

                target_for_method = target_tensor
                if args.method == "glaze":
                    target_for_method = glaze_style_transfer(batch_images)

                x_adv, pgd_info = pgd(
                    batch_images,
                    vae,
                    iters=args.iters,
                    initial_lr=args.initial_lr,
                    eps=args.eps,
                    method=args.method,
                    budget_l2_normed=budget_l2,
                    budget_lpips=budget_lpips,
                    lpips_model=lpips_model,
                    step_collector=step_collector,
                    target_image=target_for_method,
                    gray_value=args.gray_value,
                    strict_lpips_projection=args.strict_lpips_projection,
                    lpips_bisection_steps=args.lpips_bisection_steps,
                    budget_penalty_l2=args.budget_penalty_l2,
                    budget_penalty_lpips=args.budget_penalty_lpips,
                    nan_lr_decay=args.nan_lr_decay,
                    nan_min_lr=args.nan_min_lr,
                    nan_max_recoveries=args.nan_max_recoveries,
                    show_progress=args.show_progress and accelerator.is_local_main_process,
                    log_every=args.log_every,
                    return_info=True,
                )

                sample_metrics = pgd_info["final_per_sample"]
                x_adv_img = ((x_adv / 2 + 0.5).clamp(0, 1)).detach().cpu()

                for local_idx, (image_tensor, image_name, image_adv_path) in enumerate(
                    zip(x_adv_img, selected_names, selected_paths)
                ):
                    to_pil(image_tensor).convert("RGB").save(image_adv_path)
                    local_generated += 1

                    row = {
                        "exp_id": exp_id,
                        "run_id": run_id,
                        "method": args.method,
                        "budget_l2": float(budget_l2),
                        "budget_lpips": float(budget_lpips),
                        "image_name": image_name,
                        "image_adv_path": str(image_adv_path),
                        "attack_score": float(sample_metrics["attack_score"][local_idx]),
                        "input_l2": float(sample_metrics["input_l2"][local_idx]),
                        "input_l2_normed": float(sample_metrics["input_l2_normed"][local_idx]),
                        "input_linf": float(sample_metrics["input_linf"][local_idx]),
                        "input_lpips": float(sample_metrics["input_lpips"][local_idx]),
                        "decoded_lpips": float(sample_metrics["decoded_lpips"][local_idx]),
                        "decoded_l2": float(sample_metrics["decoded_l2"][local_idx]),
                        "latent_l2": float(sample_metrics["latent_l2"][local_idx]),
                        "input_psnr": float(sample_metrics["input_psnr"][local_idx]),
                        "input_ssim": float(sample_metrics["input_ssim"][local_idx]),
                        "decoded_psnr": float(sample_metrics["decoded_psnr"][local_idx]),
                        "decoded_ssim": float(sample_metrics["decoded_ssim"][local_idx]),
                    }
                    local_records.append(row)

                    if accelerator.is_local_main_process:
                        accelerator.print(f"[Saved] {image_name} -> {image_adv_path}")

                local_skipped += len(image_names) - len(selected_indices)

            accelerator.wait_for_everyone()
            run_duration = time.time() - run_start_time
            gpu_hours = run_duration * accelerator.num_processes / 3600.0

            stats = torch.tensor([local_generated, local_skipped], device=accelerator.device, dtype=torch.long)
            if accelerator.num_processes > 1:
                stats = accelerator.reduce(stats, reduction="sum")

            merged_records = gather_list_records(local_records, accelerator)

            if accelerator.is_main_process:
                merged_records = sorted(merged_records, key=lambda item: item["image_name"])
                summary = summarize_metrics(merged_records)
                summary.update(
                    {
                        "exp_id": exp_id,
                        "run_id": run_id,
                        "stage": args.stage,
                        "method": args.method,
                        "budget_l2": float(budget_l2),
                        "budget_lpips": float(budget_lpips),
                        "generated": int(stats[0].item()),
                        "skipped": int(stats[1].item()),
                        "duration_sec": float(run_duration),
                        "gpu_hours": float(gpu_hours),
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                    }
                )

                write_rows_csv(run_metrics_dir / "per_image.csv", merged_records, PER_IMAGE_FIELDS)
                with open(run_metrics_dir / "metrics.json", "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2, ensure_ascii=False)

                run_config = vars(args).copy()
                run_config.update(
                    {
                        "exp_id": exp_id,
                        "run_id": run_id,
                        "dataset_name": dataset_name,
                        "image_dir": image_dir,
                        "budget_l2": float(budget_l2),
                        "budget_lpips": float(budget_lpips),
                        "created_at": datetime.now().isoformat(timespec="seconds"),
                    }
                )
                write_config(run_metrics_dir / "config.yaml", run_config)

                if step_collector is not None and step_collector.step_metrics:
                    current_time = datetime.now().strftime("%m-%d-%H-%M")
                    tb_path = run_logs_dir / current_time
                    write_tensorboard(step_collector, str(tb_path))

                manifest_row = {
                    "exp_id": exp_id,
                    "method": args.method,
                    "budget_l2": float(budget_l2),
                    "budget_lpips": float(budget_lpips),
                    "dataset_type": args.dataset_type,
                    "dataset_name": dataset_name,
                    "task_type": args.task_type,
                    "finetune_method": args.finetune_method,
                    "src_model": args.src_model,
                    "tgt_model": args.tgt_model,
                    "lora_rank": args.lora_rank,
                    "lora_steps": args.lora_steps,
                    "dreambooth_steps": args.dreambooth_steps,
                    "perturb_budget_l2": float(budget_l2),
                    "perturb_budget_lpips_target": float(budget_lpips),
                    "optimize_steps": args.iters,
                    "gpu_hours": float(gpu_hours),
                    "seed": args.seed,
                    "prompt_set_id": args.prompt_set_id,
                    "edit_prompt_set_id": args.edit_prompt_set_id,
                    "metric_version": args.metric_version,
                }
                manifest_rows.append(manifest_row)
                accelerator.print(
                    f"[Done] run={run_id} generated={summary['generated']} skipped={summary['skipped']} "
                    f"input_lpips_mean={format_metric_for_log(summary['input_lpips_mean'])} "
                    f"decoded_lpips_mean={format_metric_for_log(summary['decoded_lpips_mean'])}"
                )

            accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        manifest_path = exp_root / "manifest.csv"
        existing_rows = read_rows_csv(manifest_path)
        final_manifest_rows = merge_manifest_rows(existing_rows, manifest_rows)
        final_manifest_rows = sorted(
            final_manifest_rows,
            key=lambda row: (
                row.get("method", ""),
                row.get("dataset_name", ""),
                str(row.get("budget_l2", "")),
                str(row.get("budget_lpips", "")),
                str(row.get("seed", "")),
            ),
        )
        write_rows_csv(manifest_path, final_manifest_rows, MANIFEST_FIELDS)
        accelerator.print(f"[Manifest] {manifest_path}")
        accelerator.print(f"[Experiment Done] {exp_root}")


if __name__ == "__main__":
    main()
