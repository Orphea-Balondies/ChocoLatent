#!/usr/bin/env python3
import argparse
import csv
import importlib
import json
import os
import random
import select
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

VALID_IMAGE_SUFFIX = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run full protocol pipeline: "
            "protect -> LoRA train(clean/adv) -> generation(clean/adv) -> conclusion metrics."
        )
    )

    # Experiment basics
    parser.add_argument("--exp_id", type=str, required=True)
    parser.add_argument("--output_root", type=str, default="experiments/outputs")
    parser.add_argument("--model_path", type=str, default="model/stable-diffusion-v1-5")
    parser.add_argument("--clip_model_path", type=str, default="model/clip-vit-base-patch32")
    parser.add_argument("--image_root", type=str, default="init_images")
    parser.add_argument("--image_dirname", type=str, required=True)
    parser.add_argument("--dataset_type", type=str, default="D_UNKNOWN")
    parser.add_argument("--task_type", type=str, default="style_imitation")
    parser.add_argument("--src_model", type=str, default="SD1.5")
    parser.add_argument("--tgt_model", type=str, default="SD1.5")

    # Method / budget sweep
    parser.add_argument(
        "--methods",
        type=str,
        default="chocolatent,glaze,photoguard,robust-ldm,mist",
        help="Comma separated methods",
    )
    parser.add_argument("--budget_l2_grid", type=str, default="4/255,8/255,12/255")
    parser.add_argument("--budget_lpips_grid", type=str, default="0.1,0.2,0.5")
    parser.add_argument("--mist_target_image_path", type=str, default="/root/chocolatent/MIST.png")
    parser.add_argument("--glaze_style_backend", choices=["onnx_mosaic", "adain"], default="onnx_mosaic")
    parser.add_argument("--glaze_style_onnx_path", type=str, default="model/style_transfer/mosaic-9.onnx")
    parser.add_argument("--glaze_style_image_path", type=str, default=None)
    parser.add_argument("--glaze_style_alpha", type=float, default=1.0)

    # Protect stage args
    parser.add_argument("--protect_iters", type=int, default=600)
    parser.add_argument("--protect_initial_lr", type=float, default=1.0)
    parser.add_argument("--protect_eps", type=float, default=0.1)
    parser.add_argument("--protect_batch_size", type=int, default=4)
    parser.add_argument("--protect_num_workers", type=int, default=4)
    parser.add_argument("--protect_mixed_precision", choices=["no", "fp16", "bf16"], default="fp16")
    parser.add_argument("--protect_log_every", type=int, default=50)
    parser.add_argument("--rewrite_protect", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--skip_protect",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip protection stage and reuse existing protected images/manifest.",
    )
    parser.add_argument("--show_protect_progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--protect_nan_lr_decay", type=float, default=0.5)
    parser.add_argument("--protect_nan_min_lr", type=float, default=1e-4)
    parser.add_argument("--protect_nan_max_recoveries", type=int, default=8)

    # LoRA training args
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--checkpointing_steps", type=int, default=5000)
    parser.add_argument("--validation_epochs", type=int, default=10)
    parser.add_argument("--validation_prompt", type=str, default=None)
    parser.add_argument("--caption_template", type=str, default="a photo of {name}")
    parser.add_argument("--caption_column", type=str, default="caption")
    parser.add_argument("--overwrite_metadata", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--overwrite_lora", action=argparse.BooleanOptionalAction, default=False)

    # Generation args
    parser.add_argument("--prompt_file", type=str, required=True, help="txt/json list prompts")
    parser.add_argument("--seed_file", type=str, default=None, help="txt/json list seeds")
    parser.add_argument("--seeds", type=str, default=None, help="Comma separated seeds")
    parser.add_argument("--num_random_seeds", type=int, default=8)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--negative_prompt", type=str, default=None)
    parser.add_argument("--overwrite_generate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--generate_dtype", choices=["fp16", "bf16", "fp32"], default="fp16")

    # Metrics args
    parser.add_argument("--metrics_batch_size", type=int, default=8)
    parser.add_argument("--compute_fid", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--strict_metrics", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--overwrite_metrics", action=argparse.BooleanOptionalAction, default=False)

    # Runtime
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strict_preflight", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry_run", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--stop_on_error", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--stream_child_logs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stream child process output to terminal while still writing pipeline.log.",
    )
    parser.add_argument(
        "--heartbeat_sec",
        type=int,
        default=120,
        help="If no child output for this many seconds, print heartbeat status.",
    )
    return parser.parse_args()


def parse_methods(text: str) -> List[str]:
    methods = [item.strip() for item in text.split(",") if item.strip()]
    allowed = {"chocolatent", "glaze", "photoguard", "robust-ldm", "mist"}
    invalid = sorted(set(methods) - allowed)
    if invalid:
        raise ValueError(f"Unknown methods: {invalid}")
    if not methods:
        raise ValueError("methods is empty")
    return methods


def parse_float_or_fraction(text: str) -> float:
    value = str(text).strip()
    if "/" in value:
        num, den = value.split("/", 1)
        return float(num) / float(den)
    return float(value)


def parse_budget_grid(text: str) -> List[float]:
    values = [parse_float_or_fraction(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError(f"Empty budget grid: {text}")
    return values


def parse_list_file_or_csv(file_path: str, csv_text: str, rng_seed: int, random_count: int) -> List[int]:
    items: List[int] = []
    if file_path:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Seed file not found: {path}")
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, list):
                raise ValueError("Seed json must be a list.")
            items.extend(int(x) for x in payload)
        else:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    items.append(int(line))
    elif csv_text:
        items.extend(int(x.strip()) for x in csv_text.split(",") if x.strip())
    else:
        rng = random.Random(rng_seed)
        items = [rng.randint(0, 2**32 - 1) for _ in range(int(random_count))]

    # stable dedup
    result = []
    seen = set()
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    if not result:
        raise ValueError("No seeds resolved.")
    return result


def load_prompts(prompt_file: str) -> List[str]:
    path = Path(prompt_file)
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("Prompt json must be a list.")
        prompts = [str(x).strip() for x in payload if str(x).strip()]
    else:
        prompts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not prompts:
        raise ValueError("No prompts resolved.")
    # stable dedup
    result = []
    seen = set()
    for prompt in prompts:
        if prompt not in seen:
            seen.add(prompt)
            result.append(prompt)
    return result


def find_images(root: Path) -> List[Path]:
    images = []
    for path in root.rglob("*"):
        if path.suffix.lower() in VALID_IMAGE_SUFFIX:
            images.append(path)
    images.sort()
    return images


def ensure_metadata_jsonl(dataset_dir: Path, dataset_token: str, caption_template: str, overwrite: bool, dry_run: bool):
    dataset_dir.mkdir(parents=True, exist_ok=True)
    images = find_images(dataset_dir)
    if not images:
        raise RuntimeError(f"No images found in dataset_dir={dataset_dir}")

    metadata_path = dataset_dir / "metadata.jsonl"
    if metadata_path.exists() and not overwrite:
        existing_lines = [line for line in metadata_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return metadata_path, len(existing_lines)

    rows = []
    for image_path in images:
        rel_path = image_path.relative_to(dataset_dir).as_posix()
        txt_path = image_path.with_suffix(".txt")
        if txt_path.exists():
            caption = txt_path.read_text(encoding="utf-8").strip()
        else:
            caption = ""
        if not caption:
            caption = caption_template.format(
                name=dataset_token,
                file_stem=image_path.stem,
                file_name=image_path.name,
            )
        rows.append({"file_name": rel_path, "caption": caption})

    if not dry_run:
        with metadata_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return metadata_path, len(rows)


def read_manifest(manifest_path: Path) -> List[Dict]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def manifest_is_complete(manifest_path: Path, expected_count: int) -> bool:
    if not manifest_path.exists():
        return False
    try:
        rows = read_manifest(manifest_path)
    except Exception:
        return False
    if len(rows) != expected_count:
        return False

    sample_ids = []
    for row in rows:
        sample_id = row.get("sample_id", "")
        image_path = row.get("image_path", "")
        if not sample_id or not image_path:
            return False
        if not Path(image_path).exists():
            return False
        sample_ids.append(sample_id)

    if len(set(sample_ids)) != expected_count:
        return False
    return True


def format_budget_tag(value) -> str:
    return f"{float(value):.6f}".rstrip("0").rstrip(".").replace(".", "p")


def run_id_from_manifest_row(row: Dict) -> str:
    method = row["method"]
    b2 = format_budget_tag(float(row["budget_l2"]))
    bp = format_budget_tag(float(row["budget_lpips"]))
    return f"{method}-b2_{b2}-bp_{bp}"


def run_cmd(
    cmd: List[str],
    cwd: Path,
    log_file: Path,
    dry_run: bool,
    stream_child_logs: bool = True,
    heartbeat_sec: int = 120,
):
    line = " ".join(cmd)
    print(f"[CMD] {line}")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as f:
        f.write(f"\n$ {line}\n")
        f.flush()
    if dry_run:
        return

    heartbeat_sec = max(0, int(heartbeat_sec))
    start_time = time.time()
    last_activity = start_time

    with log_file.open("a", encoding="utf-8") as f:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=False,
            bufsize=0,
        )
        if proc.stdout is None:
            raise RuntimeError("Failed to capture child stdout.")
        stdout_fd = proc.stdout.fileno()

        while True:
            ready, _, _ = select.select([stdout_fd], [], [], 1.0)
            if ready:
                chunk = os.read(stdout_fd, 65536)
                if chunk:
                    last_activity = time.time()
                    f.buffer.write(chunk)
                    f.flush()
                    if stream_child_logs:
                        sys.stdout.buffer.write(chunk)
                        sys.stdout.flush()
                elif proc.poll() is not None:
                    break
            else:
                if heartbeat_sec > 0 and (time.time() - last_activity) >= heartbeat_sec:
                    elapsed = int(time.time() - start_time)
                    heartbeat = (
                        f"[Heartbeat] child still running ({elapsed}s elapsed): {line}\n"
                        f"[Heartbeat] log file: {log_file}\n"
                    )
                    f.write(heartbeat)
                    f.flush()
                    if stream_child_logs:
                        print(heartbeat, end="", flush=True)
                    last_activity = time.time()
                if proc.poll() is not None:
                    break

        while True:
            chunk = proc.stdout.read()
            if not chunk:
                break
            f.buffer.write(chunk)
            f.flush()
            if stream_child_logs:
                sys.stdout.buffer.write(chunk)
                sys.stdout.flush()

        proc.wait()

    if proc.returncode != 0:
        raise RuntimeError(f"Command failed (code={proc.returncode}). See log: {log_file}")


def preflight_check(args, repo_root: Path, methods: List[str], dataset_dir: Path, prompts: List[str], seeds: List[int]):
    required_modules = [
        "torch",
        "diffusers",
        "transformers",
        "accelerate",
        "datasets",
        "lpips",
        "numpy",
        "skimage",
        "peft",
    ]
    issues = []
    module_status = {}
    for module_name in required_modules:
        try:
            importlib.import_module(module_name)
            module_status[module_name] = "ok"
        except Exception as err:
            module_status[module_name] = f"missing: {err}"
            issues.append(f"Python module missing: {module_name} ({err})")

    if shutil.which("accelerate") is None:
        issues.append("Executable missing: accelerate")

    model_path = repo_root / args.model_path
    clip_model_path = repo_root / args.clip_model_path
    if not model_path.exists():
        issues.append(f"Model path not found: {model_path}")
    if not clip_model_path.exists():
        issues.append(f"CLIP model path not found: {clip_model_path}")
    if not dataset_dir.exists():
        issues.append(f"Dataset dir not found: {dataset_dir}")
    elif not find_images(dataset_dir):
        issues.append(f"No images found under dataset dir: {dataset_dir}")
    if not prompts:
        issues.append("Prompt list is empty")
    if not seeds:
        issues.append("Seed list is empty")
    if "mist" in methods and not args.mist_target_image_path:
        issues.append("method=mist requires --mist_target_image_path")
    if "mist" in methods and args.mist_target_image_path:
        mist_target = Path(args.mist_target_image_path)
        if not mist_target.exists():
            issues.append(f"Mist target image not found: {mist_target}")
    if "glaze" in methods:
        if args.glaze_style_backend == "adain":
            if not args.glaze_style_image_path:
                issues.append("method=glaze with backend=adain requires --glaze_style_image_path")
            else:
                glaze_style = Path(args.glaze_style_image_path)
                if not glaze_style.is_absolute():
                    glaze_style = (repo_root / glaze_style).resolve()
                if not glaze_style.exists():
                    issues.append(f"Glaze style image not found: {glaze_style}")
        elif args.glaze_style_backend == "onnx_mosaic":
            glaze_onnx = Path(args.glaze_style_onnx_path)
            if not glaze_onnx.is_absolute():
                glaze_onnx = (repo_root / glaze_onnx).resolve()
            if not glaze_onnx.exists():
                issues.append(f"Glaze ONNX style model not found: {glaze_onnx}")
            try:
                importlib.import_module("onnxruntime")
            except Exception as err:
                issues.append(f"Python module missing: onnxruntime ({err})")

    scripts_to_check = [
        repo_root / "code" / "distribution_adv_tgt.py",
        repo_root / "code" / "text2image_generate.py",
        repo_root / "experiments" / "scripts" / "run_lora_train.py",
        repo_root / "experiments" / "scripts" / "run_conclusion_metrics.py",
    ]
    for script_path in scripts_to_check:
        if not script_path.exists():
            issues.append(f"Required script not found: {script_path}")

    if args.strict_metrics:
        issues.append(
            "strict_metrics is enabled, but full protocol metrics require extra assets "
            "(style classifier, edit sets, cross-model/robustness runs, IQA backend). "
            "Disable --strict_metrics or provide those assets and extend evaluators."
        )

    report = {
        "module_status": module_status,
        "issues": issues,
        "strict_preflight": args.strict_preflight,
    }
    return report


def build_lora_train_cmd(args, train_data_dir: Path, output_dir: Path, validation_prompt: str, seed: int):
    cmd = [
        "python",
        "experiments/scripts/run_lora_train.py",
        "--pretrained_model_name_or_path",
        args.model_path,
        "--train_data_dir",
        str(train_data_dir),
        "--caption_column",
        args.caption_column,
        "--output_dir",
        str(output_dir),
        "--validation_prompt",
        validation_prompt,
        "--seed",
        str(seed),
        "--resolution",
        str(args.resolution),
        "--train_batch_size",
        str(args.train_batch_size),
        "--num_train_epochs",
        str(args.num_train_epochs),
        "--learning_rate",
        str(args.learning_rate),
        "--checkpointing_steps",
        str(args.checkpointing_steps),
        "--validation_epochs",
        str(args.validation_epochs),
    ]
    if args.max_train_steps > 0:
        cmd.extend(["--max_train_steps", str(args.max_train_steps)])
    return cmd


def build_generate_cmd(
    args,
    output_root: Path,
    lora_path: Path,
    prompt_file: Path,
    seed_file: Path,
):
    cmd = [
        "python",
        "code/text2image_generate.py",
        "--model_path",
        args.model_path,
        "--loras",
        str(lora_path),
        "--output_root",
        str(output_root),
        "--prompt_file",
        str(prompt_file),
        "--seed_file",
        str(seed_file),
        "--num_inference_steps",
        str(args.num_inference_steps),
        "--guidance_scale",
        str(args.guidance_scale),
        "--dtype",
        args.generate_dtype,
    ]
    if args.negative_prompt:
        cmd.extend(["--negative_prompt", args.negative_prompt])
    cmd.append("--overwrite" if args.overwrite_generate else "--no-overwrite")
    return cmd


def get_lora_weight_path(lora_dir: Path):
    return lora_dir / "pytorch_lora_weights.safetensors"


def load_conclusion_metrics(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_summary(summary_rows: List[Dict], output_csv: Path):
    fieldnames = sorted({k for row in summary_rows for k in row.keys()})
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    exp_root = (repo_root / args.output_root / args.exp_id).resolve()
    dataset_dir = (repo_root / args.image_root / args.image_dirname).resolve()
    methods = parse_methods(args.methods)
    prompts = load_prompts(args.prompt_file)
    seeds = parse_list_file_or_csv(args.seed_file, args.seeds, args.seed, args.num_random_seeds)
    expected_samples_per_adapter = len(prompts) * len(seeds)

    log_dir = exp_root / "logs" / "pipeline"
    log_file = log_dir / "pipeline.log"
    log_dir.mkdir(parents=True, exist_ok=True)

    def run_pipeline_cmd(cmd: List[str]):
        run_cmd(
            cmd,
            cwd=repo_root,
            log_file=log_file,
            dry_run=args.dry_run,
            stream_child_logs=args.stream_child_logs,
            heartbeat_sec=args.heartbeat_sec,
        )

    preflight = preflight_check(args, repo_root=repo_root, methods=methods, dataset_dir=dataset_dir, prompts=prompts, seeds=seeds)
    preflight_path = exp_root / "pipeline_preflight.json"
    preflight_path.parent.mkdir(parents=True, exist_ok=True)
    preflight_path.write_text(json.dumps(preflight, indent=2, ensure_ascii=False), encoding="utf-8")
    if preflight["issues"]:
        print("[Preflight] issues detected:")
        for issue in preflight["issues"]:
            print(f"  - {issue}")
        if args.strict_preflight:
            raise SystemExit(
                "Preflight failed in strict mode. "
                f"Fix issues in {preflight_path} and rerun."
            )

    pipeline_config = vars(args).copy()
    pipeline_config["methods_resolved"] = methods
    pipeline_config["prompts_resolved"] = prompts
    pipeline_config["seeds_resolved"] = seeds
    (exp_root / "pipeline_config.json").write_text(json.dumps(pipeline_config, indent=2, ensure_ascii=False), encoding="utf-8")

    start_time = time.time()
    run_errors = []
    summary_rows = []

    inputs_dir = exp_root / "pipeline_inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    prompt_run_file = inputs_dir / "prompts.txt"
    seed_run_file = inputs_dir / "seeds.txt"
    prompt_run_file.write_text("\n".join(prompts) + "\n", encoding="utf-8")
    seed_run_file.write_text("\n".join(str(seed) for seed in seeds) + "\n", encoding="utf-8")

    # Step 1. Protect image sweep for all methods.
    if args.skip_protect:
        print(f"[Skip] protection stage disabled. Reusing existing manifest at: {exp_root / 'manifest.csv'}")
    else:
        for method in methods:
            cmd = [
                "python",
                "code/distribution_adv_tgt.py",
                "--model_path",
                args.model_path,
                "--clip_model_path",
                args.clip_model_path,
                "--image_root",
                args.image_root,
                "--image_dirname",
                args.image_dirname,
                "--output_root",
                args.output_root,
                "--exp_id",
                args.exp_id,
                "--method",
                method,
                "--stage",
                "A",
                "--dataset_type",
                args.dataset_type,
                "--task_type",
                args.task_type,
                "--finetune_method",
                "LoRA",
                "--src_model",
                args.src_model,
                "--tgt_model",
                args.tgt_model,
                "--budget_l2_grid",
                args.budget_l2_grid,
                "--budget_lpips_grid",
                args.budget_lpips_grid,
                "--iters",
                str(args.protect_iters),
                "--initial_lr",
                str(args.protect_initial_lr),
                "--eps",
                str(args.protect_eps),
                "--batch_size",
                str(args.protect_batch_size),
                "--num_workers",
                str(args.protect_num_workers),
                "--mixed_precision",
                args.protect_mixed_precision,
                "--seed",
                str(args.seed),
                "--log_every",
                str(args.protect_log_every),
                "--nan_lr_decay",
                str(args.protect_nan_lr_decay),
                "--nan_min_lr",
                str(args.protect_nan_min_lr),
                "--nan_max_recoveries",
                str(args.protect_nan_max_recoveries),
                "--show_progress" if args.show_protect_progress else "--no-show_progress",
                "--rewrite" if args.rewrite_protect else "--no-rewrite",
            ]
            if method == "mist":
                cmd.extend(["--target_image_path", str(args.mist_target_image_path)])
            if method == "glaze":
                cmd.extend(
                    [
                        "--glaze_style_backend",
                        str(args.glaze_style_backend),
                        "--glaze_style_onnx_path",
                        str(args.glaze_style_onnx_path),
                        "--glaze_style_image_path",
                        str(args.glaze_style_image_path or ""),
                        "--glaze_style_alpha",
                        str(args.glaze_style_alpha),
                    ]
                )
            try:
                run_pipeline_cmd(cmd)
            except Exception as err:
                run_errors.append(f"protect:{method}:{err}")
                if args.stop_on_error:
                    raise

    # Step 2. Resolve run list from manifest.
    manifest_path = exp_root / "manifest.csv"
    if manifest_path.exists():
        manifest_rows = read_manifest(manifest_path)
        manifest_rows = [
            row
            for row in manifest_rows
            if row.get("method") in methods and row.get("dataset_name") == args.image_dirname
        ]
    elif args.dry_run:
        budget_l2_values = parse_budget_grid(args.budget_l2_grid)
        budget_lpips_values = parse_budget_grid(args.budget_lpips_grid)
        manifest_rows = []
        for method in methods:
            for b2 in budget_l2_values:
                for bp in budget_lpips_values:
                    manifest_rows.append(
                        {
                            "method": method,
                            "budget_l2": str(b2),
                            "budget_lpips": str(bp),
                            "dataset_name": args.image_dirname,
                        }
                    )
    else:
        raise FileNotFoundError(
            f"manifest not found after protection stage: {manifest_path}. "
            "Check protection logs."
        )
    run_id_to_manifest = {}
    for row in manifest_rows:
        run_id_to_manifest[run_id_from_manifest_row(row)] = row
    run_ids = sorted(run_id_to_manifest.keys())
    if not run_ids:
        raise RuntimeError(f"No run ids found in manifest after protection: {manifest_path}")

    # Step 3. Prepare clean dataset metadata.
    validation_prompt = args.validation_prompt or args.image_dirname
    _, clean_count = ensure_metadata_jsonl(
        dataset_dir=dataset_dir,
        dataset_token=args.image_dirname,
        caption_template=args.caption_template,
        overwrite=args.overwrite_metadata,
        dry_run=args.dry_run,
    )
    print(f"[Dataset] clean metadata ready, images={clean_count} dir={dataset_dir}")

    # Step 4. Train clean LoRA once.
    clean_lora_dir = exp_root / "ft_clean" / "lora_clean"
    clean_weight = get_lora_weight_path(clean_lora_dir)
    if args.overwrite_lora or not clean_weight.exists():
        clean_lora_dir.mkdir(parents=True, exist_ok=True)
        clean_cmd = build_lora_train_cmd(
            args=args,
            train_data_dir=dataset_dir,
            output_dir=clean_lora_dir,
            validation_prompt=validation_prompt,
            seed=args.seed,
        )
        run_pipeline_cmd(clean_cmd)
    else:
        print(f"[Skip] clean LoRA exists: {clean_weight}")

    # Step 5. Generate clean images once.
    clean_generate_root = exp_root / "generated_clean"
    clean_manifest = clean_generate_root / clean_lora_dir.name / "manifest.csv"
    clean_gen_cmd = build_generate_cmd(
        args=args,
        output_root=clean_generate_root,
        lora_path=clean_lora_dir,
        prompt_file=prompt_run_file,
        seed_file=seed_run_file,
    )
    if args.overwrite_generate:
        run_pipeline_cmd(clean_gen_cmd)
    elif manifest_is_complete(clean_manifest, expected_count=expected_samples_per_adapter):
        print(f"[Skip] clean generation complete: {clean_manifest}")
    else:
        run_pipeline_cmd(clean_gen_cmd)

    # Step 6. For each run: train adv LoRA, generate adv images, evaluate conclusion metrics.
    for run_id in run_ids:
        row = run_id_to_manifest[run_id]
        method = row["method"]
        protect_dir = exp_root / "protected_images" / method / run_id / args.image_dirname
        if not protect_dir.exists() and not args.dry_run:
            run_errors.append(f"{run_id}:protect_dir_missing:{protect_dir}")
            if args.stop_on_error:
                raise FileNotFoundError(f"Protect dir missing for {run_id}: {protect_dir}")
            continue

        if not args.dry_run:
            ensure_metadata_jsonl(
                dataset_dir=protect_dir,
                dataset_token=args.image_dirname,
                caption_template=args.caption_template,
                overwrite=args.overwrite_metadata,
                dry_run=args.dry_run,
            )

        adv_lora_dir = exp_root / "ft_adv" / run_id
        adv_weight = get_lora_weight_path(adv_lora_dir)
        if args.overwrite_lora or args.dry_run or not adv_weight.exists():
            adv_lora_dir.mkdir(parents=True, exist_ok=True)
            adv_seed = args.seed + abs(hash(run_id)) % 100000
            adv_cmd = build_lora_train_cmd(
                args=args,
                train_data_dir=protect_dir,
                output_dir=adv_lora_dir,
                validation_prompt=validation_prompt,
                seed=adv_seed,
            )
            try:
                run_pipeline_cmd(adv_cmd)
            except Exception as err:
                run_errors.append(f"{run_id}:adv_train:{err}")
                if args.stop_on_error:
                    raise
                continue
        elif not args.dry_run:
            print(f"[Skip] adv LoRA exists: {adv_weight}")

        adv_generate_root = exp_root / "generated_adv" / run_id
        adv_manifest = adv_generate_root / adv_lora_dir.name / "manifest.csv"
        adv_gen_cmd = build_generate_cmd(
            args=args,
            output_root=adv_generate_root,
            lora_path=adv_lora_dir,
            prompt_file=prompt_run_file,
            seed_file=seed_run_file,
        )
        if args.overwrite_generate:
            should_run_adv_generate = True
        else:
            should_run_adv_generate = not manifest_is_complete(
                adv_manifest,
                expected_count=expected_samples_per_adapter,
            )

        if should_run_adv_generate:
            try:
                run_pipeline_cmd(adv_gen_cmd)
            except Exception as err:
                run_errors.append(f"{run_id}:adv_generate:{err}")
                if args.stop_on_error:
                    raise
                continue
        else:
            print(f"[Skip] adv generation complete: {adv_manifest}")

        metrics_output_dir = exp_root / "metrics" / run_id / "conclusion"
        protect_metrics_json = exp_root / "metrics" / run_id / "metrics.json"
        conclusion_json = metrics_output_dir / "metrics.json"
        metrics_cmd = [
            "python",
            "experiments/scripts/run_conclusion_metrics.py",
            "--clean_manifest",
            str(clean_manifest),
            "--adv_manifest",
            str(adv_manifest),
            "--reference_dir",
            str(dataset_dir),
            "--output_dir",
            str(metrics_output_dir),
            "--clip_model_path",
            args.clip_model_path,
            "--batch_size",
            str(args.metrics_batch_size),
            "--protect_metrics_json",
            str(protect_metrics_json),
            "--compute_fid" if args.compute_fid else "--no-compute_fid",
            "--strict" if args.strict_metrics else "--no-strict",
        ]
        if args.overwrite_metrics or not conclusion_json.exists():
            try:
                run_pipeline_cmd(metrics_cmd)
            except Exception as err:
                run_errors.append(f"{run_id}:metrics:{err}")
                if args.stop_on_error:
                    raise
                continue
        else:
            print(f"[Skip] metrics exists: {conclusion_json}")

        # Collect summary row.
        if not args.dry_run:
            if conclusion_json.exists():
                conclusion = load_conclusion_metrics(conclusion_json)
                summary_rows.append(
                    {
                        "run_id": run_id,
                        "method": method,
                        "budget_l2": row["budget_l2"],
                        "budget_lpips": row["budget_lpips"],
                        "num_pairs": conclusion.get("num_pairs"),
                        "pg": conclusion.get("pg"),
                        "pgg_lpips_mean": (conclusion.get("pgg_lpips") or {}).get("mean"),
                        "pgg_clip_mean": (conclusion.get("pgg_clip") or {}).get("mean"),
                        "clip_t_clean_mean": (conclusion.get("clip_t_clean") or {}).get("mean"),
                        "clip_t_adv_mean": (conclusion.get("clip_t_adv") or {}).get("mean"),
                        "qrr_clip_t": conclusion.get("qrr_clip_t"),
                        "fid_clean": conclusion.get("fid_clean"),
                        "fid_adv": conclusion.get("fid_adv"),
                        "delta_fid": conclusion.get("delta_fid"),
                    }
                )

    # Step 7. Save global summary.
    elapsed = time.time() - start_time
    pipeline_status = {
        "exp_id": args.exp_id,
        "elapsed_sec": elapsed,
        "run_count": len(run_ids),
        "summary_count": len(summary_rows),
        "errors": run_errors,
        "dry_run": args.dry_run,
    }
    (exp_root / "pipeline_status.json").write_text(json.dumps(pipeline_status, indent=2, ensure_ascii=False), encoding="utf-8")
    if summary_rows:
        write_summary(summary_rows, exp_root / "analysis" / "conclusion_summary.csv")
        (exp_root / "analysis" / "conclusion_summary.json").write_text(
            json.dumps(summary_rows, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    if run_errors:
        print("[Pipeline] completed with errors:")
        for err in run_errors:
            print(f"  - {err}")
        if args.stop_on_error:
            raise SystemExit("Pipeline terminated with errors.")
    else:
        print(f"[Pipeline] success. exp_root={exp_root}")


if __name__ == "__main__":
    main()
