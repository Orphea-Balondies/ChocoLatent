import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import torch
from diffusers import StableDiffusionPipeline


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="model/stable-diffusion-v1-5")
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--loras", type=str, nargs="*", default=None)

    # Prompt / seed settings
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--prompt_file", type=str, default=None, help="txt (one prompt per line) or json list.")
    parser.add_argument("--seeds", type=str, default=None, help="Comma separated seeds, e.g. 1,2,3")
    parser.add_argument("--seed_file", type=str, default=None, help="txt/json list of seeds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used when sampling seeds.")
    parser.add_argument("--generate_num", type=int, default=10, help="Used only when explicit seeds are not given.")

    # Generation parameters
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--negative_prompt", type=str, default=None)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)

    # Adapter settings
    parser.add_argument("--include_clean", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--clean_name", type=str, default="clean")

    # Runtime
    parser.add_argument("--device", type=str, default=None, help="cuda/cpu, auto if unset.")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    return parser.parse_args()


def read_text_lines(path: Path) -> List[str]:
    lines = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)
    return lines


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_prompts(args) -> List[str]:
    prompts: List[str] = []
    if args.prompt_file:
        prompt_path = Path(args.prompt_file)
        if prompt_path.suffix.lower() == ".json":
            payload = read_json(prompt_path)
            if not isinstance(payload, list):
                raise ValueError("prompt_file json must be a list of prompt strings.")
            prompts.extend(str(x).strip() for x in payload if str(x).strip())
        else:
            prompts.extend(read_text_lines(prompt_path))

    if args.prompt:
        prompts.append(args.prompt.strip())

    if not prompts:
        if args.loras:
            prompts = [Path(args.loras[0]).parent.name]
        else:
            prompts = ["a high quality image"]

    # Stable order + dedup
    dedup = []
    seen = set()
    for prompt in prompts:
        if prompt not in seen:
            seen.add(prompt)
            dedup.append(prompt)
    return dedup


def load_seeds(args) -> List[int]:
    seeds: List[int] = []
    if args.seed_file:
        seed_path = Path(args.seed_file)
        if seed_path.suffix.lower() == ".json":
            payload = read_json(seed_path)
            if not isinstance(payload, list):
                raise ValueError("seed_file json must be a list of integers.")
            seeds.extend(int(x) for x in payload)
        else:
            seeds.extend(int(x) for x in read_text_lines(seed_path))
    elif args.seeds:
        seeds.extend(int(x.strip()) for x in args.seeds.split(",") if x.strip())
    else:
        rng = random.Random(args.seed)
        seeds = [rng.randint(0, 2**32 - 1) for _ in range(int(args.generate_num))]

    if not seeds:
        raise ValueError("No seeds provided.")

    # Stable order + dedup
    dedup = []
    seen = set()
    for seed in seeds:
        if seed not in seen:
            seen.add(seed)
            dedup.append(seed)
    return dedup


def model_dtype(dtype_str, device):
    if device != "cuda":
        return torch.float32
    if dtype_str == "bf16":
        return torch.bfloat16
    if dtype_str == "fp32":
        return torch.float32
    return torch.float16


def parse_adapters(args) -> List[Dict]:
    adapters: List[Dict] = []
    if args.include_clean:
        adapters.append({"name": args.clean_name, "path": None})

    for lora_entry in (args.loras or []):
        p = Path(lora_entry)
        if not p.exists():
            raise FileNotFoundError(f"LoRA path not found: {p}")
        if p.is_dir():
            name = p.name
        else:
            name = p.stem
        adapters.append({"name": name, "path": str(p)})

    if not adapters:
        raise ValueError("No adapter to generate. Use --include_clean and/or --loras.")
    return adapters


def resolve_lora_location(adapter_path: Path):
    if adapter_path.is_file():
        return str(adapter_path.parent), adapter_path.name

    safetensors_files = sorted(adapter_path.glob("*.safetensors"))
    if len(safetensors_files) == 1:
        return str(adapter_path), safetensors_files[0].name
    return str(adapter_path), None


def load_adapter(pipe, adapter_path: Optional[str]):
    # Always clear previous adapter to avoid unintended composition.
    if hasattr(pipe, "unload_lora_weights"):
        try:
            pipe.unload_lora_weights()
        except Exception:
            pass

    if adapter_path is None:
        return

    adapter_path_obj = Path(adapter_path)
    model_dir, weight_name = resolve_lora_location(adapter_path_obj)
    if weight_name is None:
        pipe.load_lora_weights(model_dir)
    else:
        pipe.load_lora_weights(model_dir, weight_name=weight_name)


def write_csv(path: Path, rows: List[Dict], fieldnames: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = model_dtype(args.dtype, device=device)
    prompts = load_prompts(args)
    seeds = load_seeds(args)
    adapters = parse_adapters(args)

    output_root = (
        Path(args.output_root)
        if args.output_root
        else Path(args.loras[-1]).parent.parent / "generated_images"
        if args.loras
        else Path("generated_images")
    )
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"[Init] Loading base model from {args.model_path}")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    global_manifest = {
        "model_path": args.model_path,
        "output_root": str(output_root.resolve()),
        "device": device,
        "dtype": args.dtype,
        "num_prompts": len(prompts),
        "num_seeds": len(seeds),
        "prompts": prompts,
        "seeds": seeds,
        "adapters": adapters,
        "num_total_images": len(prompts) * len(seeds) * len(adapters),
    }
    with (output_root / "generation_plan.json").open("w", encoding="utf-8") as f:
        json.dump(global_manifest, f, indent=2, ensure_ascii=False)

    for adapter in adapters:
        adapter_name = adapter["name"]
        adapter_path = adapter["path"]
        save_dir = output_root / adapter_name
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"[Adapter] {adapter_name} path={adapter_path}")
        load_adapter(pipe, adapter_path=adapter_path)

        rows = []
        for prompt_idx, prompt in enumerate(prompts):
            for seed in seeds:
                sample_id = f"p{prompt_idx:03d}_s{seed}"
                save_path = save_dir / f"{sample_id}.png"

                if save_path.exists() and not args.overwrite:
                    rows.append(
                        {
                            "sample_id": sample_id,
                            "prompt_idx": prompt_idx,
                            "prompt": prompt,
                            "seed": seed,
                            "adapter_name": adapter_name,
                            "adapter_path": adapter_path or "",
                            "image_path": str(save_path.resolve()),
                            "skipped_existing": 1,
                        }
                    )
                    continue

                generator = torch.Generator(device=device).manual_seed(int(seed))
                image = pipe(
                    prompt=prompt,
                    negative_prompt=args.negative_prompt,
                    guidance_scale=args.guidance_scale,
                    num_inference_steps=args.num_inference_steps,
                    height=args.height,
                    width=args.width,
                    generator=generator,
                ).images[0]
                image.save(save_path)
                print(f"[Saved] {save_path}")

                rows.append(
                    {
                        "sample_id": sample_id,
                        "prompt_idx": prompt_idx,
                        "prompt": prompt,
                        "seed": seed,
                        "adapter_name": adapter_name,
                        "adapter_path": adapter_path or "",
                        "image_path": str(save_path.resolve()),
                        "skipped_existing": 0,
                    }
                )

        write_csv(
            save_dir / "manifest.csv",
            rows,
            [
                "sample_id",
                "prompt_idx",
                "prompt",
                "seed",
                "adapter_name",
                "adapter_path",
                "image_path",
                "skipped_existing",
            ],
        )

    print("[DONE] All generation jobs finished.")


if __name__ == "__main__":
    main()
