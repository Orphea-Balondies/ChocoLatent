#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import lpips as LPIPS
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPVisionModel

VALID_IMAGE_SUFFIX = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated_clean_dir", type=str, required=True)
    parser.add_argument("--generated_adv_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--clip_model_path", type=str, default="model/clip-vit-base-patch32")
    parser.add_argument("--lpips_net", type=str, default="alex")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def collect_images(image_root):
    root = Path(image_root)
    if not root.exists():
        raise FileNotFoundError(f"Directory does not exist: {root}")
    image_map = {}
    for file_path in root.rglob("*"):
        if file_path.suffix.lower() not in VALID_IMAGE_SUFFIX:
            continue
        rel = str(file_path.relative_to(root))
        image_map[rel] = file_path
    return image_map


def load_image_tensor(path, device):
    image = Image.open(path).convert("RGB")
    arr = np.asarray(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device=device)
    return tensor * 2.0 - 1.0


def clip_preprocess(x):
    x_01 = (x / 2 + 0.5).clamp(0, 1)
    x_224 = F.interpolate(x_01, size=(224, 224), mode="bilinear", align_corners=False)
    mean = CLIP_MEAN.to(device=x.device, dtype=x.dtype)
    std = CLIP_STD.to(device=x.device, dtype=x.dtype)
    return (x_224 - mean) / std


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_dtype = torch.float16 if device == "cuda" else torch.float32

    clean_map = collect_images(args.generated_clean_dir)
    adv_map = collect_images(args.generated_adv_dir)

    clean_keys = set(clean_map.keys())
    adv_keys = set(adv_map.keys())
    common_keys = sorted(clean_keys & adv_keys)
    only_clean = sorted(clean_keys - adv_keys)
    only_adv = sorted(adv_keys - clean_keys)

    if not common_keys:
        raise RuntimeError("No paired images found between generated_clean_dir and generated_adv_dir.")

    lpips_model = LPIPS.LPIPS(net=args.lpips_net).to(device).eval()
    for p in lpips_model.parameters():
        p.requires_grad_(False)

    clip_model = CLIPVisionModel.from_pretrained(args.clip_model_path).to(device=device, dtype=model_dtype).eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)

    per_image_rows = []
    for rel_path in common_keys:
        clean_path = clean_map[rel_path]
        adv_path = adv_map[rel_path]

        x_clean = load_image_tensor(clean_path, device=device)
        x_adv = load_image_tensor(adv_path, device=device)
        if x_clean.shape != x_adv.shape:
            x_adv = F.interpolate(x_adv, size=x_clean.shape[-2:], mode="bilinear", align_corners=False)

        with torch.no_grad():
            pgg_lpips = float(lpips_model(x_clean.float(), x_adv.float()).flatten().mean().item())
            clip_clean = clip_model(clip_preprocess(x_clean).to(dtype=model_dtype)).pooler_output.float()
            clip_adv = clip_model(clip_preprocess(x_adv).to(dtype=model_dtype)).pooler_output.float()
            pgg_clip = float((1.0 - F.cosine_similarity(clip_clean, clip_adv, dim=1)).mean().item())

        per_image_rows.append(
            {
                "rel_path": rel_path,
                "clean_path": str(clean_path),
                "adv_path": str(adv_path),
                "pgg_lpips": pgg_lpips,
                "pgg_clip": pgg_clip,
            }
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    write_csv(
        output_dir / "per_image.csv",
        per_image_rows,
        ["rel_path", "clean_path", "adv_path", "pgg_lpips", "pgg_clip"],
    )

    lpips_values = np.array([row["pgg_lpips"] for row in per_image_rows], dtype=np.float64)
    clip_values = np.array([row["pgg_clip"] for row in per_image_rows], dtype=np.float64)
    summary = {
        "num_pairs": len(per_image_rows),
        "pgg_lpips_mean": float(lpips_values.mean()),
        "pgg_lpips_std": float(lpips_values.std(ddof=1)) if len(lpips_values) > 1 else 0.0,
        "pgg_clip_mean": float(clip_values.mean()),
        "pgg_clip_std": float(clip_values.std(ddof=1)) if len(clip_values) > 1 else 0.0,
        "missing_in_adv_count": len(only_clean),
        "missing_in_clean_count": len(only_adv),
        "missing_in_adv_examples": only_clean[:20],
        "missing_in_clean_examples": only_adv[:20],
        "metric_version": "latent-protocol-v0.2",
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    config_payload = {
        "generated_clean_dir": str(Path(args.generated_clean_dir).resolve()),
        "generated_adv_dir": str(Path(args.generated_adv_dir).resolve()),
        "clip_model_path": args.clip_model_path,
        "lpips_net": args.lpips_net,
        "device": device,
    }
    with open(output_dir / "config.yaml", "w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2, ensure_ascii=False)

    print(
        f"[Done] num_pairs={summary['num_pairs']} "
        f"PGG_lpips={summary['pgg_lpips_mean']:.4f} "
        f"PGG_clip={summary['pgg_clip_mean']:.4f}"
    )


if __name__ == "__main__":
    main()
