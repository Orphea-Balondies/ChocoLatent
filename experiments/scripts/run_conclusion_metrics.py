#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import lpips as LPIPS
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

VALID_IMAGE_SUFFIX = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def clip_features_to_tensor(features: Any) -> torch.Tensor:
    """Normalize CLIP feature outputs across transformers versions."""
    if torch.is_tensor(features):
        return features

    pooler_output = getattr(features, "pooler_output", None)
    if torch.is_tensor(pooler_output):
        return pooler_output

    if isinstance(features, tuple):
        if len(features) > 1 and torch.is_tensor(features[1]):
            return features[1]
        if len(features) > 0 and torch.is_tensor(features[0]):
            return features[0]

    raise TypeError(f"Unsupported CLIP feature output type: {type(features)!r}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_manifest", type=str, required=True)
    parser.add_argument("--adv_manifest", type=str, required=True)
    parser.add_argument("--reference_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--clip_model_path", type=str, default="model/clip-vit-base-patch32")
    parser.add_argument("--lpips_net", type=str, default="alex")
    parser.add_argument("--protect_metrics_json", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--compute_fid", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def load_csv(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict], fieldnames: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_reference_images(reference_dir: Path) -> List[Path]:
    images = []
    for path in reference_dir.rglob("*"):
        if path.suffix.lower() in VALID_IMAGE_SUFFIX:
            images.append(path)
    images.sort()
    return images


def load_image_pil(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def image_tensor_from_pil(image: Image.Image, size: Tuple[int, int] = (512, 512)) -> torch.Tensor:
    if size is not None:
        image = image.resize(size, Image.BICUBIC)
    arr = np.asarray(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return tensor * 2.0 - 1.0


def summarize(values: List[float]) -> Dict[str, float]:
    x = np.array(values, dtype=np.float64)
    if x.size == 0:
        return {"mean": None, "std": None, "ci95": None}
    mean = float(np.mean(x))
    std = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    ci95 = float(1.96 * std / np.sqrt(x.size)) if x.size > 1 else 0.0
    return {"mean": mean, "std": std, "ci95": ci95}


def maybe_compute_fid(generated_paths, reference_paths, device, batch_size):
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except Exception:
        return None, "torchmetrics.image.fid unavailable"

    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)

    def update_with_paths(paths, real_flag):
        batch = []
        for image_path in paths:
            image = Image.open(image_path).convert("RGB")
            arr = np.asarray(image).astype(np.uint8)
            tensor = torch.from_numpy(arr).permute(2, 0, 1)
            batch.append(tensor)
            if len(batch) >= batch_size:
                fid.update(torch.stack(batch).to(device), real=real_flag)
                batch = []
        if batch:
            fid.update(torch.stack(batch).to(device), real=real_flag)

    update_with_paths(reference_paths, real_flag=True)
    update_with_paths(generated_paths, real_flag=False)
    return float(fid.compute().item()), None


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_dtype = torch.float16 if device == "cuda" else torch.float32

    clean_rows = load_csv(Path(args.clean_manifest))
    adv_rows = load_csv(Path(args.adv_manifest))
    clean_map = {row["sample_id"]: row for row in clean_rows}
    adv_map = {row["sample_id"]: row for row in adv_rows}

    common_ids = sorted(set(clean_map.keys()) & set(adv_map.keys()))
    missing_in_adv = sorted(set(clean_map.keys()) - set(adv_map.keys()))
    missing_in_clean = sorted(set(adv_map.keys()) - set(clean_map.keys()))

    if not common_ids:
        raise RuntimeError("No overlapping sample_id between clean and adv manifests.")

    reference_paths = collect_reference_images(Path(args.reference_dir))
    if not reference_paths:
        raise RuntimeError(f"No reference images found under {args.reference_dir}")

    lpips_model = LPIPS.LPIPS(net=args.lpips_net).to(device).eval()
    for p in lpips_model.parameters():
        p.requires_grad_(False)

    clip_processor = CLIPProcessor.from_pretrained(args.clip_model_path)
    clip_model = CLIPModel.from_pretrained(args.clip_model_path).to(device=device, dtype=model_dtype).eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)

    # Reference embeddings for MLS.
    ref_embeddings_batches = []
    for start in range(0, len(reference_paths), args.batch_size):
        batch_paths = reference_paths[start : start + args.batch_size]
        pil_images = [Image.open(p).convert("RGB") for p in batch_paths]
        clip_inputs = clip_processor(images=pil_images, return_tensors="pt").to(device)
        with torch.no_grad():
            feats = clip_features_to_tensor(clip_model.get_image_features(**clip_inputs)).float()
            feats = F.normalize(feats, dim=-1)
        ref_embeddings_batches.append(feats.cpu())
    ref_embeddings = torch.cat(ref_embeddings_batches, dim=0)  # [R, D]

    # Pre-compute prompt text embeddings.
    prompt_set = sorted({clean_map[sid]["prompt"] for sid in common_ids})
    prompt_to_embedding = {}
    for start in range(0, len(prompt_set), args.batch_size):
        batch_prompts = prompt_set[start : start + args.batch_size]
        text_inputs = clip_processor(text=batch_prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            text_feats = clip_features_to_tensor(clip_model.get_text_features(**text_inputs)).float()
            text_feats = F.normalize(text_feats, dim=-1)
        for prompt, embedding in zip(batch_prompts, text_feats):
            prompt_to_embedding[prompt] = embedding.cpu()

    per_pair_rows = []
    pgg_lpips_values = []
    pgg_clip_values = []
    mls_clean_values = []
    mls_adv_values = []
    clip_t_clean_values = []
    clip_t_adv_values = []

    clean_img_paths = []
    adv_img_paths = []

    for start in range(0, len(common_ids), args.batch_size):
        batch_ids = common_ids[start : start + args.batch_size]
        clean_pils = []
        adv_pils = []
        clean_tensors = []
        adv_tensors = []
        prompts = []

        for sample_id in batch_ids:
            clean_row = clean_map[sample_id]
            adv_row = adv_map[sample_id]
            clean_path = clean_row["image_path"]
            adv_path = adv_row["image_path"]

            clean_pil = load_image_pil(clean_path)
            adv_pil = load_image_pil(adv_path)
            clean_pils.append(clean_pil)
            adv_pils.append(adv_pil)
            clean_tensors.append(image_tensor_from_pil(clean_pil))
            adv_tensors.append(image_tensor_from_pil(adv_pil))
            prompts.append(clean_row["prompt"])

            clean_img_paths.append(clean_path)
            adv_img_paths.append(adv_path)

        clean_tensor_batch = torch.stack(clean_tensors).to(device)
        adv_tensor_batch = torch.stack(adv_tensors).to(device)
        with torch.no_grad():
            pgg_lpips_batch = lpips_model(clean_tensor_batch.float(), adv_tensor_batch.float()).flatten().cpu()

        clean_clip_inputs = clip_processor(images=clean_pils, return_tensors="pt").to(device)
        adv_clip_inputs = clip_processor(images=adv_pils, return_tensors="pt").to(device)
        with torch.no_grad():
            clean_feats = clip_features_to_tensor(clip_model.get_image_features(**clean_clip_inputs)).float()
            adv_feats = clip_features_to_tensor(clip_model.get_image_features(**adv_clip_inputs)).float()
            clean_feats = F.normalize(clean_feats, dim=-1)
            adv_feats = F.normalize(adv_feats, dim=-1)
            pgg_clip_batch = (1.0 - F.cosine_similarity(clean_feats, adv_feats, dim=-1)).cpu()

        # MLS and CLIP-T
        clean_feats_cpu = clean_feats.cpu()
        adv_feats_cpu = adv_feats.cpu()
        mls_clean_batch = torch.max(clean_feats_cpu @ ref_embeddings.T, dim=1).values
        mls_adv_batch = torch.max(adv_feats_cpu @ ref_embeddings.T, dim=1).values

        text_feat_batch = torch.stack([prompt_to_embedding[p] for p in prompts], dim=0)
        clip_t_clean_batch = F.cosine_similarity(clean_feats_cpu, text_feat_batch, dim=-1)
        clip_t_adv_batch = F.cosine_similarity(adv_feats_cpu, text_feat_batch, dim=-1)

        for idx, sample_id in enumerate(batch_ids):
            clean_row = clean_map[sample_id]
            adv_row = adv_map[sample_id]
            per_pair_rows.append(
                {
                    "sample_id": sample_id,
                    "prompt_idx": clean_row.get("prompt_idx", ""),
                    "prompt": clean_row["prompt"],
                    "seed": clean_row.get("seed", ""),
                    "clean_image_path": clean_row["image_path"],
                    "adv_image_path": adv_row["image_path"],
                    "pgg_lpips": float(pgg_lpips_batch[idx].item()),
                    "pgg_clip": float(pgg_clip_batch[idx].item()),
                    "mls_clean": float(mls_clean_batch[idx].item()),
                    "mls_adv": float(mls_adv_batch[idx].item()),
                    "clip_t_clean": float(clip_t_clean_batch[idx].item()),
                    "clip_t_adv": float(clip_t_adv_batch[idx].item()),
                }
            )

            pgg_lpips_values.append(float(pgg_lpips_batch[idx].item()))
            pgg_clip_values.append(float(pgg_clip_batch[idx].item()))
            mls_clean_values.append(float(mls_clean_batch[idx].item()))
            mls_adv_values.append(float(mls_adv_batch[idx].item()))
            clip_t_clean_values.append(float(clip_t_clean_batch[idx].item()))
            clip_t_adv_values.append(float(clip_t_adv_batch[idx].item()))

    # Optional FID
    fid_clean, fid_clean_err = (None, None)
    fid_adv, fid_adv_err = (None, None)
    if args.compute_fid:
        fid_clean, fid_clean_err = maybe_compute_fid(clean_img_paths, reference_paths, device, args.batch_size)
        fid_adv, fid_adv_err = maybe_compute_fid(adv_img_paths, reference_paths, device, args.batch_size)
        if args.strict and (fid_clean is None or fid_adv is None):
            raise RuntimeError(
                "FID requested but unavailable. "
                f"clean_error={fid_clean_err}, adv_error={fid_adv_err}"
            )

    summary = {
        "num_pairs": len(common_ids),
        "missing_in_adv_count": len(missing_in_adv),
        "missing_in_clean_count": len(missing_in_clean),
        "missing_in_adv_examples": missing_in_adv[:20],
        "missing_in_clean_examples": missing_in_clean[:20],
        "pgg_lpips": summarize(pgg_lpips_values),
        "pgg_clip": summarize(pgg_clip_values),
        "mls_clean": summarize(mls_clean_values),
        "mls_adv": summarize(mls_adv_values),
        "pg": float(np.mean(mls_clean_values) - np.mean(mls_adv_values)),
        "clip_t_clean": summarize(clip_t_clean_values),
        "clip_t_adv": summarize(clip_t_adv_values),
        "qrr_clip_t": float(np.mean(clip_t_adv_values) / np.mean(clip_t_clean_values))
        if np.mean(clip_t_clean_values) != 0
        else None,
        "fid_clean": fid_clean,
        "fid_adv": fid_adv,
        "delta_fid": (fid_adv - fid_clean) if (fid_clean is not None and fid_adv is not None) else None,
        "fid_clean_error": fid_clean_err,
        "fid_adv_error": fid_adv_err,
        "metric_version": "latent-protocol-v0.2",
    }
    summary["metric_coverage"] = {
        "implemented": [
            "PGG_lpips",
            "PGG_clip",
            "MLS_clean",
            "MLS_adv",
            "PG",
            "CLIP-T_clean",
            "CLIP-T_adv",
            "QRR_clip_t",
            "InputDistortionSummary(from protect metrics.json)",
            "FID(optional)",
        ],
        "not_computed_without_extra_assets": {
            "Diffusion-CLS-Acc@Top3": "Requires a trained style/diffusion classifier checkpoint.",
            "Edit-SSIM/Edit-PSNR/Edit-VIFp": "Requires edited_clean and edited_adv image sets.",
            "TR(transfer rate)": "Requires cross-model run pairs (e.g., 1.5->XL, XL->1.5).",
            "SR_T/AUC(robustness curves)": "Requires transformed data experiments (JPEG/resize/crop/noise) and paired PG.",
            "MUSIQ_or_NIQE": "Requires additional IQA backend/model integration.",
            "Holm-Bonferroni and effect-size report": "Requires multi-run statistical analysis script over repeated seeds.",
        },
    }

    if args.protect_metrics_json:
        protect_metrics_path = Path(args.protect_metrics_json)
        if protect_metrics_path.exists():
            protect_summary = load_json(protect_metrics_path)
            summary["protect_input_metrics"] = {
                "input_l2_normed_mean": protect_summary.get("input_l2_normed_mean"),
                "input_linf_mean": protect_summary.get("input_linf_mean"),
                "input_lpips_mean": protect_summary.get("input_lpips_mean"),
                "input_psnr_mean": protect_summary.get("input_psnr_mean"),
                "input_ssim_mean": protect_summary.get("input_ssim_mean"),
            }
        elif args.strict:
            raise FileNotFoundError(f"protect_metrics_json not found: {protect_metrics_path}")

    if args.strict:
        gaps = summary["metric_coverage"]["not_computed_without_extra_assets"]
        if gaps:
            raise RuntimeError(
                "Strict mode requires full metric coverage, but some metrics need extra assets/config. "
                f"Missing keys: {list(gaps.keys())}"
            )

    write_csv(
        output_dir / "per_pair.csv",
        per_pair_rows,
        [
            "sample_id",
            "prompt_idx",
            "prompt",
            "seed",
            "clean_image_path",
            "adv_image_path",
            "pgg_lpips",
            "pgg_clip",
            "mls_clean",
            "mls_adv",
            "clip_t_clean",
            "clip_t_adv",
        ],
    )
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    print(
        f"[Done] pairs={summary['num_pairs']} "
        f"PG={summary['pg']:.4f} "
        f"PGG_lpips={summary['pgg_lpips']['mean']:.4f} "
        f"PGG_clip={summary['pgg_clip']['mean']:.4f}"
    )


if __name__ == "__main__":
    main()
