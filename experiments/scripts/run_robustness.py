#!/usr/bin/env python3
import argparse
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

VALID_IMAGE_SUFFIX = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_float_or_fraction(text):
    value = str(text).strip()
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        return float(numerator) / float(denominator)
    return float(value)


def parse_list(text, parse_fn):
    values = []
    for token in text.split(","):
        token = token.strip()
        if token:
            values.append(parse_fn(token))
    return values


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--jpeg_qualities", type=str, default="95,75,50")
    parser.add_argument("--resize_ratios", type=str, default="1.0,0.75,0.5")
    parser.add_argument("--crop_ratios", type=str, default="0,0.05,0.1")
    parser.add_argument("--noise_sigmas", type=str, default="1/255,2/255,4/255")
    return parser.parse_args()


def collect_images(input_dir):
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"Input directory not found: {root}")
    files = []
    for path in root.rglob("*"):
        if path.suffix.lower() in VALID_IMAGE_SUFFIX:
            files.append(path)
    return files


def jpeg_transform(image, quality):
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=int(quality))
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def resize_transform(image, ratio):
    w, h = image.size
    new_w = max(1, int(round(w * ratio)))
    new_h = max(1, int(round(h * ratio)))
    small = image.resize((new_w, new_h), resample=Image.BICUBIC)
    return small.resize((w, h), resample=Image.BICUBIC)


def crop_transform(image, ratio):
    if ratio <= 0:
        return image
    w, h = image.size
    crop_w = max(1, int(round(w * (1.0 - ratio))))
    crop_h = max(1, int(round(h * (1.0 - ratio))))
    left = (w - crop_w) // 2
    top = (h - crop_h) // 2
    cropped = image.crop((left, top, left + crop_w, top + crop_h))
    return cropped.resize((w, h), resample=Image.BICUBIC)


def noise_transform(image, sigma):
    arr = np.asarray(image).astype(np.float32) / 255.0
    noise = np.random.normal(0.0, sigma, arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0.0, 1.0)
    arr = (arr * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)


def save_variant(variant_img, output_root, transform_name, severity_tag, rel_path):
    out_path = output_root / transform_name / severity_tag / rel_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    variant_img.save(out_path)


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jpeg_qualities = parse_list(args.jpeg_qualities, int)
    resize_ratios = parse_list(args.resize_ratios, float)
    crop_ratios = parse_list(args.crop_ratios, float)
    noise_sigmas = parse_list(args.noise_sigmas, parse_float_or_fraction)

    images = collect_images(input_dir)
    if not images:
        raise RuntimeError(f"No images found in {input_dir}")

    for image_path in images:
        rel = image_path.relative_to(input_dir)
        image = Image.open(image_path).convert("RGB")

        for quality in jpeg_qualities:
            transformed = jpeg_transform(image, quality)
            save_variant(transformed, output_dir, "jpeg", f"q{quality}", rel)

        for ratio in resize_ratios:
            transformed = resize_transform(image, ratio)
            ratio_tag = f"r{ratio:.3f}".rstrip("0").rstrip(".").replace(".", "p")
            save_variant(transformed, output_dir, "resize", ratio_tag, rel)

        for ratio in crop_ratios:
            transformed = crop_transform(image, ratio)
            ratio_tag = f"c{ratio:.3f}".rstrip("0").rstrip(".").replace(".", "p")
            save_variant(transformed, output_dir, "crop", ratio_tag, rel)

        for sigma in noise_sigmas:
            transformed = noise_transform(image, sigma)
            sigma_tag = f"s{sigma:.6f}".rstrip("0").rstrip(".").replace(".", "p")
            save_variant(transformed, output_dir, "gaussian_noise", sigma_tag, rel)

    print(
        f"[Done] transformed {len(images)} images "
        f"-> {output_dir} with jpeg/resize/crop/gaussian_noise variants"
    )


if __name__ == "__main__":
    main()
