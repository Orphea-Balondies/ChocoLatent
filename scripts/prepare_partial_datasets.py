#!/usr/bin/env python3
"""
Prepare partial benchmark subsets for:
  - WikiArt (style-heavy subset)
  - CustomConcept101 (concept subset from official zip)
  - Person (LFW fallback or LAION URL manifest mode)

Design goals:
  - Keep storage bounded.
  - Prioritize "hot" groups by available sample count.
  - Keep each group roughly in [images_min, images_max].
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
import tarfile
import time
import urllib.parse
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pyarrow.parquet as pq
import requests
from PIL import Image, UnidentifiedImageError


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
CUSTOMCONCEPT_GDRIVE_ID = "1jj8JMtIS5-8vRtNtZ2x8isieWH9yetuK"
CUSTOMCONCEPT_DATASET_JSON_URL = (
    "https://raw.githubusercontent.com/adobe-research/custom-diffusion/main/customconcept101/dataset.json"
)
LFW_DEFAULT_URL = "https://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz"
LFW_PARQUET_DEFAULT_URL = (
    "https://huggingface.co/datasets/bitmind/lfw/resolve/main/data/train-00000-of-00001.parquet?download=1"
)
HF_ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"


def slugify(text: str) -> str:
    out = []
    for ch in text.strip().lower():
        if ch.isalnum():
            out.append(ch)
        elif ch in {" ", "-", "_", "+", "."}:
            out.append("_")
    collapsed = "".join(out)
    while "__" in collapsed:
        collapsed = collapsed.replace("__", "_")
    return collapsed.strip("_") or "group"


def clamp_images_per_group(images_per_group: int, images_min: int, images_max: int) -> int:
    return max(images_min, min(images_per_group, images_max))


def is_image_file(path_like: str) -> bool:
    return Path(path_like).suffix.lower() in IMAGE_EXTS


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_image_bytes_as_jpeg(
    image_bytes: bytes,
    out_path: Path,
    max_side: Optional[int],
    jpeg_quality: int,
) -> int:
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img = img.convert("RGB")
            if max_side and max(img.width, img.height) > max_side:
                img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(out_path, format="JPEG", quality=jpeg_quality, optimize=True)
        return out_path.stat().st_size
    except (UnidentifiedImageError, OSError):
        return 0


def check_storage_limit(current_bytes: int, limit_bytes: int) -> None:
    if current_bytes > limit_bytes:
        raise RuntimeError(
            f"Output size exceeded limit: {current_bytes / (1024**3):.2f} GiB > {limit_bytes / (1024**3):.2f} GiB"
        )


def find_wikiart_snapshot(user_path: Optional[Path]) -> Path:
    if user_path:
        if (user_path / "data").exists():
            return user_path
        if user_path.exists():
            return user_path
        raise FileNotFoundError(f"WikiArt snapshot path not found: {user_path}")

    base = Path.home() / ".cache/huggingface/hub/datasets--huggan--wikiart/snapshots"
    if not base.exists():
        raise FileNotFoundError(f"WikiArt snapshots directory not found: {base}")

    snapshots = sorted([p for p in base.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
    if not snapshots:
        raise FileNotFoundError(f"No WikiArt snapshot found under: {base}")
    return snapshots[0]


def list_complete_wikiart_parquets(snapshot: Path) -> List[Path]:
    data_dir = snapshot / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"WikiArt snapshot has no data dir: {data_dir}")

    files: List[Path] = []
    for p in sorted(data_dir.glob("train-*-of-00072.parquet")):
        if p.exists() and p.stat().st_size > 50 * 1024 * 1024:
            files.append(p)
    if not files:
        raise RuntimeError("No complete WikiArt parquet shards found in snapshot.")
    return files


def load_wikiart_label_names(dataset_info_path: Path) -> Tuple[List[str], List[str], List[str]]:
    info = read_json(dataset_info_path)
    first_key = next(iter(info.keys()))
    feat = info[first_key]["features"]
    return feat["style"]["names"], feat["artist"]["names"], feat["genre"]["names"]


def collect_wikiart_value_counts(parquet_files: Sequence[Path], column_name: str) -> Counter:
    counts: Counter = Counter()
    for p in parquet_files:
        t = pq.read_table(str(p), columns=[column_name])
        counts.update(t.column(0).to_pylist())
    return counts


def resolve_group_index(group_name_or_slug: str, label_names: Sequence[str]) -> int:
    raw = group_name_or_slug.strip()
    if not raw:
        raise RuntimeError("focus_group cannot be empty.")
    if raw.isdigit():
        idx = int(raw)
        if 0 <= idx < len(label_names):
            return idx
        raise RuntimeError(f"focus_group index out of range: {idx}")

    target_slug = slugify(raw)
    target_lower = raw.lower()
    for idx, name in enumerate(label_names):
        if name.lower() == target_lower or slugify(name) == target_slug:
            return idx
    raise RuntimeError(f"focus_group not found in labels: {group_name_or_slug}")


def select_top_wikiart_groups(
    value_counts: Counter,
    label_names: Sequence[str],
    target_groups: int,
    images_min: int,
    focus_group_idx: Optional[int],
) -> List[int]:
    if focus_group_idx is not None:
        available = int(value_counts.get(focus_group_idx, 0))
        if available < images_min:
            name = label_names[focus_group_idx]
            raise RuntimeError(
                f"Focused group '{name}' has only {available} images (< images_min={images_min}) in loaded shards."
            )
        return [focus_group_idx]

    selected: List[int] = []
    for group_idx, count in value_counts.most_common():
        if count < images_min:
            continue
        if group_idx < 0 or group_idx >= len(label_names):
            continue
        selected.append(group_idx)
        if len(selected) >= target_groups:
            break
    if not selected:
        raise RuntimeError("No WikiArt groups satisfy the minimum image requirement.")
    return selected


def decode_wikiart_label(idx: int, label_names: Sequence[str]) -> str:
    if 0 <= idx < len(label_names):
        return label_names[idx]
    return str(idx)


def run_wikiart(args: argparse.Namespace) -> None:
    snapshot = find_wikiart_snapshot(Path(args.wikiart_snapshot) if args.wikiart_snapshot else None)
    dataset_info = Path(args.dataset_info)
    parquet_files = list_complete_wikiart_parquets(snapshot)
    style_names, artist_names, genre_names = load_wikiart_label_names(dataset_info)

    if args.group_by == "artist":
        group_column = "artist"
        group_names = artist_names
    else:
        group_column = "style"
        group_names = style_names

    group_counts = collect_wikiart_value_counts(parquet_files, group_column)
    focus_group_idx = resolve_group_index(args.focus_group, group_names) if args.focus_group else None

    target_per_group = clamp_images_per_group(args.images_per_group, args.images_min, args.images_max)
    selected_groups = select_top_wikiart_groups(
        group_counts,
        group_names,
        args.target_groups,
        args.images_min,
        focus_group_idx=focus_group_idx,
    )

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    image_rows = []
    saved_counts: Dict[int, int] = {idx: 0 for idx in selected_groups}
    selected_set = set(selected_groups)
    limit_bytes = int(args.max_output_gb * (1024**3))
    used_bytes = 0

    for shard_path in parquet_files:
        if all(saved_counts[idx] >= target_per_group for idx in selected_groups):
            break

        read_columns = list(dict.fromkeys([group_column, "style", "artist", "genre", "image"]))
        tbl = pq.read_table(str(shard_path), columns=read_columns)
        group_col = tbl[group_column].to_pylist()
        style_col = tbl["style"].to_pylist()
        artist_col = tbl["artist"].to_pylist()
        genre_col = tbl["genre"].to_pylist()
        image_col = tbl["image"].to_pylist()

        for row_idx, group_idx in enumerate(group_col):
            if group_idx not in selected_set:
                continue
            if saved_counts[group_idx] >= target_per_group:
                continue
            item = image_col[row_idx]
            if not isinstance(item, dict) or not item.get("bytes"):
                continue

            group_name = group_names[group_idx]
            group_slug = slugify(group_name)
            out_dir = out_root / group_slug
            out_name = f"{group_slug}_{saved_counts[group_idx] + 1:04d}.jpg"
            out_path = out_dir / out_name

            saved_size = save_image_bytes_as_jpeg(
                item["bytes"],
                out_path=out_path,
                max_side=args.max_side,
                jpeg_quality=args.jpeg_quality,
            )
            if saved_size <= 0:
                continue
            used_bytes += saved_size
            check_storage_limit(used_bytes, limit_bytes)

            artist_idx = artist_col[row_idx]
            genre_idx = genre_col[row_idx]
            style_idx = style_col[row_idx]
            image_rows.append(
                {
                    "dataset": "WikiArt",
                    "group_by": args.group_by,
                    "group": group_name,
                    "artist": decode_wikiart_label(artist_idx, artist_names),
                    "style": decode_wikiart_label(style_idx, style_names),
                    "genre": decode_wikiart_label(genre_idx, genre_names),
                    "source_shard": shard_path.name,
                    "source_row": row_idx,
                    "output_path": str(out_path),
                }
            )
            saved_counts[group_idx] += 1

    for group_idx in selected_groups:
        group_name = group_names[group_idx]
        summary_rows.append(
            {
                "dataset": "WikiArt",
                "group_by": args.group_by,
                "group": group_name,
                "available_in_loaded_shards": int(group_counts[group_idx]),
                "saved_images": int(saved_counts[group_idx]),
                "target_images": target_per_group,
            }
        )

    write_csv(
        out_root / "manifest.csv",
        image_rows,
        ["dataset", "group_by", "group", "artist", "style", "genre", "source_shard", "source_row", "output_path"],
    )
    write_csv(
        out_root / "group_summary.csv",
        summary_rows,
        ["dataset", "group_by", "group", "available_in_loaded_shards", "saved_images", "target_images"],
    )
    write_json(
        out_root / "summary.json",
        {
            "dataset": "WikiArt",
            "group_by": args.group_by,
            "snapshot": str(snapshot),
            "parquet_shards_used": [p.name for p in parquet_files],
            "target_groups": args.target_groups,
            "selected_groups": [group_names[idx] for idx in selected_groups],
            "images_per_group_target": target_per_group,
            "images_saved_total": len(image_rows),
            "output_size_gb": round(used_bytes / (1024**3), 4),
            "focus_group": args.focus_group or None,
            "note": "Groups are ranked by frequency from locally available parquet shards.",
        },
    )
    print(
        f"[WikiArt] group_by={args.group_by} groups={len(selected_groups)} "
        f"saved={len(image_rows)} size={used_bytes / (1024**3):.2f} GiB"
    )


def fetch_or_load_customconcept_dataset_json(dataset_json_path: Optional[Path], dataset_json_url: str) -> List[dict]:
    if dataset_json_path and dataset_json_path.exists():
        data = read_json(dataset_json_path)
        if not isinstance(data, list):
            raise RuntimeError("CustomConcept dataset.json must be a list.")
        return data

    resp = requests.get(dataset_json_url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list):
        raise RuntimeError("Fetched CustomConcept dataset.json is not a list.")
    return data


def maybe_download_customconcept_zip(zip_path: Path, gdrive_id: str) -> None:
    if zip_path.exists() and zip_path.stat().st_size > 1024:
        return
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import gdown  # type: ignore
    except ImportError as e:
        raise RuntimeError("gdown is required to download CustomConcept101 zip.") from e
    url = f"https://drive.google.com/uc?id={gdrive_id}"
    print(f"[Concept] downloading official zip -> {zip_path}")
    gdown.download(url=url, output=str(zip_path), quiet=False)


def map_zip_images_to_instances(image_names: Sequence[str], valid_instances: Sequence[str]) -> Dict[str, List[str]]:
    valid_set = set(valid_instances)
    mapping: Dict[str, List[str]] = defaultdict(list)
    for name in image_names:
        parts = Path(name).parts
        matched = None
        for seg in parts:
            if seg in valid_set:
                matched = seg
                break
        if matched:
            mapping[matched].append(name)
    for inst in mapping:
        mapping[inst].sort()
    return mapping


def run_concept(args: argparse.Namespace) -> None:
    zip_path = Path(args.zip_path)
    if args.download_if_missing:
        maybe_download_customconcept_zip(zip_path, args.gdrive_id)
    if not zip_path.exists():
        raise FileNotFoundError(f"CustomConcept zip not found: {zip_path}")

    dataset_json = fetch_or_load_customconcept_dataset_json(
        Path(args.dataset_json) if args.dataset_json else None,
        args.dataset_json_url,
    )
    target_per_group = clamp_images_per_group(args.images_per_group, args.images_min, args.images_max)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    entries_by_class: Dict[str, List[str]] = defaultdict(list)
    all_instance_names: List[str] = []
    for e in dataset_json:
        class_prompt = e.get("class_prompt", "").strip()
        inst_name = Path(e.get("instance_data_dir", "")).name
        if not class_prompt or not inst_name:
            continue
        entries_by_class[class_prompt].append(inst_name)
        all_instance_names.append(inst_name)

    with zipfile.ZipFile(zip_path, "r") as zf:
        image_members = [n for n in zf.namelist() if not n.endswith("/") and is_image_file(n)]
        instance_to_files = map_zip_images_to_instances(image_members, all_instance_names)

        class_candidates: List[Tuple[str, int, int]] = []
        for cls, instances in entries_by_class.items():
            uniq_instances = sorted(set(instances))
            total = sum(len(instance_to_files.get(inst, [])) for inst in uniq_instances)
            if total >= args.images_min:
                class_candidates.append((cls, len(uniq_instances), total))

        class_candidates.sort(key=lambda x: (x[1], x[2], x[0]), reverse=True)
        selected_classes = [c[0] for c in class_candidates[: args.target_groups]]
        if not selected_classes:
            raise RuntimeError("No Concept groups satisfy the minimum image requirement.")

        used_bytes = 0
        limit_bytes = int(args.max_output_gb * (1024**3))
        image_rows = []
        summary_rows = []

        for cls in selected_classes:
            cls_slug = slugify(cls)
            out_dir = out_root / cls_slug
            out_dir.mkdir(parents=True, exist_ok=True)

            instances = sorted(set(entries_by_class[cls]), key=lambda inst: len(instance_to_files.get(inst, [])), reverse=True)
            chosen_members: List[Tuple[str, str]] = []
            for inst in instances:
                for member in instance_to_files.get(inst, []):
                    if len(chosen_members) >= target_per_group:
                        break
                    chosen_members.append((inst, member))
                if len(chosen_members) >= target_per_group:
                    break

            saved_for_group = 0
            for idx, (inst, member) in enumerate(chosen_members, start=1):
                with zf.open(member, "r") as rf:
                    raw_bytes = rf.read()
                out_path = out_dir / f"{cls_slug}_{idx:04d}.jpg"
                saved_size = save_image_bytes_as_jpeg(
                    raw_bytes,
                    out_path=out_path,
                    max_side=args.max_side,
                    jpeg_quality=args.jpeg_quality,
                )
                if saved_size <= 0:
                    raw_ext = Path(member).suffix.lower() or ".img"
                    out_path = out_dir / f"{cls_slug}_{idx:04d}{raw_ext}"
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    with out_path.open("wb") as wf:
                        wf.write(raw_bytes)
                    saved_size = out_path.stat().st_size
                    if saved_size <= 0:
                        continue
                used_bytes += saved_size
                check_storage_limit(used_bytes, limit_bytes)
                saved_for_group += 1

                image_rows.append(
                    {
                        "dataset": "Concept",
                        "group": cls,
                        "source_instance": inst,
                        "source_member": member,
                        "output_path": str(out_path),
                    }
                )

            summary_rows.append(
                {
                    "dataset": "Concept",
                    "group": cls,
                    "instances_used": len(instances),
                    "saved_images": saved_for_group,
                    "target_images": target_per_group,
                }
            )

    write_csv(
        out_root / "manifest.csv",
        image_rows,
        ["dataset", "group", "source_instance", "source_member", "output_path"],
    )
    write_csv(
        out_root / "group_summary.csv",
        summary_rows,
        ["dataset", "group", "instances_used", "saved_images", "target_images"],
    )
    write_json(
        out_root / "summary.json",
        {
            "dataset": "Concept",
            "zip_path": str(zip_path),
            "target_groups": args.target_groups,
            "selected_groups": selected_classes,
            "images_per_group_target": target_per_group,
            "images_saved_total": len(image_rows),
            "output_size_gb": round(sum(Path(r["output_path"]).stat().st_size for r in image_rows) / (1024**3), 4)
            if image_rows
            else 0.0,
            "note": "Groups are ranked by number of available instances and available images in the official zip.",
        },
    )
    print(f"[Concept] groups={len(selected_classes)} saved={len(image_rows)}")


def download_file(url: str, out_path: Path, timeout: int = 30, chunk_size: int = 1024 * 1024) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with out_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)


def download_url_bytes(url: str, timeout: int = 20) -> Optional[bytes]:
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code == 200 and resp.content:
            return resp.content
    except Exception:
        pass

    # requests can intermittently fail with SSL EOF in some proxy environments.
    try:
        proc = subprocess.run(
            [
                "curl",
                "-sS",
                "-L",
                "--retry",
                "2",
                "--retry-delay",
                "1",
                "--max-time",
                str(timeout),
                url,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        if proc.stdout:
            return proc.stdout
    except Exception:
        return None
    return None


def run_person_lfw(args: argparse.Namespace) -> None:
    tgz_path = Path(args.lfw_tgz)
    if args.download_if_missing and (not tgz_path.exists() or tgz_path.stat().st_size < 1024):
        print(f"[Person/LFW] downloading -> {tgz_path}")
        try:
            download_file(args.lfw_url, tgz_path, timeout=args.http_timeout)
        except Exception as e:
            raise RuntimeError(
                "Failed to download LFW tarball. You can switch to --source lfw_parquet "
                "or override --lfw-url with a reachable mirror."
            ) from e
    if not tgz_path.exists():
        raise FileNotFoundError(f"LFW tgz not found: {tgz_path}")

    target_per_group = clamp_images_per_group(args.images_per_group, args.images_min, args.images_max)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tgz_path, "r:gz") as tf:
        identity_to_members: Dict[str, List[tarfile.TarInfo]] = defaultdict(list)
        for member in tf.getmembers():
            if not member.isfile():
                continue
            if not is_image_file(member.name):
                continue
            parts = Path(member.name).parts
            if len(parts) < 2:
                continue
            identity = parts[-2]
            identity_to_members[identity].append(member)

        candidates = [(identity, len(members)) for identity, members in identity_to_members.items() if len(members) >= args.images_min]
        candidates.sort(key=lambda x: (x[1], x[0]), reverse=True)
        selected = [identity for identity, _ in candidates[: args.target_groups]]
        if not selected:
            raise RuntimeError("No LFW identities satisfy the minimum image requirement.")

        image_rows = []
        summary_rows = []
        used_bytes = 0
        limit_bytes = int(args.max_output_gb * (1024**3))

        for identity in selected:
            members = sorted(identity_to_members[identity], key=lambda m: m.name)[:target_per_group]
            identity_slug = slugify(identity)
            out_dir = out_root / identity_slug
            out_dir.mkdir(parents=True, exist_ok=True)

            for idx, member in enumerate(members, start=1):
                fobj = tf.extractfile(member)
                if fobj is None:
                    continue
                raw = fobj.read()
                out_path = out_dir / f"{identity_slug}_{idx:04d}.jpg"
                saved_size = save_image_bytes_as_jpeg(
                    raw,
                    out_path=out_path,
                    max_side=args.max_side,
                    jpeg_quality=args.jpeg_quality,
                )
                if saved_size <= 0:
                    continue
                used_bytes += saved_size
                check_storage_limit(used_bytes, limit_bytes)
                image_rows.append(
                    {
                        "dataset": "Person",
                        "group": identity,
                        "source_member": member.name,
                        "output_path": str(out_path),
                    }
                )

            summary_rows.append(
                {
                    "dataset": "Person",
                    "group": identity,
                    "available_images": len(identity_to_members[identity]),
                    "saved_images": len(members),
                    "target_images": target_per_group,
                }
            )

    write_csv(out_root / "manifest.csv", image_rows, ["dataset", "group", "source_member", "output_path"])
    write_csv(
        out_root / "group_summary.csv",
        summary_rows,
        ["dataset", "group", "available_images", "saved_images", "target_images"],
    )
    write_json(
        out_root / "summary.json",
        {
            "dataset": "Person",
            "source": "LFW",
            "target_groups": args.target_groups,
            "selected_groups": selected,
            "images_per_group_target": target_per_group,
            "images_saved_total": len(image_rows),
            "output_size_gb": round(sum(Path(r["output_path"]).stat().st_size for r in image_rows) / (1024**3), 4)
            if image_rows
            else 0.0,
            "note": "LFW fallback ranked by identity image counts. Use official_laion mode when URL manifest is available.",
        },
    )
    print(f"[Person/LFW] groups={len(selected)} saved={len(image_rows)}")


def identity_from_lfw_filename(filename: str) -> str:
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) >= 2 and parts[-1].isdigit():
        return "_".join(parts[:-1])
    return stem


def run_person_lfw_parquet(args: argparse.Namespace) -> None:
    parquet_path = Path(args.lfw_parquet_path)
    if args.download_if_missing and (not parquet_path.exists() or parquet_path.stat().st_size < 1024):
        print(f"[Person/LFW-Parquet] downloading -> {parquet_path}")
        download_file(args.lfw_parquet_url, parquet_path, timeout=args.http_timeout)
    if not parquet_path.exists():
        raise FileNotFoundError(f"LFW parquet not found: {parquet_path}")

    target_per_group = clamp_images_per_group(args.images_per_group, args.images_min, args.images_max)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    tbl = pq.read_table(str(parquet_path), columns=["filename", "image"])
    filenames = tbl["filename"].to_pylist()
    images = tbl["image"].to_pylist()

    identity_to_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, fn in enumerate(filenames):
        identity_to_indices[identity_from_lfw_filename(fn)].append(idx)

    candidates = [(identity, len(idxs)) for identity, idxs in identity_to_indices.items() if len(idxs) >= args.images_min]
    candidates.sort(key=lambda x: (x[1], x[0]), reverse=True)
    selected = [identity for identity, _ in candidates[: args.target_groups]]
    if not selected:
        raise RuntimeError("No LFW identities in parquet satisfy the minimum image requirement.")

    image_rows = []
    summary_rows = []
    used_bytes = 0
    limit_bytes = int(args.max_output_gb * (1024**3))

    for identity in selected:
        idxs = identity_to_indices[identity][:target_per_group]
        identity_slug = slugify(identity)
        out_dir = out_root / identity_slug
        out_dir.mkdir(parents=True, exist_ok=True)

        for local_idx, row_idx in enumerate(idxs, start=1):
            item = images[row_idx]
            if not isinstance(item, dict) or not item.get("bytes"):
                continue
            out_path = out_dir / f"{identity_slug}_{local_idx:04d}.jpg"
            saved_size = save_image_bytes_as_jpeg(
                item["bytes"],
                out_path=out_path,
                max_side=args.max_side,
                jpeg_quality=args.jpeg_quality,
            )
            if saved_size <= 0:
                continue
            used_bytes += saved_size
            check_storage_limit(used_bytes, limit_bytes)
            image_rows.append(
                {
                    "dataset": "Person",
                    "group": identity,
                    "source_filename": filenames[row_idx],
                    "output_path": str(out_path),
                }
            )

        summary_rows.append(
            {
                "dataset": "Person",
                "group": identity,
                "available_images": len(identity_to_indices[identity]),
                "saved_images": min(len(identity_to_indices[identity]), target_per_group),
                "target_images": target_per_group,
            }
        )

    write_csv(out_root / "manifest.csv", image_rows, ["dataset", "group", "source_filename", "output_path"])
    write_csv(
        out_root / "group_summary.csv",
        summary_rows,
        ["dataset", "group", "available_images", "saved_images", "target_images"],
    )
    write_json(
        out_root / "summary.json",
        {
            "dataset": "Person",
            "source": "LFW parquet (bitmind/lfw)",
            "target_groups": args.target_groups,
            "selected_groups": selected,
            "images_per_group_target": target_per_group,
            "images_saved_total": len(image_rows),
            "output_size_gb": round(sum(Path(r["output_path"]).stat().st_size for r in image_rows) / (1024**3), 4)
            if image_rows
            else 0.0,
        },
    )
    print(f"[Person/LFW-Parquet] groups={len(selected)} saved={len(image_rows)}")


def fetch_hf_rows_page(
    dataset: str,
    config: str,
    split: str,
    offset: int,
    length: int,
    timeout: int,
    retries: int = 4,
) -> dict:
    params = {
        "dataset": dataset,
        "config": config,
        "split": split,
        "offset": offset,
        "length": length,
    }
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            resp = requests.get(HF_ROWS_ENDPOINT, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            last_err = e
            if attempt + 1 >= retries:
                break
            time.sleep(1.0 + attempt * 1.5)
    # Fallback via curl for environments where requests+TLS is flaky.
    try:
        query = urllib.parse.urlencode(params)
        url = f"{HF_ROWS_ENDPOINT}?{query}"
        proc = subprocess.run(
            [
                "curl",
                "-sS",
                "-L",
                "--retry",
                "3",
                "--retry-delay",
                "1",
                "--max-time",
                str(timeout),
                url,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True,
        )
        payload = json.loads(proc.stdout)
        return payload
    except Exception as curl_err:
        raise RuntimeError(f"rows API failed at offset={offset}, length={length}") from (last_err or curl_err)


def run_person_hf_rows(args: argparse.Namespace) -> None:
    target_per_group = clamp_images_per_group(args.images_per_group, args.images_min, args.images_max)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    page_size = max(1, min(100, args.hf_rows_page_size))
    first = fetch_hf_rows_page(
        dataset=args.hf_rows_dataset,
        config=args.hf_rows_config,
        split=args.hf_rows_split,
        offset=0,
        length=page_size,
        timeout=args.http_timeout,
    )

    total_rows = int(first.get("num_rows_total", 0))
    if total_rows <= 0:
        raise RuntimeError("Failed to get rows from datasets-server for person source.")

    counts: Counter = Counter()
    samples: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

    def ingest_rows(payload: dict) -> None:
        for item in payload.get("rows", []):
            row = item.get("row", {})
            filename = (row.get("filename") or "").strip()
            image = row.get("image") or {}
            image_src = (image.get("src") or "").strip()
            if not filename or not image_src:
                continue
            identity = identity_from_lfw_filename(filename)
            counts[identity] += 1
            if len(samples[identity]) < target_per_group:
                samples[identity].append((filename, image_src))

    ingest_rows(first)
    scan_incomplete = False
    for offset in range(page_size, total_rows, page_size):
        try:
            payload = fetch_hf_rows_page(
                dataset=args.hf_rows_dataset,
                config=args.hf_rows_config,
                split=args.hf_rows_split,
                offset=offset,
                length=page_size,
                timeout=args.http_timeout,
            )
            ingest_rows(payload)
            if offset % (page_size * 20) == 0:
                print(f"[Person/HF-rows] scanned rows: {offset}/{total_rows}")
        except Exception as e:
            print(f"[Person/HF-rows] warning: stop scan at offset={offset}: {e}")
            scan_incomplete = True
            break

    candidates = [(identity, cnt) for identity, cnt in counts.items() if cnt >= args.images_min]
    candidates.sort(key=lambda x: (x[1], x[0]), reverse=True)
    selected = [identity for identity, _ in candidates[: args.target_groups]]
    if not selected:
        raise RuntimeError("No person groups in HF rows source satisfy the minimum image requirement.")

    image_rows = []
    summary_rows = []
    used_bytes = 0
    limit_bytes = int(args.max_output_gb * (1024**3))
    for identity in selected:
        identity_slug = slugify(identity)
        out_dir = out_root / identity_slug
        out_dir.mkdir(parents=True, exist_ok=True)

        saved = 0
        for filename, image_src in samples[identity]:
            if saved >= target_per_group:
                break
            try:
                raw_bytes = download_url_bytes(image_src, timeout=args.http_timeout)
                if not raw_bytes:
                    continue
                out_path = out_dir / f"{identity_slug}_{saved + 1:04d}.jpg"
                saved_size = save_image_bytes_as_jpeg(
                    raw_bytes,
                    out_path=out_path,
                    max_side=args.max_side,
                    jpeg_quality=args.jpeg_quality,
                )
                if saved_size <= 0:
                    continue
                used_bytes += saved_size
                check_storage_limit(used_bytes, limit_bytes)
                saved += 1
                image_rows.append(
                    {
                        "dataset": "Person",
                        "group": identity,
                        "source_filename": filename,
                        "source_url": image_src,
                        "output_path": str(out_path),
                    }
                )
            except Exception:
                continue

        summary_rows.append(
            {
                "dataset": "Person",
                "group": identity,
                "available_images": counts[identity],
                "saved_images": saved,
                "target_images": target_per_group,
            }
        )

    write_csv(
        out_root / "manifest.csv",
        image_rows,
        ["dataset", "group", "source_filename", "source_url", "output_path"],
    )
    write_csv(
        out_root / "group_summary.csv",
        summary_rows,
        ["dataset", "group", "available_images", "saved_images", "target_images"],
    )
    write_json(
        out_root / "summary.json",
        {
            "dataset": "Person",
            "source": f"datasets-server rows ({args.hf_rows_dataset})",
            "target_groups": args.target_groups,
            "selected_groups": selected,
            "images_per_group_target": target_per_group,
            "images_saved_total": len(image_rows),
            "output_size_gb": round(sum(Path(r["output_path"]).stat().st_size for r in image_rows) / (1024**3), 4)
            if image_rows
            else 0.0,
            "scan_incomplete": scan_incomplete,
        },
    )
    print(f"[Person/HF-rows] groups={len(selected)} saved={len(image_rows)}")


def run_person_official_laion(args: argparse.Namespace) -> None:
    manifest_path = Path(args.laion_manifest) if args.laion_manifest else None
    if not manifest_path or not manifest_path.exists():
        raise RuntimeError(
            "official_laion mode requires --laion-manifest CSV (columns: person,url). "
            "The public paper setup (10 celebrities x 15 images) was derived from LAION, "
            "but no direct open download bundle was found in the referenced repositories."
        )

    rows = []
    with manifest_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "person" not in reader.fieldnames or "url" not in reader.fieldnames:
            raise RuntimeError("laion-manifest must contain columns: person,url")
        for row in reader:
            person = (row.get("person") or "").strip()
            url = (row.get("url") or "").strip()
            if person and url:
                rows.append((person, url))

    target_per_group = clamp_images_per_group(args.images_per_group, args.images_min, args.images_max)
    person_to_urls: Dict[str, List[str]] = defaultdict(list)
    for person, url in rows:
        person_to_urls[person].append(url)

    candidates = [(person, len(urls)) for person, urls in person_to_urls.items() if len(urls) >= args.images_min]
    candidates.sort(key=lambda x: (x[1], x[0]), reverse=True)
    selected_people = [p for p, _ in candidates[: args.target_groups]]
    if not selected_people:
        raise RuntimeError("No person groups in manifest satisfy the minimum image requirement.")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    image_rows = []
    summary_rows = []
    used_bytes = 0
    limit_bytes = int(args.max_output_gb * (1024**3))

    for person in selected_people:
        urls = person_to_urls[person]
        person_slug = slugify(person)
        out_dir = out_root / person_slug
        out_dir.mkdir(parents=True, exist_ok=True)
        saved = 0

        for url in urls:
            if saved >= target_per_group:
                break
            try:
                raw_bytes = download_url_bytes(url, timeout=args.http_timeout)
                if not raw_bytes:
                    continue
                out_path = out_dir / f"{person_slug}_{saved + 1:04d}.jpg"
                saved_size = save_image_bytes_as_jpeg(
                    raw_bytes,
                    out_path=out_path,
                    max_side=args.max_side,
                    jpeg_quality=args.jpeg_quality,
                )
                if saved_size <= 0:
                    continue
                used_bytes += saved_size
                check_storage_limit(used_bytes, limit_bytes)
                image_rows.append(
                    {
                        "dataset": "Person",
                        "group": person,
                        "source_url": url,
                        "output_path": str(out_path),
                    }
                )
                saved += 1
            except Exception:
                continue

        summary_rows.append(
            {
                "dataset": "Person",
                "group": person,
                "available_urls": len(urls),
                "saved_images": saved,
                "target_images": target_per_group,
            }
        )

    write_csv(out_root / "manifest.csv", image_rows, ["dataset", "group", "source_url", "output_path"])
    write_csv(
        out_root / "group_summary.csv",
        summary_rows,
        ["dataset", "group", "available_urls", "saved_images", "target_images"],
    )
    write_json(
        out_root / "summary.json",
        {
            "dataset": "Person",
            "source": "official_laion_manifest",
            "target_groups": args.target_groups,
            "selected_groups": selected_people,
            "images_per_group_target": target_per_group,
            "images_saved_total": len(image_rows),
            "output_size_gb": round(sum(Path(r["output_path"]).stat().st_size for r in image_rows) / (1024**3), 4)
            if image_rows
            else 0.0,
        },
    )
    print(f"[Person/LAION] groups={len(selected_people)} saved={len(image_rows)}")


def run_person(args: argparse.Namespace) -> None:
    if args.source == "lfw":
        run_person_lfw(args)
    elif args.source == "lfw_parquet":
        run_person_lfw_parquet(args)
    elif args.source == "hf_rows_lfw":
        run_person_hf_rows(args)
    elif args.source == "official_laion":
        run_person_official_laion(args)
    else:
        raise RuntimeError(f"Unsupported person source: {args.source}")


def add_common_group_args(parser: argparse.ArgumentParser, target_groups: int, images_per_group: int, max_output_gb: float) -> None:
    parser.add_argument("--target-groups", type=int, default=target_groups, help="How many groups to keep.")
    parser.add_argument("--images-per-group", type=int, default=images_per_group, help="Target images per group.")
    parser.add_argument("--images-min", type=int, default=25, help="Minimum images per group.")
    parser.add_argument("--images-max", type=int, default=40, help="Maximum images per group.")
    parser.add_argument("--max-output-gb", type=float, default=max_output_gb, help="Storage cap for saved subset.")
    parser.add_argument("--max-side", type=int, default=768, help="Resize longer side to this value (0 disables resize).")
    parser.add_argument("--jpeg-quality", type=int, default=92, help="JPEG quality when saving extracted images.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare partial benchmark subsets with storage limits.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_wiki = sub.add_parser("wikiart", help="Build WikiArt subset from local HF parquet cache.")
    add_common_group_args(p_wiki, target_groups=15, images_per_group=40, max_output_gb=4.0)
    p_wiki.add_argument(
        "--group-by",
        choices=["style", "artist"],
        default="style",
        help="How to define groups for WikiArt ranking and sampling.",
    )
    p_wiki.add_argument(
        "--focus-group",
        type=str,
        default="",
        help="Optional exact name / slug / index to force a single target group.",
    )
    p_wiki.add_argument("--wikiart-snapshot", type=str, default="", help="Path to HF snapshot dir; auto-detect if empty.")
    p_wiki.add_argument(
        "--dataset-info",
        type=str,
        default="/root/wikiart/dataset_infos.json",
        help="Path to dataset_infos.json for label decoding.",
    )
    p_wiki.add_argument("--out-root", type=str, default="datasets/partial/WikiArt")
    p_wiki.set_defaults(func=run_wikiart)

    p_concept = sub.add_parser("concept", help="Build Concept subset from CustomConcept101 official zip.")
    add_common_group_args(p_concept, target_groups=15, images_per_group=30, max_output_gb=6.0)
    p_concept.add_argument("--zip-path", type=str, default="datasets/raw/customconcept101/benchmark_dataset.zip")
    p_concept.add_argument("--download-if-missing", action="store_true", help="Download official zip using gdown if missing.")
    p_concept.add_argument("--gdrive-id", type=str, default=CUSTOMCONCEPT_GDRIVE_ID)
    p_concept.add_argument("--dataset-json", type=str, default="", help="Local dataset.json path (optional).")
    p_concept.add_argument("--dataset-json-url", type=str, default=CUSTOMCONCEPT_DATASET_JSON_URL)
    p_concept.add_argument("--out-root", type=str, default="datasets/partial/Concept")
    p_concept.set_defaults(func=run_concept)

    p_person = sub.add_parser("person", help="Build Person subset (LFW fallback or official LAION URL manifest).")
    add_common_group_args(p_person, target_groups=15, images_per_group=30, max_output_gb=4.0)
    p_person.add_argument(
        "--source",
        choices=["lfw", "lfw_parquet", "hf_rows_lfw", "official_laion"],
        default="hf_rows_lfw",
    )
    p_person.add_argument("--out-root", type=str, default="datasets/partial/Person")
    p_person.add_argument("--download-if-missing", action="store_true")
    p_person.add_argument("--lfw-url", type=str, default=LFW_DEFAULT_URL)
    p_person.add_argument("--lfw-tgz", type=str, default="datasets/raw/person/lfw-deepfunneled.tgz")
    p_person.add_argument("--lfw-parquet-url", type=str, default=LFW_PARQUET_DEFAULT_URL)
    p_person.add_argument("--lfw-parquet-path", type=str, default="datasets/raw/person/bitmind_lfw.parquet")
    p_person.add_argument("--hf-rows-dataset", type=str, default="bitmind/lfw")
    p_person.add_argument("--hf-rows-config", type=str, default="default")
    p_person.add_argument("--hf-rows-split", type=str, default="train")
    p_person.add_argument("--hf-rows-page-size", type=int, default=100)
    p_person.add_argument("--laion-manifest", type=str, default="", help="CSV with columns person,url for official_laion mode.")
    p_person.add_argument("--http-timeout", type=int, default=20)
    p_person.set_defaults(func=run_person)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, "max_side", 0) <= 0:
        args.max_side = None
    args.func(args)


if __name__ == "__main__":
    main()
