#!/usr/bin/env python3
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

METRIC_KEYS = [
    "num_pairs",
    "pg",
    "pgg_lpips_mean",
    "pgg_clip_mean",
    "clip_t_clean_mean",
    "clip_t_adv_mean",
    "qrr_clip_t",
    "fid_clean",
    "fid_adv",
    "delta_fid",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_ids", type=str, required=True, help="Comma-separated exp_id list.")
    parser.add_argument("--output_root", type=str, default="experiments/outputs")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--methods", type=str, default="chocolatent,glaze,photoguard,robust-ldm,mist")
    parser.add_argument("--budget_l2_grid", type=str, default="8/255,12/255,24/255")
    parser.add_argument("--budget_lpips_grid", type=str, default="0.1,0.2,0.5")
    return parser.parse_args()


def parse_methods(text: str) -> List[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


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


def float_or_none(value) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except Exception:
        return None


def in_budget_grid(value: Optional[float], grid: Sequence[float], atol: float = 1e-9) -> bool:
    if value is None:
        return False
    return any(abs(value - item) <= atol for item in grid)


def parse_run_id(run_id: str) -> Optional[Tuple[str, float, float]]:
    if "-b2_" not in run_id or "-bp_" not in run_id:
        return None
    method, suffix = run_id.split("-b2_", 1)
    b2_tag, bp_tag = suffix.split("-bp_", 1)
    try:
        b2 = float(b2_tag.replace("p", "."))
        bp = float(bp_tag.replace("p", "."))
    except Exception:
        return None
    return method, b2, bp


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def detect_dataset_kind(image_root: str) -> str:
    root_lower = image_root.lower()
    if "wikiart" in root_lower:
        return "wikiart"
    if "concept" in root_lower:
        return "concept"
    return "unknown"


def normalize_summary_row(row: Dict) -> Dict:
    normalized = {
        "run_id": row.get("run_id", ""),
        "method": row.get("method", ""),
        "budget_l2": float_or_none(row.get("budget_l2")),
        "budget_lpips": float_or_none(row.get("budget_lpips")),
    }
    for key in METRIC_KEYS:
        normalized[key] = float_or_none(row.get(key))
    return normalized


def summary_row_from_conclusion(run_id: str, method: str, b2: float, bp: float, payload: Dict) -> Dict:
    return {
        "run_id": run_id,
        "method": method,
        "budget_l2": b2,
        "budget_lpips": bp,
        "num_pairs": float_or_none(payload.get("num_pairs")),
        "pg": float_or_none(payload.get("pg")),
        "pgg_lpips_mean": float_or_none((payload.get("pgg_lpips") or {}).get("mean")),
        "pgg_clip_mean": float_or_none((payload.get("pgg_clip") or {}).get("mean")),
        "clip_t_clean_mean": float_or_none((payload.get("clip_t_clean") or {}).get("mean")),
        "clip_t_adv_mean": float_or_none((payload.get("clip_t_adv") or {}).get("mean")),
        "qrr_clip_t": float_or_none(payload.get("qrr_clip_t")),
        "fid_clean": float_or_none(payload.get("fid_clean")),
        "fid_adv": float_or_none(payload.get("fid_adv")),
        "delta_fid": float_or_none(payload.get("delta_fid")),
    }


def row_identity(row: Dict) -> str:
    run_id = str(row.get("run_id", "")).strip()
    if run_id:
        return run_id
    method = str(row.get("method", "")).strip()
    b2 = row.get("budget_l2")
    bp = row.get("budget_lpips")
    return f"{method}|{round_budget(b2)}|{round_budget(bp)}"


def collect_exp_rows(exp_root: Path, exp_id: str) -> List[Dict]:
    config_path = exp_root / "pipeline_config.json"
    if config_path.exists():
        config = read_json(config_path)
    else:
        config = {}

    image_root = str(config.get("image_root", ""))
    dataset_name = str(config.get("image_dirname", exp_id))
    dataset_kind = detect_dataset_kind(image_root)

    rows_by_id: Dict[str, Dict] = {}
    summary_csv = exp_root / "analysis" / "conclusion_summary.csv"
    if summary_csv.exists():
        with summary_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for item in reader:
                row = normalize_summary_row(item)
                row["exp_id"] = exp_id
                row["dataset_name"] = dataset_name
                row["dataset_kind"] = dataset_kind
                row["image_root"] = image_root
                rows_by_id[row_identity(row)] = row

    metrics_root = exp_root / "metrics"
    if not metrics_root.exists():
        return list(rows_by_id.values())

    for run_dir in sorted(p for p in metrics_root.iterdir() if p.is_dir()):
        parsed = parse_run_id(run_dir.name)
        if parsed is None:
            continue
        method, b2, bp = parsed
        run_config_path = run_dir / "config.yaml"
        if run_config_path.exists():
            try:
                run_config = read_json(run_config_path)
            except Exception:
                run_config = {}
            method = str(run_config.get("method", method))
            b2 = float_or_none(run_config.get("budget_l2")) or b2
            bp = float_or_none(run_config.get("budget_lpips")) or bp
        conclusion_json = run_dir / "conclusion" / "metrics.json"
        if not conclusion_json.exists():
            continue
        payload = read_json(conclusion_json)
        row = summary_row_from_conclusion(run_dir.name, method, b2, bp, payload)
        row["exp_id"] = exp_id
        row["dataset_name"] = dataset_name
        row["dataset_kind"] = dataset_kind
        row["image_root"] = image_root
        rows_by_id[row_identity(row)] = row
    return list(rows_by_id.values())


def round_budget(value: float) -> float:
    return round(float(value), 12)


def key_for_budget(b2: float, bp: float) -> Tuple[float, float]:
    return round_budget(b2), round_budget(bp)


def summarize_values(values: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    arr = np.array(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


def write_csv(path: Path, rows: List[Dict]):
    fieldnames = sorted({key for row in rows for key in row.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_aggregate_rows(rows: List[Dict], methods: Sequence[str], budget_pairs: Sequence[Tuple[float, float]]) -> List[Dict]:
    grouped: Dict[Tuple[str, float, float], Dict] = {}
    for row in rows:
        method = row["method"]
        b2 = row["budget_l2"]
        bp = row["budget_lpips"]
        key = (method, round_budget(b2), round_budget(bp))
        if key not in grouped:
            grouped[key] = {
                "datasets": set(),
                "exp_ids": set(),
                "values": defaultdict(list),
            }
        grouped[key]["datasets"].add(row["dataset_name"])
        grouped[key]["exp_ids"].add(row["exp_id"])
        for metric in METRIC_KEYS:
            value = row.get(metric)
            if value is not None:
                grouped[key]["values"][metric].append(float(value))

    method_order = {name: idx for idx, name in enumerate(methods)}
    pair_order = {key_for_budget(b2, bp): idx for idx, (b2, bp) in enumerate(budget_pairs)}

    summary_rows = []
    for (method, b2, bp), payload in grouped.items():
        row = {
            "method": method,
            "budget_l2": b2,
            "budget_lpips": bp,
            "dataset_count": len(payload["datasets"]),
            "dataset_names": ";".join(sorted(payload["datasets"])),
            "exp_ids": ";".join(sorted(payload["exp_ids"])),
        }
        for metric in METRIC_KEYS:
            mean_value, std_value = summarize_values(payload["values"].get(metric, []))
            row[f"{metric}_mean"] = mean_value
            row[f"{metric}_std"] = std_value
        summary_rows.append(row)

    summary_rows.sort(
        key=lambda item: (
            method_order.get(item["method"], 10**6),
            pair_order.get(key_for_budget(item["budget_l2"], item["budget_lpips"]), 10**6),
        )
    )
    return summary_rows


def build_coverage_rows(
    rows: List[Dict],
    methods: Sequence[str],
    budget_pairs: Sequence[Tuple[float, float]],
) -> List[Dict]:
    datasets = sorted({(row["exp_id"], row["dataset_kind"], row["dataset_name"]) for row in rows})
    observed = {
        (row["exp_id"], row["dataset_name"], row["method"], key_for_budget(row["budget_l2"], row["budget_lpips"]))
        for row in rows
    }

    coverage_rows = []
    for exp_id, dataset_kind, dataset_name in datasets:
        for method in methods:
            for b2, bp in budget_pairs:
                key = (exp_id, dataset_name, method, key_for_budget(b2, bp))
                coverage_rows.append(
                    {
                        "exp_id": exp_id,
                        "dataset_kind": dataset_kind,
                        "dataset_name": dataset_name,
                        "method": method,
                        "budget_l2": b2,
                        "budget_lpips": bp,
                        "has_result": 1 if key in observed else 0,
                    }
                )
    return coverage_rows


def try_plot_heatmaps(
    aggregate_rows: List[Dict],
    methods: Sequence[str],
    budget_pairs: Sequence[Tuple[float, float]],
    output_dir: Path,
) -> Optional[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as err:
        return str(err)

    method_order = {name: idx for idx, name in enumerate(methods)}
    pair_order = {key_for_budget(b2, bp): idx for idx, (b2, bp) in enumerate(budget_pairs)}
    budget_labels = [f"L2={b2:.4f}\nLP={bp:.3f}" for b2, bp in budget_pairs]

    metrics_to_plot = [
        ("pg_mean", "PG", "viridis"),
        ("pgg_lpips_mean_mean", "PGG LPIPS", "magma"),
        ("pgg_clip_mean_mean", "PGG CLIP", "cividis"),
        ("qrr_clip_t_mean", "QRR CLIP-T", "plasma"),
    ]

    for metric_key, title, cmap in metrics_to_plot:
        matrix = np.full((len(methods), len(budget_pairs)), np.nan, dtype=np.float64)
        for row in aggregate_rows:
            method_idx = method_order.get(row["method"])
            pair_idx = pair_order.get(key_for_budget(row["budget_l2"], row["budget_lpips"]))
            value = float_or_none(row.get(metric_key))
            if method_idx is None or pair_idx is None or value is None:
                continue
            matrix[method_idx, pair_idx] = value

        fig, ax = plt.subplots(figsize=(max(7, len(budget_pairs) * 1.5), max(4, len(methods) * 0.9)))
        image = ax.imshow(matrix, cmap=cmap, aspect="auto")
        ax.set_title(title)
        ax.set_yticks(np.arange(len(methods)))
        ax.set_yticklabels(methods)
        ax.set_xticks(np.arange(len(budget_pairs)))
        ax.set_xticklabels(budget_labels, rotation=35, ha="right")

        for i in range(len(methods)):
            for j in range(len(budget_pairs)):
                value = matrix[i, j]
                if np.isfinite(value):
                    ax.text(j, i, f"{value:.3f}", ha="center", va="center", color="white", fontsize=8)

        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(output_dir / f"heatmap_{metric_key}.png", dpi=180)
        plt.close(fig)

    return None


def main():
    args = parse_args()
    exp_ids = [item.strip() for item in args.exp_ids.split(",") if item.strip()]
    if not exp_ids:
        raise ValueError("exp_ids is empty")

    methods = parse_methods(args.methods)
    budget_l2_values = parse_budget_grid(args.budget_l2_grid)
    budget_lpips_values = parse_budget_grid(args.budget_lpips_grid)
    budget_pairs = [(b2, bp) for b2 in budget_l2_values for bp in budget_lpips_values]

    output_root = Path(args.output_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for exp_id in exp_ids:
        exp_root = output_root / exp_id
        if not exp_root.exists():
            print(f"[Warn] exp root missing, skip: {exp_root}")
            continue
        all_rows.extend(collect_exp_rows(exp_root, exp_id))

    if not all_rows:
        raise RuntimeError("No rows collected from provided exp_ids.")

    filtered_rows = []
    for row in all_rows:
        method = row.get("method", "")
        b2 = row.get("budget_l2")
        bp = row.get("budget_lpips")
        if method not in methods:
            continue
        if not in_budget_grid(b2, budget_l2_values):
            continue
        if not in_budget_grid(bp, budget_lpips_values):
            continue
        filtered_rows.append(row)

    if not filtered_rows:
        raise RuntimeError("No rows left after method/budget filtering.")

    write_csv(output_dir / "all_runs_filtered.csv", filtered_rows)
    with (output_dir / "all_runs_filtered.json").open("w", encoding="utf-8") as f:
        json.dump(filtered_rows, f, indent=2, ensure_ascii=False)

    aggregate_rows = build_aggregate_rows(filtered_rows, methods=methods, budget_pairs=budget_pairs)
    write_csv(output_dir / "aggregate_method_budget.csv", aggregate_rows)
    with (output_dir / "aggregate_method_budget.json").open("w", encoding="utf-8") as f:
        json.dump(aggregate_rows, f, indent=2, ensure_ascii=False)

    coverage_rows = build_coverage_rows(filtered_rows, methods=methods, budget_pairs=budget_pairs)
    write_csv(output_dir / "coverage_matrix.csv", coverage_rows)

    plot_err = try_plot_heatmaps(aggregate_rows, methods=methods, budget_pairs=budget_pairs, output_dir=output_dir)
    if plot_err:
        print(f"[Warn] skip heatmaps: {plot_err}")

    report = {
        "exp_ids_requested": exp_ids,
        "exp_ids_used": sorted({row["exp_id"] for row in filtered_rows}),
        "row_count": len(filtered_rows),
        "dataset_count": len({(row["exp_id"], row["dataset_name"]) for row in filtered_rows}),
        "method_count": len({row["method"] for row in filtered_rows}),
        "budget_l2_grid": budget_l2_values,
        "budget_lpips_grid": budget_lpips_values,
        "plot_error": plot_err,
    }
    with (output_dir / "analysis_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(
        f"[Done] rows={report['row_count']} datasets={report['dataset_count']} "
        f"methods={report['method_count']} output_dir={output_dir}"
    )


if __name__ == "__main__":
    main()
