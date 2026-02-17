#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_root", type=str, required=True, help="Path to experiments/outputs/<exp_id>")
    parser.add_argument("--output_csv", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    exp_root = Path(args.exp_root)
    metrics_root = exp_root / "metrics"
    if not metrics_root.exists():
        raise FileNotFoundError(f"metrics directory not found: {metrics_root}")

    rows = []
    for metrics_file in sorted(metrics_root.glob("*/metrics.json")):
        with open(metrics_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
        rows.append(payload)

    if not rows:
        raise RuntimeError(f"No metrics.json found under {metrics_root}")

    fieldnames = sorted({k for row in rows for k in row.keys()})
    output_csv = Path(args.output_csv) if args.output_csv else exp_root / "analysis_summary.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"[Done] wrote {len(rows)} rows -> {output_csv}")


if __name__ == "__main__":
    main()
