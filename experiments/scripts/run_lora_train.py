#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--train_data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--caption_column", type=str, default="caption")
    parser.add_argument("--validation_prompt", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--checkpointing_steps", type=int, default=5000)
    parser.add_argument("--validation_epochs", type=int, default=10)
    return parser.parse_known_args()


def main():
    args, extra = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    train_script = repo_root / "code" / "train_text_to_image_lora.py"

    cmd = [
        "accelerate",
        "launch",
        str(train_script),
        "--pretrained_model_name_or_path",
        args.pretrained_model_name_or_path,
        "--train_data_dir",
        args.train_data_dir,
        "--caption_column",
        args.caption_column,
        "--resolution",
        str(args.resolution),
        "--random_flip",
        "--train_batch_size",
        str(args.train_batch_size),
        "--num_train_epochs",
        str(args.num_train_epochs),
        "--checkpointing_steps",
        str(args.checkpointing_steps),
        "--learning_rate",
        str(args.learning_rate),
        "--lr_scheduler",
        "constant",
        "--lr_warmup_steps",
        "0",
        "--seed",
        str(args.seed),
        "--output_dir",
        args.output_dir,
        "--validation_epochs",
        str(args.validation_epochs),
    ]
    if args.validation_prompt:
        cmd.extend(["--validation_prompt", args.validation_prompt])
    cmd.extend(extra)

    print("[Run]", " ".join(cmd))
    subprocess.run(cmd, cwd=repo_root, check=True)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"[Error] LoRA train failed with code={exc.returncode}", file=sys.stderr)
        raise
