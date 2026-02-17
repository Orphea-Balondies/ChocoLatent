#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="model/stable-diffusion-v1-5")
    parser.add_argument("--loras", type=str, nargs="+", required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--generate_num", type=int, default=10)
    return parser.parse_known_args()


def main():
    args, extra = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    generate_script = repo_root / "code" / "text2image_generate.py"

    cmd = [
        "python",
        str(generate_script),
        "--model_path",
        args.model_path,
        "--generate_num",
        str(args.generate_num),
        "--output_root",
        args.output_root,
        "--loras",
        *args.loras,
    ]
    if args.prompt:
        cmd.extend(["--prompt", args.prompt])
    cmd.extend(extra)

    print("[Run]", " ".join(cmd))
    subprocess.run(cmd, cwd=repo_root, check=True)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"[Error] generation failed with code={exc.returncode}", file=sys.stderr)
        raise
