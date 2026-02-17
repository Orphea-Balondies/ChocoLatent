#!/usr/bin/env python3
import sys
from pathlib import Path


def main():
    repo_root = Path(__file__).resolve().parents[2]
    code_dir = repo_root / "code"
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))

    from distribution_adv_tgt import main as protect_main

    protect_main()


if __name__ == "__main__":
    main()
