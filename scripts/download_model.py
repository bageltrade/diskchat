#!/usr/bin/env python3
"""Download a GGUF model for DiskChat from Hugging Face."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

PRESETS = {
    "tiny": ("Qwen/Qwen2.5-1.5B-Instruct-GGUF", "qwen2.5-1.5b-instruct-q4_k_m.gguf"),
    "small": ("Qwen/Qwen2.5-3B-Instruct-GGUF", "qwen2.5-3b-instruct-q3_k_m.gguf"),
    "medium": ("Qwen/Qwen2.5-7B-Instruct-GGUF", "qwen2.5-7b-instruct-q3_k_m.gguf"),
    # Large models: prefer aggressive quant for low-RAM hosts
    "large": ("Qwen/Qwen2.5-14B-Instruct-GGUF", "qwen2.5-14b-instruct-q2_k.gguf"),
    "xlarge": ("Qwen/Qwen2.5-32B-Instruct-GGUF", "qwen2.5-32b-instruct-q2_k.gguf"),
}


def main() -> int:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Install: pip install huggingface_hub")
        return 1

    ap = argparse.ArgumentParser(description="Download GGUF model for DiskChat")
    ap.add_argument("--repo", default="")
    ap.add_argument("--file", default="")
    ap.add_argument(
        "--preset",
        choices=list(PRESETS),
        default="tiny",
        help="tiny|small|medium|large|xlarge (larger presets use Q2/Q3 for low RAM)",
    )
    ap.add_argument(
        "--dir",
        default=os.environ.get(
            "DISKCHAT_MODEL_DIR", str(Path.home() / ".cache" / "diskchat" / "models")
        ),
    )
    args = ap.parse_args()
    repo, filename = PRESETS[args.preset]
    if args.repo:
        repo = args.repo
    if args.file:
        filename = args.file
    Path(args.dir).mkdir(parents=True, exist_ok=True)
    print(f"Downloading {repo}/{filename} → {args.dir}")
    print("(large models: use DiskChat --extreme-low-ram so KV stays tiny; weights mmap from disk)")
    path = hf_hub_download(repo_id=repo, filename=filename, local_dir=args.dir)
    size = Path(path).stat().st_size / (1024 * 1024)
    print(f"OK {path} ({size:.1f} MB)")
    print(
        f"\nRun:\n  export DISKCHAT_MODEL={path}\n"
        f"  python diskchat.py --extreme-low-ram --doctor\n"
        f"  python diskchat.py --extreme-low-ram --once \"Hi\""
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
