#!/usr/bin/env python3
"""Download a GGUF model for DiskChat from Hugging Face."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

def main() -> int:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Install: pip install huggingface_hub")
        return 1

    ap = argparse.ArgumentParser(description="Download GGUF model for DiskChat")
    ap.add_argument("--repo", default="Qwen/Qwen2.5-1.5B-Instruct-GGUF")
    ap.add_argument("--file", default="qwen2.5-1.5b-instruct-q4_k_m.gguf")
    ap.add_argument(
        "--dir",
        default=os.environ.get("DISKCHAT_MODEL_DIR", str(Path.home() / ".cache" / "diskchat" / "models")),
    )
    args = ap.parse_args()
    Path(args.dir).mkdir(parents=True, exist_ok=True)
    print(f"Downloading {args.repo}/{args.file} → {args.dir}")
    path = hf_hub_download(repo_id=args.repo, filename=args.file, local_dir=args.dir)
    size = Path(path).stat().st_size / (1024 * 1024)
    print(f"OK {path} ({size:.1f} MB)")
    print(f"\nRun with:\n  export DISKCHAT_MODEL={path}\n  python diskchat.py --selftest")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
