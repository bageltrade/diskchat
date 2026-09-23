#!/usr/bin/env python3
"""Download a GGUF model for DiskChat from Hugging Face (incl. split shards)."""
from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path

# Single-file presets use huggingface_hub when available; split uses direct URLs.
PRESETS: dict[str, dict] = {
    "tiny": {
        "repo": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "files": ["qwen2.5-1.5b-instruct-q4_k_m.gguf"],
    },
    "small": {
        "repo": "Qwen/Qwen2.5-3B-Instruct-GGUF",
        "files": ["qwen2.5-3b-instruct-q4_k_m.gguf"],
    },
    "medium": {
        "repo": "Qwen/Qwen2.5-7B-Instruct-GGUF",
        "files": [
            "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
            "qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf",
        ],
    },
    "large": {
        "repo": "Qwen/Qwen2.5-14B-Instruct-GGUF",
        "files": ["qwen2.5-14b-instruct-q2_k.gguf"],
    },
    "xlarge": {
        "repo": "Qwen/Qwen2.5-32B-Instruct-GGUF",
        "files": ["qwen2.5-32b-instruct-q2_k.gguf"],
    },
}


def _download_hf(repo: str, filename: str, dest_dir: Path) -> Path:
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=repo, filename=filename, local_dir=str(dest_dir))
        return Path(path)
    except Exception:
        url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
        out = dest_dir / filename
        print(f"  curl-style fetch {url}")
        urllib.request.urlretrieve(url, out)
        return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Download GGUF model(s) for DiskChat")
    ap.add_argument("--repo", default="")
    ap.add_argument("--file", default="", help="single file (repeat via preset for shards)")
    ap.add_argument(
        "--preset",
        choices=list(PRESETS),
        default="tiny",
        help="tiny=1.5B Q4 | small=3B Q4 | medium=7B Q4 split | large/xlarge Q2",
    )
    ap.add_argument(
        "--dir",
        default=os.environ.get(
            "DISKCHAT_MODEL_DIR", str(Path.home() / ".cache" / "diskchat" / "models")
        ),
    )
    args = ap.parse_args()
    dest = Path(args.dir)
    dest.mkdir(parents=True, exist_ok=True)

    if args.repo and args.file:
        repo, files = args.repo, [args.file]
    else:
        preset = PRESETS[args.preset]
        repo, files = preset["repo"], list(preset["files"])
        if args.repo:
            repo = args.repo

    print(f"Preset/repo: {repo}")
    print("Large models → use: python diskchat.py --extreme-low-ram")
    paths = []
    for fn in files:
        print(f"Downloading {fn} ...")
        paths.append(_download_hf(repo, fn, dest))
    total = sum(p.stat().st_size for p in paths) / (1024 * 1024)
    primary = paths[0]
    print(f"OK {len(paths)} file(s), total {total:.1f} MB")
    print(f"Primary: {primary}")
    print(
        f"\nexport DISKCHAT_MODEL={primary}\n"
        f"python diskchat.py --extreme-low-ram --doctor\n"
        f"python diskchat.py --extreme-low-ram --once \"Hi\""
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
