# DiskChat Agent v2.1

**Ultra-low-RAM local LLM** with **tool calling** and an optional **HTTP API** for outer agents.

Supports **Linux x86_64** and **Linux aarch64 (arm64)**.

**v2.1:** extreme-low-RAM mode for **multi-GB / 7B–70B-class** GGUFs (auto RAM budget, tiny KV, mmap paging, split GGUF support).

| Capability | Detail |
|------------|--------|
| Weights | GGUF **mmap** (not fully loaded into RAM) |
| Context | Adaptive, ceiling **131072** |
| KV cache | `q8_0` / `f16` |
| Tools | 11 builtins + pluggable registry |
| Agent API | CLI + **HTTP** `/v1/chat` |
| Sessions | `/save` `/load` |
| Profiles | `default` `coding` `creative` `agent` |
| Platforms | Linux **x86_64**, **aarch64** |

---

## Install

```bash
git clone https://github.com/bageltrade/diskchat.git
cd diskchat

# Runtime for your CPU (x86_64 or aarch64)
bash scripts/install_runtime.sh

pip install -r requirements.txt
python scripts/download_model.py

export DISKCHAT_LLAMA_CLI="$PWD/bin/llama-cli"
export DISKCHAT_LIB_DIR="$PWD/bin"
export LD_LIBRARY_PATH="$PWD/bin:${LD_LIBRARY_PATH:-}"
export DISKCHAT_MODEL="$HOME/.cache/diskchat/models/qwen2.5-1.5b-instruct-q4_k_m.gguf"

python diskchat.py --doctor
python diskchat.py --selftest
```

### Architecture matrix

| `uname -m` | Binary selected by `install_runtime.sh` |
|------------|-------------------------------------------|
| `x86_64` | `llama-*-bin-ubuntu-x64.tar.gz` |
| `aarch64` | `llama-*-bin-ubuntu-arm64.tar.gz` |

---

## Usage

```bash
# Plain chat
python diskchat.py --once "Hello"

# Tool agent
python diskchat.py --agent --once "What is 17*19? Use tools."

# Profiles
python diskchat.py --profile coding --agent
python diskchat.py --profile creative

# HTTP API for external agents / apps
python diskchat.py --serve --agent --port 8765

# Diagnostics
python diskchat.py --doctor
python diskchat.py --list-tools
python diskchat.py --version
```

### Interactive commands

| Command | Action |
|---------|--------|
| `/reset` | Clear history |
| `/mem` | RAM snapshot |
| `/tools` | List tools |
| `/save [name]` | Save session |
| `/load [name]` | Load session |
| `/quit` | Exit |

### HTTP API

```bash
# Health
curl http://127.0.0.1:8765/health

# Chat
curl -s http://127.0.0.1:8765/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"What is 2+2?"}'

# Tools schema (OpenAI-style)
curl http://127.0.0.1:8765/v1/tools

# Reset conversation
curl -X POST http://127.0.0.1:8765/v1/reset
```

Also accepts OpenAI-ish body: `{"messages":[{"role":"user","content":"..."}]}`.

---

## Built-in tools (11)

`calculator` · `get_time` · `list_dir` · `read_file` · `write_file` · `memory_stats` · `echo` · **`http_get`** · **`search_workspace`** · **`glob_files`** · **`platform_info`**

Workspace: `~/.cache/diskchat/workspace` (override with `DISKCHAT_WORKSPACE`).

Parallel tool execution is enabled by default when the model emits multiple calls.

### Register custom tools

```python
import importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location("diskchat", Path("diskchat.py"))
dc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dc)

reg = dc.ToolRegistry.with_builtins()
reg.register("my_tool", lambda x="": {"ok": x}, "desc",
             {"type":"object","properties":{"x":{"type":"string"}}})
```

---


---

## Extreme low RAM (large models)

DiskChat never `mlock`s weights. For **huge GGUFs** on small machines:

```bash
# Auto-tune ctx/batch from free RAM + model size
python diskchat.py --extreme-low-ram --doctor
python diskchat.py --extreme-low-ram --once "Hello"

# Or set an explicit budget (MB)
python diskchat.py --ram-budget 2048 --extreme-low-ram --agent

# Large quant downloads (Q2/Q3 preferred)
python scripts/download_model.py --preset large    # ~14B Q2
python scripts/download_model.py --preset xlarge   # ~32B Q2
```

| Knob | Effect |
|------|--------|
| `--extreme-low-ram` | ctx≤1024, batch 16/8, few threads, small `n_predict` |
| `--ram-budget MB` | plan KV from this ceiling instead of MemAvailable |
| mmap weights | OS pages tensors from disk when RAM is tight |
| split GGUF | pass `...-00001-of-00002.gguf` or a directory of shards |

**Honest limit:** if the GGUF is much larger than physical RAM, inference **works** but becomes **disk-bound** (slow). Prefer **Q2/Q3** quants and **short context** on 8–16 GB hosts for 12B–32B models.

## Low RAM design

1. Model weights → OS **mmap** (never `--mlock`)
2. KV → quantized K, F16 V
3. Context → **adaptive** (only prompt + reply size, not full 131k every turn)

Typical: **~1 GB model on disk**, **~10–20 MB parent RSS**.

---

## Environment

| Variable | Meaning |
|----------|---------|
| `DISKCHAT_MODEL` | GGUF path |
| `DISKCHAT_LLAMA_CLI` | `llama-cli` path |
| `DISKCHAT_LIB_DIR` | Shared libs dir |
| `DISKCHAT_WORKSPACE` | File-tool sandbox |
| `DISKCHAT_SESSIONS` | Session JSON dir |
| `LD_LIBRARY_PATH` | Include bin dir |

---

## Project layout

```
diskchat/
  diskchat.py
  scripts/install_runtime.sh    # x86_64 + aarch64
  scripts/download_model.py
  requirements.txt
  README.md
  LICENSE
```

## License

MIT (Python code). `llama.cpp` and model weights follow their own licenses.
