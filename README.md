# DiskChat Agent v2.4

**Ultra-low-RAM local LLM** with **tool calling** and an optional **HTTP API**.

**Linux x86_64** and **Linux aarch64 (arm64)**.

Weights stay on **disk** (mmap). Process RAM stays tiny even for multi‑GB GGUFs.

| Capability | Detail |
|------------|--------|
| Weights | GGUF **mmap** (never `mlock`) |
| Context | Adaptive, ceiling **131072** |
| KV cache | `q8_0` / `f16` |
| Tools | 11 builtins + pluggable registry |
| Agent API | CLI + **HTTP** `/v1/chat` |
| Large models | `--extreme-low-ram`, split GGUF, auto RAM plan |
| Platforms | Linux **x86_64**, **aarch64** |

---

## Verified on a 1.2 GB RAM host

| Model | Quant | Disk | Parent RSS | Result |
|-------|--------|------|------------|--------|
| Qwen2.5-1.5B | Q4_K_M | ~1.1 GB | ~11–14 MB | OK |
| Qwen2.5-3B | Q2_K | ~1.3 GB | ~20–22 MB | OK |
| Qwen2.5-3B | Q4_K_M | ~2.0 GB | ~20 MB | OK |
| **Qwen2.5-7B** | **Q4_K_M** (split) | **~4.4 GB** | **~20 MB** | OK (`2+2` → `4`) |

When the GGUF is larger than physical RAM, the OS **pages** weights from disk (slower, still correct).

---



## App-ready: pass a GGUF directly

DiskChat accepts a model file the way a mobile app would hand it over:

```bash
# positional (share-sheet / file picker style)
python diskchat.py /sdcard/Download/qwen2.5-1.5b-instruct-q4_k_m.gguf --doctor
python diskchat.py /path/to/model.gguf --extreme-low-ram --once "Hi"

# flags (aliases)
python diskchat.py --gguf /path/to/model.gguf --doctor
python diskchat.py --model /path/to/model.gguf --doctor

# file:// URIs OK
python diskchat.py --gguf 'file:///storage/emulated/0/Download/model.gguf' --doctor

# scan known folders (Download, ~/.cache/diskchat/models, ...)
python diskchat.py --list-models
```

Also accepted:
- directory of split shards (`*-00001-of-*.gguf`)
- bare filename if it exists under `DISKCHAT_MODEL_DIR` or Download

Env for apps:
| Variable | Meaning |
|----------|---------|
| `DISKCHAT_MODEL` | default GGUF path |
| `DISKCHAT_MODEL_DIR` | folder to scan / resolve bare names |
| `DISKCHAT_LLAMA_CLI` | path to `llama-cli` |
| `DISKCHAT_LIB_DIR` | shared libs next to the binary |

Phone tip: keep quant small enough for free RAM (see extreme-low-ram). Watchdog-style guards should track **llama-cli RSS**, not only the Python parent.


## Install

```bash
git clone https://github.com/bageltrade/diskchat.git
cd diskchat

bash scripts/install_runtime.sh          # x86_64 or aarch64
pip install -r requirements.txt

# Models
python scripts/download_model.py --preset tiny    # 1.5B Q4
python scripts/download_model.py --preset small   # 3B Q4
python scripts/download_model.py --preset medium  # 7B Q4 split (~4.4 GB)

export DISKCHAT_LLAMA_CLI="$PWD/bin/llama-cli"
export DISKCHAT_LIB_DIR="$PWD/bin"
export LD_LIBRARY_PATH="$PWD/bin:${LD_LIBRARY_PATH:-}"
export DISKCHAT_MODEL="$HOME/.cache/diskchat/models/qwen2.5-1.5b-instruct-q4_k_m.gguf"

python diskchat.py --doctor
python diskchat.py --selftest
```

### Architecture

| `uname -m` | Runtime |
|------------|---------|
| `x86_64` | `llama-*-bin-ubuntu-x64` |
| `aarch64` | `llama-*-bin-ubuntu-arm64` |

---

## Usage

```bash
python diskchat.py --once "Hello"
python diskchat.py --agent --once "What is 17*19? Use tools."
python diskchat.py --profile coding --agent
python diskchat.py --serve --agent --port 8765

# Huge GGUF on small RAM
python diskchat.py --extreme-low-ram --doctor
python diskchat.py --extreme-low-ram --ram-budget 2048 --once "Hi"
python diskchat.py --extreme-low-ram --once "What is 2+2?"
```

### Interactive

`/reset` `/mem` `/tools` `/save [name]` `/load [name]` `/quit`

### HTTP API

```bash
curl http://127.0.0.1:8765/health
curl -s http://127.0.0.1:8765/v1/chat -H 'Content-Type: application/json' \
  -d '{"message":"Hello"}'
curl http://127.0.0.1:8765/v1/tools
curl -X POST http://127.0.0.1:8765/v1/reset
```

---

## Extreme low RAM (7B+ / multi‑GB)

```bash
python diskchat.py --extreme-low-ram --ram-budget 900 --once "Hello"
```

| Knob | Effect |
|------|--------|
| `--extreme-low-ram` | tiny KV/batch/threads |
| `--ram-budget MB` | plan from this ceiling |
| split GGUF | pass `…-00001-of-00002.gguf` (siblings auto-loaded) |
| mmap | OS pages weights when file ≫ RAM |

**Honest limit:** correct answers still work when the file is larger than RAM; expect **disk-bound** speed.

---

## Tools (11)

`calculator` · `get_time` · `list_dir` · `read_file` · `write_file` · `memory_stats` · `echo` · `http_get` · `search_workspace` · `glob_files` · `platform_info`

---

## Environment

| Variable | Meaning |
|----------|---------|
| `DISKCHAT_MODEL` | GGUF path (first shard OK for splits) |
| `DISKCHAT_LLAMA_CLI` | `llama-cli` |
| `DISKCHAT_LIB_DIR` | shared libs |
| `DISKCHAT_WORKSPACE` | file-tool sandbox |
| `DISKCHAT_SESSIONS` | session JSON dir |
| `LD_LIBRARY_PATH` | include bin dir |

---

## License

MIT (Python). `llama.cpp` and model weights follow their own licenses.


## Split GGUF

Multi-part models (`*-00001-of-0000N.gguf`) are fully supported:

```bash
# any shard path works — DiskChat selects 00001 for llama-cli
python diskchat.py --gguf /models/qwen-7b-q4-00002-of-00002.gguf --doctor

# incomplete sets are rejected with a clear missing-shard error
python diskchat.py --gguf /models/partial-00001-of-00003.gguf --doctor
```

`validate_split_gguf()` / `--doctor` report `found/expected` and total size across shards.

## Web chat app (v2.5)

Beautiful browser UI served by DiskChat itself:

```bash
python diskchat.py --serve --host 0.0.0.0 --port 8765 --extreme-low-ram
# open http://127.0.0.1:8765/
```

Features: chat bubbles, model path / scan GGUF, agent toggle, extreme low RAM, doctor, health pill.
