# DiskChat Agent

**Ultra-low-RAM local LLM chat + tool calling** for **Linux x86_64** and **Linux aarch64 (arm64)**.

Model weights stay on **disk** (memory-mapped). Process RAM stays tiny. Optional **agent loop** with pluggable tools so you can hook outer agents.

| Feature | Detail |
|--------|--------|
| Platforms | Linux **x86_64**, Linux **aarch64** |
| Weights | GGUF via `llama.cpp` **mmap** (never `mlock`) |
| Context | Adaptive, ceiling **131072** |
| KV cache | `q8_0` / `f16` (low RAM) |
| Tools | calculator, files, time, memory_stats, + your own |
| Agent API | OpenAI-style tool schemas (`--list-tools`) |

---

## Requirements

- Linux **x86_64** or **aarch64**
- Python **3.10+**
- `curl` or `wget`, `tar`
- Disk space for a GGUF model (≈1 GB for Qwen2.5-1.5B Q4)

Optional: `pip install huggingface_hub` to download models with the helper script.

---

## Quick install

```bash
git clone https://github.com/bageltrade/diskchat.git
cd diskchat

# 1) llama.cpp runtime for your CPU arch (x86_64 or aarch64)
bash scripts/install_runtime.sh

# 2) Python deps (only needed for model download helper)
pip install -r requirements.txt

# 3) Download a small GGUF (example: Qwen2.5-1.5B Instruct Q4)
python scripts/download_model.py

# 4) Point env at the installed runtime (or use defaults under ./bin)
export DISKCHAT_LLAMA_CLI="$PWD/bin/llama-cli"
export DISKCHAT_LIB_DIR="$PWD/bin"
export LD_LIBRARY_PATH="$PWD/bin:${LD_LIBRARY_PATH:-}"
export DISKCHAT_MODEL="$HOME/.cache/diskchat/models/qwen2.5-1.5b-instruct-q4_k_m.gguf"

# 5) Verify
python diskchat.py --selftest
```

### aarch64 (ARM64) boards / servers

Same steps. `install_runtime.sh` selects:

| `uname -m` | Runtime archive |
|------------|-----------------|
| `x86_64` / `amd64` | `llama-*-bin-ubuntu-x64.tar.gz` |
| `aarch64` / `arm64` | `llama-*-bin-ubuntu-arm64.tar.gz` |

On Snapdragon-class devices you can override the tag/asset if needed by editing `LLAMA_CPP_TAG` in the install script (default `b11140`).

---

## Usage

```bash
# Plain chat
python diskchat.py --once "Hello"

# Interactive
python diskchat.py

# Tool-calling agent
python diskchat.py --agent --once "What is 17*19? Use tools."
python diskchat.py --agent

# List OpenAI-style tool schemas (for outer agents)
python diskchat.py --list-tools

# Self-test (chat + tools + RAM checks)
python diskchat.py --selftest
```

### Useful flags

```
--model PATH          GGUF model path
--ctx N               working context budget (default 2048)
--max-ctx N           hard ceiling (default 131072)
--agent               enable tool-calling loop
--max-tool-rounds N   agent tool iterations (default 4)
--cache-k q8_0|f16    KV K type
--cache-v f16         KV V type
--threads N
--n-predict N
--temp F
```

### Environment variables

| Variable | Meaning |
|----------|---------|
| `DISKCHAT_MODEL` | Path to `.gguf` |
| `DISKCHAT_LLAMA_CLI` | Path to `llama-cli` |
| `DISKCHAT_LIB_DIR` | Directory with `libllama.so` / `libggml*.so` |
| `DISKCHAT_WORKSPACE` | Sandbox for file tools |
| `LD_LIBRARY_PATH` | Must include the bin dir with shared libs |

---

## How low RAM works

1. **Weights** — `llama.cpp` maps the GGUF file (`mmap`). Do not use `--mlock`.
2. **KV cache** — quantized K (`q8_0`), F16 V (V quant needs flash-attn on many builds).
3. **Context** — adaptive: only allocates enough for the current prompt + reply, up to `--max-ctx`.

On a ~1–2 GB host, keep `--ctx` modest (512–2048). Raising the **ceiling** to 131k does not force a 131k allocation every turn.

Typical measurement (Qwen2.5-1.5B Q4, ~1066 MB on disk): **parent process RSS ≈ 10–20 MB**.

---

## Built-in tools

| Tool | Purpose |
|------|---------|
| `calculator` | Safe arithmetic |
| `get_time` | UTC time |
| `list_dir` | List workspace files |
| `read_file` | Read workspace file |
| `write_file` | Write workspace file |
| `memory_stats` | Process / system RAM |
| `echo` | Debug |

Workspace defaults to `~/.cache/diskchat/workspace` (override with `DISKCHAT_WORKSPACE`).

### Hook your own tools

```python
# Example: embed DiskChat in another agent
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "diskchat", Path("diskchat.py")
)
dc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dc)

reg = dc.ToolRegistry.with_builtins()
reg.register(
    "search",
    lambda query="": {"hits": []},
    "Search the knowledge base",
    {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
)

cfg = dc.EngineConfig(agent_mode=True, model_path="...", llama_cli="...", lib_dir="...")
engine = dc.DiskChatEngine(cfg, tools=reg)
print(engine.generate("Find X with search").text)
print(reg.openai_tools())  # for LangChain / custom orchestrators
```

Tool call format the model is instructed to emit:

```text
<tool_call>
{"name": "calculator", "arguments": {"expression": "17*19"}}
</tool_call>
```

---

## Larger / ~4B models

```bash
python scripts/download_model.py \
  --repo Qwen/Qwen2.5-3B-Instruct-GGUF \
  --file qwen2.5-3b-instruct-q2_k.gguf

export DISKCHAT_MODEL="$HOME/.cache/diskchat/models/qwen2.5-3b-instruct-q2_k.gguf"
```

Use stronger quants (Q3/Q4) only if you have enough RAM/disk. Prefer Q2/Q3 on small boards.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `llama-cli: not found` | Run `bash scripts/install_runtime.sh` |
| `error while loading shared libraries` | `export LD_LIBRARY_PATH=$PWD/bin:$LD_LIBRARY_PATH` |
| OOM / killed | Lower `--ctx`, use smaller GGUF quant |
| Empty replies | Ensure `--no-conversation` path (built-in); try `--temp 0.1` |
| aarch64 wrong binary | Confirm `uname -m` is `aarch64`; re-run install script |

---

## License

This project’s Python code is provided as-is for your use.  
`llama.cpp` binaries are from [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) under their license.  
Models are subject to their own licenses (e.g. Qwen).

---

## Project layout

```
diskchat/
  diskchat.py                 # main program (chat + agent + tools)
  requirements.txt
  scripts/
    install_runtime.sh        # Linux x86_64 / aarch64 llama.cpp install
    download_model.py         # HF GGUF download helper
  README.md
```
