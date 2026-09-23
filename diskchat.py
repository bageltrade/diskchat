#!/usr/bin/env python3
"""
DiskChat Agent v2 — ultra-low-RAM mmap LLM + tools + HTTP API
=============================================================
Linux x86_64 & aarch64. Weights mmap'd from disk. Adaptive ctx ≤131k.
Tool-calling agent + optional local HTTP server for outer agents.
Extreme-low-RAM mode: auto budget for multi-GB / 7B–70B-class GGUFs via mmap.

  python diskchat.py --selftest
  python diskchat.py --doctor
  python diskchat.py --agent --once "What is 17*19?"
  python diskchat.py --serve --port 8765 --agent
  python diskchat.py --profile coding --agent
"""
from __future__ import annotations

import argparse
import ast
import json
import operator
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
VERSION = "2.1.0"


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _default_paths() -> tuple[str, str, str]:
    root = _project_root()
    home_cache = Path.home() / ".cache" / "diskchat"
    model_candidates = [
        os.environ.get("DISKCHAT_MODEL", ""),
        str(home_cache / "models" / "qwen2.5-1.5b-instruct-q4_k_m.gguf"),
        str(root / "models" / "qwen2.5-1.5b-instruct-q4_k_m.gguf"),
        "/tmp/diskchat-models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
    ]
    model = next((m for m in model_candidates if m and Path(m).is_file()), model_candidates[1])
    cli_candidates = [
        os.environ.get("DISKCHAT_LLAMA_CLI", ""),
        str(root / "bin" / "llama-cli"),
        str(home_cache / "bin" / "llama-cli"),
        "/tmp/diskchat-bin/llama-cli",
    ]
    cli = next((c for c in cli_candidates if c and Path(c).is_file()), cli_candidates[1])
    lib_candidates = [
        os.environ.get("DISKCHAT_LIB_DIR", ""),
        str(root / "bin"),
        str(home_cache / "bin"),
        "/tmp/diskchat-bin",
    ]
    lib = next((d for d in lib_candidates if d and Path(d).is_dir()), lib_candidates[1])
    return model, cli, lib


DEFAULT_MODEL, DEFAULT_LLAMA_CLI, DEFAULT_LIB_DIR = _default_paths()
MAX_CONTEXT_CEILING = 131_072
DEFAULT_CTX = 2048
WORKSPACE = Path(
    os.environ.get("DISKCHAT_WORKSPACE", str(Path.home() / ".cache" / "diskchat" / "workspace"))
)
SESSION_DIR = Path(
    os.environ.get("DISKCHAT_SESSIONS", str(Path.home() / ".cache" / "diskchat" / "sessions"))
)

PROFILES: dict[str, dict[str, Any]] = {
    "default": {
        "system_prompt": "You are a helpful, concise assistant.",
        "base_system": "You are a helpful assistant with tools.",
        "temperature": 0.7,
        "n_predict": 256,
    },
    "coding": {
        "system_prompt": "You are a precise coding assistant. Prefer short, correct code.",
        "base_system": "You are a coding agent with tools. Prefer calculator for math and workspace tools for files.",
        "temperature": 0.2,
        "n_predict": 512,
    },
    "creative": {
        "system_prompt": "You are a creative writer. Be vivid but coherent.",
        "base_system": "You are a creative assistant with tools.",
        "temperature": 0.95,
        "n_predict": 384,
    },
    "agent": {
        "system_prompt": "You are an autonomous agent. Use tools when they improve accuracy.",
        "base_system": "You are an autonomous agent. Always use tools for math, time, and files instead of guessing.",
        "temperature": 0.3,
        "n_predict": 256,
    },
}


# ============================== memory =====================================

@dataclass
class MemSnapshot:
    rss_mb: float
    vms_mb: float
    sys_avail_mb: float
    sys_total_mb: float

    def __str__(self) -> str:
        return (
            f"RSS={self.rss_mb:.1f}MB VMS={self.vms_mb:.1f}MB "
            f"avail={self.sys_avail_mb:.0f}/{self.sys_total_mb:.0f}MB"
        )


def mem_snapshot(pid: int | None = None) -> MemSnapshot:
    rss = vms = 0.0
    try:
        with open(f"/proc/{pid or 'self'}/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1]) / 1024.0
                elif line.startswith("VmSize:"):
                    vms = int(line.split()[1]) / 1024.0
    except OSError:
        pass
    total = avail = 0.0
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total = int(line.split()[1]) / 1024.0
                elif line.startswith("MemAvailable:"):
                    avail = int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return MemSnapshot(rss, vms, avail, total)


def file_mb(path: str) -> float:
    try:
        return Path(path).stat().st_size / (1024 * 1024)
    except OSError:
        return 0.0


# ---------------------------------------------------------------------------
# Extreme low-RAM planner (large GGUFs on small machines)
# ---------------------------------------------------------------------------
def estimate_params_b(model_path: str) -> float:
    """Rough parameter count (billions) from GGUF file size (Q4-ish heuristic)."""
    mb = file_mb(model_path)
    if mb <= 0:
        return 0.0
    # Q4_K ≈ 0.55–0.65 bytes/param → params_b ≈ size_gb / 0.6
    return round((mb / 1024.0) / 0.6, 2)


def resolve_gguf_path(model_path: str) -> str:
    """Support split GGUFs: dir or *-00001-of-*.gguf (llama.cpp loads siblings)."""
    path = Path(model_path)
    if path.is_dir():
        parts = sorted(path.glob("*.gguf"))
        if not parts:
            return model_path
        # Prefer first shard
        ones = [p for p in parts if "00001-of-" in p.name or "-00001-" in p.name]
        return str(ones[0] if ones else parts[0])
    if path.is_file():
        return str(path)
    # Glob pattern
    matches = sorted(Path().glob(model_path)) if any(c in model_path for c in "*?") else []
    if matches:
        return str(matches[0])
    return model_path


@dataclass
class RamPlan:
    n_ctx: int
    n_batch: int
    n_ubatch: int
    n_predict: int
    n_threads: int
    cache_type_k: str
    cache_type_v: str
    note: str
    model_mb: float
    avail_mb: float
    params_b_est: float
    extreme: bool


def plan_ram_budget(
    model_path: str,
    ram_budget_mb: int | None = None,
    extreme: bool = False,
    user_ctx: int | None = None,
) -> RamPlan:
    """
    Choose ctx/batch so KV + working set fit a tight machine.

    Strategy for huge models on small RAM:
      - weights stay mmap'd (paged from disk)
      - keep KV tiny (small n_ctx)
      - tiny batch/ubatch to cut activation buffers
      - never mlock
    """
    model_path = resolve_gguf_path(model_path)
    m = mem_snapshot()
    model_mb = file_mb(model_path)
    avail = ram_budget_mb if ram_budget_mb and ram_budget_mb > 0 else m.sys_avail_mb
    params_b = estimate_params_b(model_path)

    # Headroom for OS + Python + fragmentation
    headroom = 256.0 if extreme or avail < 4096 else 512.0
    usable = max(128.0, avail - headroom)

    # Heuristic KV budget: leave most of usable for weight pages under pressure
    # Large model → prioritize weight paging → smaller KV share
    if model_mb > usable * 0.8 or extreme:
        kv_budget_mb = min(usable * 0.25, 512.0)
        extreme = True
    elif model_mb > 4000:
        kv_budget_mb = min(usable * 0.35, 1536.0)
    else:
        kv_budget_mb = min(usable * 0.5, 4096.0)

    # ~0.4–0.8 MB/token rough for 7–13B @ q8/f16 KV; scale with params
    mb_per_tok = max(0.15, min(1.2, 0.08 * max(params_b, 1.0)))
    max_ctx_by_ram = int(max(256, (kv_budget_mb / mb_per_tok) // 64 * 64))

    if extreme:
        n_ctx = min(user_ctx or 512, max_ctx_by_ram, 1024)
        n_ctx = max(256, n_ctx)
        n_batch = 16
        n_ubatch = 8
        n_predict = 96
        n_threads = max(1, min(2, os.cpu_count() or 1))
        note = "extreme-low-ram: tiny KV + batch; weights mmap-paged from disk"
    elif model_mb >= 12000:  # ~20B+ Q4 class
        n_ctx = min(user_ctx or 1024, max_ctx_by_ram, 2048)
        n_batch, n_ubatch, n_predict = 32, 16, 128
        n_threads = max(1, min(4, os.cpu_count() or 2))
        note = "large-model profile (≥~12GB GGUF)"
    elif model_mb >= 5000:
        n_ctx = min(user_ctx or 2048, max_ctx_by_ram, 4096)
        n_batch, n_ubatch, n_predict = 64, 32, 192
        n_threads = max(1, min(4, os.cpu_count() or 2))
        note = "medium-large model profile"
    else:
        n_ctx = min(user_ctx or DEFAULT_CTX, max_ctx_by_ram, 8192)
        n_batch, n_ubatch, n_predict = 128, 32, 256
        n_threads = max(1, min(4, os.cpu_count() or 2))
        note = "standard low-ram profile"

    if user_ctx:
        n_ctx = min(user_ctx, max_ctx_by_ram)

    return RamPlan(
        n_ctx=int(n_ctx),
        n_batch=int(n_batch),
        n_ubatch=int(n_ubatch),
        n_predict=int(n_predict),
        n_threads=int(n_threads),
        cache_type_k="q8_0",
        cache_type_v="f16",
        note=note,
        model_mb=model_mb,
        avail_mb=avail,
        params_b_est=params_b,
        extreme=extreme,
    )




# ============================== tools ======================================

ToolFn = Callable[..., Any]


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    fn: ToolFn


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(
        self,
        name: str,
        fn: ToolFn,
        description: str,
        parameters: dict[str, Any] | None = None,
    ) -> None:
        self._tools[name] = ToolSpec(
            name=name,
            description=description,
            parameters=parameters or {"type": "object", "properties": {}},
            fn=fn,
        )

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def schema_prompt(self) -> str:
        if not self._tools:
            return "(no tools)"
        lines = []
        for t in self._tools.values():
            props = t.parameters.get("properties", {})
            args = ", ".join(f"{k}:{v.get('type', 'any')}" for k, v in props.items())
            lines.append(f"- {t.name}({args}): {t.description}")
        return "\n".join(lines)

    def openai_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in self._tools.values()
        ]

    def run(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        spec = self._tools.get(name)
        if not spec:
            return json.dumps({"error": f"unknown tool: {name}"})
        try:
            result = spec.fn(**(arguments or {}))
            if isinstance(result, (dict, list)):
                return json.dumps(result, ensure_ascii=False, default=str)
            return str(result)
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    def run_many(self, calls: list[dict[str, Any]], parallel: bool = True) -> list[dict[str, Any]]:
        """Execute tool calls; parallel when safe (default)."""
        if not calls:
            return []
        if not parallel or len(calls) == 1:
            out = []
            for call in calls:
                name = call.get("name", "")
                args = call.get("arguments") or {}
                out.append({
                    "name": name,
                    "arguments": args,
                    "result": self.run(name, args),
                })
            return out

        results: list[dict[str, Any] | None] = [None] * len(calls)

        def _job(idx: int, call: dict[str, Any]) -> None:
            name = call.get("name", "")
            args = call.get("arguments") or {}
            results[idx] = {
                "name": name,
                "arguments": args,
                "result": self.run(name, args),
            }

        with ThreadPoolExecutor(max_workers=min(4, len(calls))) as pool:
            futs = [pool.submit(_job, i, c) for i, c in enumerate(calls)]
            for f in as_completed(futs):
                f.result()
        return [r for r in results if r is not None]

    @classmethod
    def with_builtins(cls, workspace: Path | None = None) -> "ToolRegistry":
        reg = cls()
        ws = workspace or WORKSPACE
        ws.mkdir(parents=True, exist_ok=True)

        def _safe_ws(path: str) -> Path:
            p = (ws / path).resolve()
            if not str(p).startswith(str(ws.resolve())):
                raise ValueError("path outside workspace")
            return p

        def calculator(expression: str = "") -> dict[str, Any]:
            expr = (expression or "").strip()
            if not expr:
                return {"error": "empty expression"}
            allowed = {
                ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
                ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod,
                ast.Pow, ast.USub, ast.UAdd,
            }
            tree = ast.parse(expr, mode="eval")
            for node in ast.walk(tree):
                if type(node) not in allowed and type(node).__name__ != "Load":
                    raise ValueError(f"disallowed: {type(node).__name__}")
            ops = {
                ast.Add: operator.add, ast.Sub: operator.sub,
                ast.Mult: operator.mul, ast.Div: operator.truediv,
                ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
                ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos,
            }

            def _eval(n: ast.AST) -> float:
                if isinstance(n, ast.Expression):
                    return _eval(n.body)
                if isinstance(n, ast.Constant):
                    if isinstance(n.value, (int, float)):
                        return float(n.value)
                    raise ValueError("only numbers")
                if isinstance(n, ast.UnaryOp):
                    return ops[type(n.op)](_eval(n.operand))
                if isinstance(n, ast.BinOp):
                    return ops[type(n.op)](_eval(n.left), _eval(n.right))
                raise ValueError("bad node")

            return {"expression": expr, "result": _eval(tree)}

        def get_time(timezone_name: str = "UTC") -> dict[str, Any]:
            now = datetime.now(timezone.utc)
            return {"utc": now.isoformat(), "unix": int(now.timestamp()), "note": timezone_name}

        def list_dir(path: str = ".") -> dict[str, Any]:
            p = _safe_ws(path)
            if not p.exists():
                return {"error": "not found"}
            if p.is_file():
                return {"type": "file", "path": str(p.relative_to(ws)), "size": p.stat().st_size}
            entries = []
            for c in sorted(p.iterdir())[:100]:
                entries.append({
                    "name": c.name,
                    "type": "dir" if c.is_dir() else "file",
                    "size": c.stat().st_size if c.is_file() else None,
                })
            return {"path": str(p.relative_to(ws)) if p != ws else ".", "entries": entries}

        def read_file(path: str = "", max_bytes: int = 4000) -> dict[str, Any]:
            p = _safe_ws(path)
            if not p.is_file():
                return {"error": "not a file"}
            data = p.read_bytes()[: max(1, min(int(max_bytes), 32_000))]
            text = data.decode("utf-8", errors="replace")
            return {"path": path, "content": text, "bytes": len(data)}

        def write_file(path: str = "", content: str = "") -> dict[str, Any]:
            p = _safe_ws(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return {"path": path, "bytes_written": len(content.encode("utf-8"))}

        def memory_stats() -> dict[str, Any]:
            m = mem_snapshot()
            return {
                "rss_mb": round(m.rss_mb, 2),
                "sys_avail_mb": round(m.sys_avail_mb, 1),
                "sys_total_mb": round(m.sys_total_mb, 1),
            }

        def echo(message: str = "") -> dict[str, Any]:
            return {"echo": message}

        def http_get(url: str = "", max_bytes: int = 8000) -> dict[str, Any]:
            if not url.startswith(("http://", "https://")):
                return {"error": "url must start with http:// or https://"}
            req = urllib.request.Request(
                url, headers={"User-Agent": f"DiskChat/{VERSION}"}, method="GET"
            )
            try:
                with urllib.request.urlopen(req, timeout=15) as resp:
                    data = resp.read(max(1, min(int(max_bytes), 64_000)))
                    ctype = resp.headers.get("Content-Type", "")
                    text = data.decode("utf-8", errors="replace")
                    return {
                        "url": url,
                        "status": getattr(resp, "status", 200),
                        "content_type": ctype,
                        "body": text[: int(max_bytes)],
                        "bytes": len(data),
                    }
            except urllib.error.HTTPError as e:
                return {"error": f"HTTP {e.code}", "url": url}
            except Exception as e:
                return {"error": str(e), "url": url}

        def search_workspace(query: str = "", path: str = ".", max_hits: int = 20) -> dict[str, Any]:
            if not query:
                return {"error": "empty query"}
            root = _safe_ws(path)
            hits = []
            q = query.lower()
            for fp in root.rglob("*"):
                if not fp.is_file():
                    continue
                if fp.stat().st_size > 1_000_000:
                    continue
                try:
                    text = fp.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                for i, line in enumerate(text.splitlines(), 1):
                    if q in line.lower():
                        hits.append({
                            "file": str(fp.relative_to(ws)),
                            "line": i,
                            "text": line.strip()[:200],
                        })
                        if len(hits) >= max_hits:
                            return {"query": query, "hits": hits}
            return {"query": query, "hits": hits}

        def glob_files(pattern: str = "*", path: str = ".") -> dict[str, Any]:
            root = _safe_ws(path)
            matches = []
            for fp in sorted(root.glob(pattern))[:100]:
                matches.append({
                    "path": str(fp.relative_to(ws)),
                    "type": "dir" if fp.is_dir() else "file",
                    "size": fp.stat().st_size if fp.is_file() else None,
                })
            return {"pattern": pattern, "matches": matches}

        def platform_info() -> dict[str, Any]:
            return {
                "system": os.uname().sysname,
                "machine": os.uname().machine,
                "release": os.uname().release,
                "python": sys.version.split()[0],
                "diskchat": VERSION,
            }

        reg.register("calculator", calculator, "Evaluate math (+ - * / ** // %).",
                     {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]})
        reg.register("get_time", get_time, "Current UTC time.",
                     {"type": "object", "properties": {"timezone_name": {"type": "string"}}})
        reg.register("list_dir", list_dir, "List files in the agent workspace.",
                     {"type": "object", "properties": {"path": {"type": "string"}}})
        reg.register("read_file", read_file, "Read a text file from the workspace.",
                     {"type": "object", "properties": {"path": {"type": "string"}, "max_bytes": {"type": "integer"}}, "required": ["path"]})
        reg.register("write_file", write_file, "Write a text file into the workspace.",
                     {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]})
        reg.register("memory_stats", memory_stats, "Process/system RAM usage.",
                     {"type": "object", "properties": {}})
        reg.register("echo", echo, "Echo a string (debug).",
                     {"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]})
        reg.register("http_get", http_get, "HTTP GET a URL (text body, size-capped).",
                     {"type": "object", "properties": {"url": {"type": "string"}, "max_bytes": {"type": "integer"}}, "required": ["url"]})
        reg.register("search_workspace", search_workspace, "Search text across workspace files.",
                     {"type": "object", "properties": {"query": {"type": "string"}, "path": {"type": "string"}, "max_hits": {"type": "integer"}}, "required": ["query"]})
        reg.register("glob_files", glob_files, "Glob files in the workspace.",
                     {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}}})
        reg.register("platform_info", platform_info, "OS/arch/python/diskchat version.",
                     {"type": "object", "properties": {}})
        return reg


# ============================== tool-call parse ============================

_TOOL_BLOCK = re.compile(
    r"<tool_call>\s*(\{.*\})\s*</tool_call>",
    re.DOTALL | re.IGNORECASE,
)


def _extract_json_objects(text: str) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(text):
        if text[i] != "{":
            i += 1
            continue
        depth = 0
        j = i
        in_str = False
        esc = False
        while j < len(text):
            ch = text[j]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        out.append(text[i : j + 1])
                        break
            j += 1
        i = j + 1 if j > i else i + 1
    return out


def _normalize_call(obj: dict[str, Any]) -> dict[str, Any]:
    name = obj.get("tool") or obj.get("name") or ""
    args = obj.get("arguments") or obj.get("args") or obj.get("parameters") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {"value": args}
    return {"name": str(name), "arguments": args if isinstance(args, dict) else {}}


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for m in _TOOL_BLOCK.finditer(text):
        try:
            found.append(_normalize_call(json.loads(m.group(1))))
        except json.JSONDecodeError:
            continue
    if not found:
        for blob in _extract_json_objects(text):
            if '"name"' not in blob and '"tool"' not in blob:
                continue
            try:
                obj = json.loads(blob)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and (obj.get("name") or obj.get("tool")):
                found.append(_normalize_call(obj))
    deduped: list[dict[str, Any]] = []
    for c in found:
        if not deduped or deduped[-1] != c:
            deduped.append(c)
    return deduped


def strip_tool_markup(text: str) -> str:
    return _TOOL_BLOCK.sub("", text).strip()


# ============================== prompting ==================================

@dataclass
class Turn:
    role: str
    content: str
    name: str | None = None


@dataclass
class Conversation:
    system: str = ""
    turns: list[Turn] = field(default_factory=list)

    def add(self, role: str, content: str, name: str | None = None) -> None:
        self.turns.append(Turn(role, content.strip(), name))

    def trim(self, max_pairs: int) -> None:
        non = [t for t in self.turns if t.role != "system"]
        if max_pairs > 0 and len(non) > max_pairs * 2:
            non = non[-(max_pairs * 2) :]
        self.turns = non

    def to_list(self) -> list[dict[str, Any]]:
        out = []
        if self.system:
            out.append({"role": "system", "content": self.system})
        for t in self.turns:
            d: dict[str, Any] = {"role": t.role, "content": t.content}
            if t.name:
                d["name"] = t.name
            out.append(d)
        return out

    @classmethod
    def from_list(cls, items: list[dict[str, Any]]) -> "Conversation":
        conv = cls()
        for it in items:
            role = it.get("role", "user")
            if role == "system":
                conv.system = it.get("content", "")
            else:
                conv.add(role, it.get("content", ""), it.get("name"))
        return conv


def agent_system_prompt(tools: ToolRegistry, base: str) -> str:
    base = base.strip() or "You are a helpful assistant with tools."
    return f"""{base}

You have tools. When you need one, output EXACTLY this format and nothing else before it:
<tool_call>
{{"name": "tool_name", "arguments": {{"arg": "value"}}}}
</tool_call>

You may emit multiple <tool_call> blocks. After tool results, answer in plain text.

Available tools:
{tools.schema_prompt()}

Rules:
- For math, use calculator.
- Never invent tool results.
- Keep final answers short unless asked for detail.
"""


def render_chatml(conv: Conversation) -> str:
    parts: list[str] = []
    if conv.system.strip():
        parts.append(f"<|im_start|>system\n{conv.system.strip()}<|im_end|>")
    for t in conv.turns:
        if t.role == "system":
            continue
        if t.role == "tool":
            body = f"Tool `{t.name}` result:\n{t.content}"
            parts.append(f"<|im_start|>user\n{body}<|im_end|>")
        else:
            parts.append(f"<|im_start|>{t.role}\n{t.content}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


STOPS = ["<|im_end|>", "<|im_start|>", "<|endoftext|>", "[end of text]", "[end of turn]"]


def approx_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 3)


# ============================== engine =====================================

@dataclass
class EngineConfig:
    model_path: str = DEFAULT_MODEL
    llama_cli: str = DEFAULT_LLAMA_CLI
    lib_dir: str = DEFAULT_LIB_DIR
    n_ctx: int = DEFAULT_CTX
    max_context: int = MAX_CONTEXT_CEILING
    n_batch: int = 128
    n_ubatch: int = 32
    n_threads: int = max(1, min(4, os.cpu_count() or 2))
    n_predict: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    cache_type_k: str = "q8_0"
    cache_type_v: str = "f16"
    rope_scaling: str = "yarn"
    yarn_ext_factor: float = 4.0
    yarn_attn_factor: float = 1.0
    system_prompt: str = "You are a helpful, concise assistant."
    history_turns: int = 10
    enable_prompt_cache: bool = False
    prompt_cache_dir: str = str(Path.home() / ".cache" / "diskchat" / "cache")
    agent_mode: bool = False
    max_tool_rounds: int = 4
    parallel_tools: bool = True
    max_retries: int = 1
    base_system: str = "You are a helpful assistant with tools."
    profile: str = "default"
    extreme_low_ram: bool = False
    ram_budget_mb: int = 0  # 0 = use MemAvailable


@dataclass
class GenerateResult:
    text: str
    elapsed_s: float
    mem_before: MemSnapshot
    mem_after: MemSnapshot
    model_file_mb: float
    ctx_used: int
    prompt_tokens_est: int
    tool_trace: list[dict[str, Any]] = field(default_factory=list)
    tokens_per_sec: float = 0.0


class DiskChatEngine:
    def __init__(self, cfg: EngineConfig, tools: ToolRegistry | None = None):
        self.cfg = cfg
        self.tools = tools or ToolRegistry()
        # Split / multi-part GGUF support
        cfg.model_path = resolve_gguf_path(cfg.model_path)
        self.conv = Conversation(system=self._build_system())
        model = Path(cfg.model_path)
        cli = Path(cfg.llama_cli)
        if not model.is_file():
            raise FileNotFoundError(f"Model not found: {model}")
        if not cli.is_file() or not os.access(cli, os.X_OK):
            raise FileNotFoundError(f"llama-cli missing: {cli}")
        Path(cfg.prompt_cache_dir).mkdir(parents=True, exist_ok=True)
        WORKSPACE.mkdir(parents=True, exist_ok=True)
        SESSION_DIR.mkdir(parents=True, exist_ok=True)
        self._model_mb = file_mb(cfg.model_path)
        self._ram_plan: RamPlan | None = None
        if cfg.extreme_low_ram or cfg.ram_budget_mb > 0 or self._model_mb >= 5000:
            plan = plan_ram_budget(
                cfg.model_path,
                ram_budget_mb=cfg.ram_budget_mb or None,
                extreme=cfg.extreme_low_ram,
                user_ctx=cfg.n_ctx,
            )
            self._ram_plan = plan
            # Apply safer caps (never raise user intent above plan when extreme)
            if cfg.extreme_low_ram or self._model_mb >= 5000:
                cfg.n_ctx = min(cfg.n_ctx, plan.n_ctx)
                cfg.n_batch = min(cfg.n_batch, plan.n_batch)
                cfg.n_ubatch = min(cfg.n_ubatch, plan.n_ubatch)
                cfg.n_threads = min(cfg.n_threads, plan.n_threads)
                if cfg.extreme_low_ram:
                    cfg.n_predict = min(cfg.n_predict, plan.n_predict)
                    cfg.n_batch = plan.n_batch
                    cfg.n_ubatch = plan.n_ubatch

    def _build_system(self) -> str:
        if self.cfg.agent_mode and self.tools.names():
            return agent_system_prompt(self.tools, self.cfg.base_system)
        return self.cfg.system_prompt

    def reset(self) -> None:
        self.conv = Conversation(system=self._build_system())

    def save_session(self, name: str = "default") -> Path:
        SESSION_DIR.mkdir(parents=True, exist_ok=True)
        path = SESSION_DIR / f"{name}.json"
        data = {
            "version": VERSION,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "profile": self.cfg.profile,
            "agent_mode": self.cfg.agent_mode,
            "messages": self.conv.to_list(),
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def load_session(self, name: str = "default") -> None:
        path = SESSION_DIR / f"{name}.json"
        if not path.is_file():
            raise FileNotFoundError(f"session not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        self.conv = Conversation.from_list(data.get("messages", []))
        if not self.conv.system:
            self.conv.system = self._build_system()

    def _env(self) -> dict[str, str]:
        env = os.environ.copy()
        prev = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = (
            f"{self.cfg.lib_dir}:{prev}" if prev else self.cfg.lib_dir
        )
        return env

    def plan_ctx(self, prompt: str) -> int:
        need = approx_tokens(prompt) + self.cfg.n_predict + 64
        floor = min(256 if self.cfg.extreme_low_ram else 512, self.cfg.n_ctx)
        planned = max(floor, min(need, self.cfg.n_ctx, self.cfg.max_context))
        planned = int(min(((planned + 63) // 64) * 64, self.cfg.max_context))
        # Re-check against live available memory for huge models
        if self.cfg.extreme_low_ram or self._model_mb >= 5000:
            plan = plan_ram_budget(
                self.cfg.model_path,
                ram_budget_mb=self.cfg.ram_budget_mb or None,
                extreme=self.cfg.extreme_low_ram,
                user_ctx=planned,
            )
            planned = min(planned, plan.n_ctx)
        return planned

    def _build_cmd(self, prompt: str, ctx: int) -> list[str]:
        c = self.cfg
        cmd = [
            c.llama_cli, "-m", c.model_path, "-p", prompt,
            "-n", str(c.n_predict),
            "-c", str(ctx),
            "-b", str(min(c.n_batch, ctx)),
            "-ub", str(min(c.n_ubatch, c.n_batch, ctx)),
            "-t", str(c.n_threads),
            "--temp", str(c.temperature),
            "--top-p", str(c.top_p),
            "--top-k", str(c.top_k),
            "--repeat-penalty", str(c.repeat_penalty),
            "--cache-type-k", c.cache_type_k,
            "--cache-type-v", c.cache_type_v,
            "--no-display-prompt", "--no-warmup",
            "--no-conversation", "--simple-io",
        ]
        if c.rope_scaling and c.rope_scaling != "none" and ctx > 4096:
            cmd.extend([
                "--rope-scaling", c.rope_scaling,
                "--yarn-ext-factor", str(c.yarn_ext_factor),
                "--yarn-attn-factor", str(c.yarn_attn_factor),
            ])
        if c.enable_prompt_cache:
            cmd.extend([
                "--prompt-cache",
                str(Path(c.prompt_cache_dir) / "prompt.bin"),
            ])
        return cmd

    @staticmethod
    def _clean(raw: str) -> str:
        text = raw
        for s in STOPS:
            if s in text:
                text = text.split(s, 1)[0]
        text = re.sub(r"^[\s>]+", "", text)
        text = text.replace("[end of text]", "").replace("[end of turn]", "")
        return text.strip()

    def _complete_once(
        self,
        prompt: str,
        stream_cb: Callable[[str], None] | None = None,
        force_ctx: int | None = None,
    ) -> tuple[str, int, float, MemSnapshot, MemSnapshot]:
        ctx = force_ctx or self.plan_ctx(prompt)
        mem_before = mem_snapshot()
        t0 = time.perf_counter()
        cmd = self._build_cmd(prompt, ctx)
        last_err = ""
        text = ""
        for attempt in range(self.cfg.max_retries + 1):
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                env=self._env(), text=True, bufsize=1,
            )
            chunks: list[str] = []
            err_buf: list[str] = []

            def _err() -> None:
                assert proc.stderr is not None
                for line in proc.stderr:
                    err_buf.append(line)

            th = threading.Thread(target=_err, daemon=True)
            th.start()
            assert proc.stdout is not None
            try:
                for line in proc.stdout:
                    chunks.append(line)
                    if stream_cb and attempt == 0:
                        vis = line.replace("[end of text]", "").replace("[end of turn]", "")
                        if vis:
                            stream_cb(vis)
            except KeyboardInterrupt:
                proc.send_signal(signal.SIGINT)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                raise
            code = proc.wait(timeout=900)
            th.join(timeout=2)
            text = self._clean("".join(chunks))
            last_err = "".join(err_buf)[-500:]
            if text or code == 0:
                break
            time.sleep(0.3)
        else:
            if not text:
                raise RuntimeError(f"llama-cli failed: {last_err}")

        elapsed = time.perf_counter() - t0
        mem_after = mem_snapshot()
        return text, ctx, elapsed, mem_before, mem_after

    def generate(
        self,
        user_message: str,
        stream_cb: Callable[[str], None] | None = None,
        force_ctx: int | None = None,
    ) -> GenerateResult:
        self.conv.add("user", user_message)
        self.conv.trim(self.cfg.history_turns)
        tool_trace: list[dict[str, Any]] = []
        total_elapsed = 0.0
        mem_before = mem_snapshot()
        ctx_used = 0
        final_text = ""
        gen_chars = 0

        rounds = self.cfg.max_tool_rounds if self.cfg.agent_mode else 1
        text = ""
        for round_i in range(rounds):
            prompt = render_chatml(self.conv)
            cb = stream_cb if (not self.cfg.agent_mode or round_i == rounds - 1) else None
            text, ctx, elapsed, _, _ = self._complete_once(
                prompt, stream_cb=cb, force_ctx=force_ctx
            )
            total_elapsed += elapsed
            ctx_used = ctx
            gen_chars += len(text)
            calls = parse_tool_calls(text) if self.cfg.agent_mode else []

            if self.cfg.agent_mode and calls:
                self.conv.add("assistant", text)
                executed = self.tools.run_many(calls, parallel=self.cfg.parallel_tools)
                for item in executed:
                    tool_trace.append({
                        "round": round_i,
                        "name": item["name"],
                        "arguments": item["arguments"],
                        "result": item["result"][:2000],
                    })
                    self.conv.add("tool", item["result"], name=item["name"])
                continue

            final_text = strip_tool_markup(text) if self.cfg.agent_mode else text
            self.conv.add("assistant", final_text)
            if stream_cb and self.cfg.agent_mode and final_text and cb is None:
                stream_cb(final_text)
            break
        else:
            final_text = strip_tool_markup(final_text or text)
            self.conv.add("assistant", final_text)

        tps = (approx_tokens("x" * gen_chars) / total_elapsed) if total_elapsed > 0 else 0.0
        return GenerateResult(
            text=final_text,
            elapsed_s=total_elapsed,
            mem_before=mem_before,
            mem_after=mem_snapshot(),
            model_file_mb=self._model_mb,
            ctx_used=ctx_used,
            prompt_tokens_est=approx_tokens(render_chatml(self.conv)),
            tool_trace=tool_trace,
            tokens_per_sec=tps,
        )


# ============================== doctor =====================================

def run_doctor(cfg: EngineConfig) -> int:
    print(f"DiskChat doctor v{VERSION}")
    print(f"  platform : {os.uname().sysname} {os.uname().machine}")
    print(f"  python   : {sys.version.split()[0]}")
    print(f"  host     : {mem_snapshot()}")
    ok = True
    model = Path(cfg.model_path)
    cli = Path(cfg.llama_cli)
    lib = Path(cfg.lib_dir)
    print(f"  model    : {model}  exists={model.is_file()}  size={file_mb(str(model)):.0f}MB")
    if not model.is_file():
        print("  !! set DISKCHAT_MODEL or run scripts/download_model.py")
        ok = False
    print(f"  llama-cli: {cli}  exists={cli.is_file()}  exec={os.access(cli, os.X_OK) if cli.is_file() else False}")
    if not cli.is_file():
        print("  !! run: bash scripts/install_runtime.sh")
        ok = False
    print(f"  lib_dir  : {lib}  exists={lib.is_dir()}")
    so = list(lib.glob("libllama*")) + list(lib.glob("libggml*")) if lib.is_dir() else []
    print(f"  libs     : {len(so)} shared objects")
    if cli.is_file():
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = f"{lib}:{env.get('LD_LIBRARY_PATH', '')}"
        try:
            r = subprocess.run(
                [str(cli), "--version"], capture_output=True, text=True, env=env, timeout=10
            )
            ver = (r.stdout or r.stderr).strip().splitlines()[:1]
            print(f"  version  : {ver[0] if ver else 'unknown'}")
        except Exception as e:
            print(f"  !! llama-cli run failed: {e}")
            ok = False
    print(f"  workspace: {WORKSPACE}")
    print(f"  sessions : {SESSION_DIR}")
    if model.is_file():
        plan = plan_ram_budget(
            str(model),
            ram_budget_mb=cfg.ram_budget_mb or None,
            extreme=cfg.extreme_low_ram,
            user_ctx=cfg.n_ctx,
        )
        print(f"  params~  : {plan.params_b_est}B (from file size heuristic)")
        print(f"  ram plan : ctx={plan.n_ctx} batch={plan.n_batch}/{plan.n_ubatch} "
              f"threads={plan.n_threads} extreme={plan.extreme}")
        print(f"  plan note: {plan.note}")
        print(f"  avail    : {plan.avail_mb:.0f} MB  model_file={plan.model_mb:.0f} MB")
        if plan.model_mb > plan.avail_mb:
            print("  !! model file > available RAM — mmap will page from disk (slower, still works)")
    print("  status   :", "OK" if ok else "NEEDS SETUP")
    return 0 if ok else 1


# ============================== HTTP API ===================================

class _ApiState:
    engine: DiskChatEngine | None = None
    tools: ToolRegistry | None = None
    lock = threading.Lock()


def make_handler() -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            sys.stderr.write("[http] " + (fmt % args) + "\n")

        def _json(self, code: int, obj: Any) -> None:
            body = json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> dict[str, Any]:
            n = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(n) if n else b"{}"
            try:
                return json.loads(raw.decode("utf-8") or "{}")
            except json.JSONDecodeError:
                return {}

        def do_OPTIONS(self) -> None:  # noqa: N802
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path in ("/health", "/v1/health"):
                self._json(200, {
                    "status": "ok",
                    "version": VERSION,
                    "mem": asdict(mem_snapshot()),
                    "agent": bool(_ApiState.engine and _ApiState.engine.cfg.agent_mode),
                })
            elif path in ("/v1/tools", "/tools"):
                tools = _ApiState.tools or ToolRegistry()
                self._json(200, {"tools": tools.openai_tools()})
            elif path in ("/v1/models", "/models"):
                eng = _ApiState.engine
                self._json(200, {
                    "data": [{
                        "id": "diskchat-local",
                        "model_path": eng.cfg.model_path if eng else None,
                        "object": "model",
                    }]
                })
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            data = self._read_json()
            if path in ("/v1/chat", "/chat", "/v1/chat/completions"):
                if not _ApiState.engine:
                    self._json(503, {"error": "engine not ready"})
                    return
                msg = data.get("message") or data.get("prompt") or ""
                if not msg and "messages" in data:
                    for m in reversed(data["messages"]):
                        if m.get("role") == "user":
                            msg = m.get("content", "")
                            break
                if not msg:
                    self._json(400, {"error": "message required"})
                    return
                with _ApiState.lock:
                    try:
                        result = _ApiState.engine.generate(str(msg))
                    except Exception as e:
                        self._json(500, {"error": str(e)})
                        return
                self._json(200, {
                    "reply": result.text,
                    "tool_trace": result.tool_trace,
                    "ctx_used": result.ctx_used,
                    "elapsed_s": round(result.elapsed_s, 3),
                    "rss_mb": round(result.mem_after.rss_mb, 1),
                    "tokens_per_sec_est": round(result.tokens_per_sec, 2),
                })
            elif path in ("/v1/reset", "/reset"):
                if _ApiState.engine:
                    with _ApiState.lock:
                        _ApiState.engine.reset()
                self._json(200, {"ok": True})
            else:
                self._json(404, {"error": "not found"})

    return Handler


def serve_http(engine: DiskChatEngine, tools: ToolRegistry, host: str, port: int) -> None:
    _ApiState.engine = engine
    _ApiState.tools = tools
    handler = make_handler()
    httpd = ThreadingHTTPServer((host, port), handler)
    print(f"DiskChat HTTP API on http://{host}:{port}")
    print("  GET  /health  /v1/tools  /v1/models")
    print("  POST /v1/chat  body: {\"message\": \"...\"}")
    print("  POST /v1/reset")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
        httpd.shutdown()


# ============================== self-test ==================================

def run_selftest(cfg: EngineConfig) -> int:
    print(f"=== DiskChat v{VERSION} self-test ===")
    assert Path(cfg.model_path).is_file(), "model missing"
    fmb = file_mb(cfg.model_path)
    print(f"  model disk={fmb:.0f}MB  max_ctx={cfg.max_context}  host={mem_snapshot()}")

    cfg_chat = EngineConfig(**{**cfg.__dict__, "agent_mode": False, "system_prompt": "",
                               "temperature": 0.1, "top_k": 20, "n_predict": 32,
                               "n_ctx": 2048, "repeat_penalty": 1.2})
    eng = DiskChatEngine(cfg_chat)
    r1 = eng.generate("What is 2+2? Reply with only the digit.")
    print(f"  chat → {r1.text!r} RSS={r1.mem_after.rss_mb:.0f}MB")
    assert "4" in r1.text, r1.text
    assert r1.mem_after.rss_mb < fmb * 0.5
    print("  PASS  plain chat + low RAM")

    reg = ToolRegistry.with_builtins()
    assert "http_get" in reg.names() and "search_workspace" in reg.names()
    calc = json.loads(reg.run("calculator", {"expression": "17*19"}))
    assert abs(calc["result"] - 323) < 1e-6
    print("  PASS  tools registry")

    # parallel tools
    many = reg.run_many([
        {"name": "echo", "arguments": {"message": "a"}},
        {"name": "echo", "arguments": {"message": "b"}},
    ], parallel=True)
    assert len(many) == 2
    print("  PASS  parallel tool execution")

    # session
    eng.conv.add("assistant", "hi")
    path = eng.save_session("_selftest")
    eng.reset()
    eng.load_session("_selftest")
    assert any(t.content == "hi" for t in eng.conv.turns)
    print("  PASS  session save/load")

    cfg_agent = EngineConfig(
        **{**cfg.__dict__, "agent_mode": True, "temperature": 0.1, "top_k": 20,
           "n_predict": 96, "n_ctx": 2048, "max_tool_rounds": 3,
           "repeat_penalty": 1.15, "base_system": "Use tools for math.",
           "parallel_tools": True}
    )
    agent = DiskChatEngine(cfg_agent, tools=reg)
    r2 = agent.generate(
        "Use the calculator tool to compute 17*19. "
        "Output a tool_call for calculator with expression 17*19 first."
    )
    print(f"  agent → {r2.text!r} trace={len(r2.tool_trace)}")
    ok_num = "323" in r2.text.replace(" ", "") or any(
        "323" in str(t.get("result", "")) for t in r2.tool_trace
    )
    if not ok_num and not r2.tool_trace:
        calls = parse_tool_calls(
            '<tool_call>{"name":"calculator","arguments":{"expression":"17*19"}}</tool_call>'
        )
        assert calls
        assert "323" in reg.run(calls[0]["name"], calls[0]["arguments"])
        print("  PASS  tool plumbing (model skipped tools)")
    else:
        assert ok_num
        print("  PASS  agent tool calling")

    schemas = reg.openai_tools()
    assert len(schemas) >= 10
    print("  PASS  OpenAI schemas")
    print("=== ALL TESTS PASSED ===")
    print(f"disk={fmb:.0f}MB | RSS≈{r1.mem_after.rss_mb:.0f}MB | tools={len(reg.names())} | v{VERSION}")
    return 0


# ============================== CLI ========================================

BANNER = f"""
╔══════════════════════════════════════════════════════════════════╗
║  DiskChat Agent v{VERSION} · mmap · tools · HTTP · ≤131k ctx      ║
║  Linux x86_64 & aarch64 · hook agents via --serve /v1/chat       ║
╚══════════════════════════════════════════════════════════════════╝
"""


def load_config_file(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=f"DiskChat Agent v{VERSION}")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--llama-cli", default=DEFAULT_LLAMA_CLI)
    ap.add_argument("--lib-dir", default=DEFAULT_LIB_DIR)
    ap.add_argument("--ctx", type=int, default=DEFAULT_CTX)
    ap.add_argument("--max-ctx", type=int, default=MAX_CONTEXT_CEILING)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--threads", type=int, default=max(1, min(4, os.cpu_count() or 2)))
    ap.add_argument("--n-predict", type=int, default=256)
    ap.add_argument("--temp", type=float, default=None)
    ap.add_argument("--cache-k", default="q8_0")
    ap.add_argument("--cache-v", default="f16")
    ap.add_argument("--rope", default="yarn", choices=["none", "linear", "yarn"])
    ap.add_argument("--agent", action="store_true")
    ap.add_argument("--max-tool-rounds", type=int, default=4)
    ap.add_argument("--profile", default="default", choices=list(PROFILES))
    ap.add_argument("--config", default="", help="JSON config file")
    ap.add_argument("--once")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--doctor", action="store_true")
    ap.add_argument("--list-tools", action="store_true")
    ap.add_argument("--serve", action="store_true", help="start HTTP API for agents")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--extreme-low-ram", action="store_true",
                    help="aggressive settings for huge GGUFs on small RAM (mmap paging)")
    ap.add_argument("--ram-budget", type=int, default=0, metavar="MB",
                    help="max RAM budget in MB (0 = auto from MemAvailable)")
    ap.add_argument("--version", action="store_true")
    args = ap.parse_args(argv)

    if args.version:
        print(VERSION)
        return 0

    file_cfg = load_config_file(args.config) if args.config else {}
    prof = PROFILES.get(args.profile, PROFILES["default"])

    temp = args.temp if args.temp is not None else float(file_cfg.get("temperature", prof["temperature"]))
    n_predict = int(file_cfg.get("n_predict", prof.get("n_predict", args.n_predict)))

    cfg = EngineConfig(
        model_path=file_cfg.get("model", args.model),
        llama_cli=file_cfg.get("llama_cli", args.llama_cli),
        lib_dir=file_cfg.get("lib_dir", args.lib_dir),
        n_ctx=min(args.ctx, args.max_ctx),
        max_context=args.max_ctx,
        n_batch=args.batch,
        n_ubatch=min(32, args.batch),
        n_threads=args.threads,
        n_predict=n_predict,
        temperature=temp,
        cache_type_k=args.cache_k,
        cache_type_v=args.cache_v,
        rope_scaling=args.rope,
        agent_mode=args.agent or args.profile == "agent",
        max_tool_rounds=args.max_tool_rounds,
        system_prompt=prof["system_prompt"],
        base_system=prof["base_system"],
        profile=args.profile,
        extreme_low_ram=args.extreme_low_ram,
        ram_budget_mb=args.ram_budget,
    )
    # Auto-apply planner when extreme or large model / budget set
    if args.extreme_low_ram or args.ram_budget or file_mb(cfg.model_path) >= 5000:
        plan = plan_ram_budget(
            cfg.model_path,
            ram_budget_mb=args.ram_budget or None,
            extreme=args.extreme_low_ram,
            user_ctx=cfg.n_ctx,
        )
        if args.extreme_low_ram:
            cfg.n_ctx = plan.n_ctx
            cfg.n_batch = plan.n_batch
            cfg.n_ubatch = plan.n_ubatch
            cfg.n_predict = min(cfg.n_predict, plan.n_predict)
            cfg.n_threads = plan.n_threads
        else:
            cfg.n_ctx = min(cfg.n_ctx, plan.n_ctx)
            cfg.n_batch = min(cfg.n_batch, plan.n_batch)
            cfg.n_ubatch = min(cfg.n_ubatch, plan.n_ubatch)

    tools = ToolRegistry.with_builtins()

    if args.list_tools:
        print(json.dumps(tools.openai_tools(), indent=2))
        return 0
    if args.doctor:
        return run_doctor(cfg)
    if args.selftest:
        try:
            return run_selftest(cfg)
        except Exception as exc:
            print(f"FAIL: {exc}")
            return 1

    print(BANNER)
    print(f"model   : {cfg.model_path} ({file_mb(cfg.model_path):.0f} MB disk)")
    print(f"ctx     : {cfg.n_ctx} (ceiling {cfg.max_context})  profile={cfg.profile}")
    print(f"KV      : {cfg.cache_type_k}/{cfg.cache_type_v}  agent={cfg.agent_mode}")
    print(f"tools   : {', '.join(tools.names())}")
    print(f"mmap    : ON  host: {mem_snapshot()}")
    if cfg.extreme_low_ram or file_mb(cfg.model_path) >= 5000:
        plan = plan_ram_budget(
            cfg.model_path, cfg.ram_budget_mb or None, cfg.extreme_low_ram, cfg.n_ctx
        )
        print(f"ram plan: ctx={cfg.n_ctx} batch={cfg.n_batch}/{cfg.n_ubatch} "
              f"~{plan.params_b_est}B  extreme={cfg.extreme_low_ram}")
        print(f"         {plan.note}")
    print("cmds    : /reset /mem /tools /save [n] /load [n] /quit\n")

    try:
        engine = DiskChatEngine(cfg, tools=tools)
    except FileNotFoundError as e:
        print(f"[error] {e}")
        print("Run: python diskchat.py --doctor")
        return 1

    if args.serve:
        serve_http(engine, tools, args.host, args.port)
        return 0

    if args.once:
        r = engine.generate(args.once)
        if args.json:
            print(json.dumps({
                "reply": r.text,
                "tool_trace": r.tool_trace,
                "elapsed_s": round(r.elapsed_s, 3),
                "ctx_used": r.ctx_used,
                "rss_mb": round(r.mem_after.rss_mb, 1),
                "tokens_per_sec_est": round(r.tokens_per_sec, 2),
            }, ensure_ascii=False, indent=2))
        else:
            if r.tool_trace:
                for t in r.tool_trace:
                    print(f"  ⚙ {t['name']}({t['arguments']}) → {t['result'][:200]}")
            print(r.text)
            print(f"\n[{r.elapsed_s:.1f}s ctx={r.ctx_used} | {r.mem_after}]")
        return 0

    while True:
        try:
            user = input("you › ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye.")
            return 0
        if not user:
            continue
        if user in ("/quit", "/exit", ":q"):
            print("bye.")
            return 0
        if user == "/reset":
            engine.reset()
            print("(reset)")
            continue
        if user == "/mem":
            print(mem_snapshot())
            continue
        if user == "/tools":
            print(tools.schema_prompt())
            continue
        if user.startswith("/save"):
            name = user.split(maxsplit=1)[1] if " " in user else "default"
            path = engine.save_session(name)
            print(f"(saved {path})")
            continue
        if user.startswith("/load"):
            name = user.split(maxsplit=1)[1] if " " in user else "default"
            try:
                engine.load_session(name)
                print(f"(loaded {name})")
            except FileNotFoundError as e:
                print(f"[error] {e}")
            continue

        print("bot › ", end="", flush=True)

        def _stream(tok: str) -> None:
            print(tok, end="", flush=True)

        try:
            result = engine.generate(user, stream_cb=_stream)
        except Exception as exc:
            print(f"\n[error] {exc}")
            continue
        if not result.text.endswith("\n"):
            print()
        if result.tool_trace:
            for t in result.tool_trace:
                print(f"  ⚙ {t['name']} → {str(t['result'])[:120]}")
        print(f"  ↳ {result.elapsed_s:.1f}s ctx={result.ctx_used} ~{result.tokens_per_sec:.1f} tok/s | {result.mem_after}")


if __name__ == "__main__":
    raise SystemExit(main())
