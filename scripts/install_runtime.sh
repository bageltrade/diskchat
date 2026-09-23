#!/usr/bin/env bash
# Install llama.cpp runtime for DiskChat (Linux x86_64 / aarch64)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN_DIR="${DISKCHAT_BIN_DIR:-$ROOT/bin}"
TAG="${LLAMA_CPP_TAG:-b11140}"
ARCH="$(uname -m)"
OS="$(uname -s | tr '[:upper:]' '[:lower:]')"

mkdir -p "$BIN_DIR" "$ROOT/.cache"

pick_asset() {
  case "$OS-$ARCH" in
    linux-x86_64|linux-amd64)
      echo "llama-${TAG}-bin-ubuntu-x64.tar.gz"
      ;;
    linux-aarch64|linux-arm64)
      echo "llama-${TAG}-bin-ubuntu-arm64.tar.gz"
      ;;
    *)
      echo "Unsupported platform: $OS $ARCH" >&2
      echo "Supported: Linux x86_64 (amd64), Linux aarch64 (arm64)" >&2
      exit 1
      ;;
  esac
}

ASSET="$(pick_asset)"
URL="https://github.com/ggml-org/llama.cpp/releases/download/${TAG}/${ASSET}"
ARCHIVE="$ROOT/.cache/${ASSET}"

echo "==> Platform: $OS $ARCH"
echo "==> Downloading $ASSET"
if command -v curl >/dev/null 2>&1; then
  curl -fL --retry 3 -o "$ARCHIVE" "$URL"
elif command -v wget >/dev/null 2>&1; then
  wget -O "$ARCHIVE" "$URL"
else
  echo "Need curl or wget" >&2
  exit 1
fi

echo "==> Extracting into $BIN_DIR"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
tar -xzf "$ARCHIVE" -C "$TMP"

# Find llama-cli in extracted tree
CLI=$(find "$TMP" -type f -name 'llama-cli' | head -1)
if [ -z "$CLI" ]; then
  echo "llama-cli not found in archive" >&2
  exit 1
fi
SRCDIR=$(dirname "$CLI")
# Copy cli + shared libs
cp -a "$SRCDIR"/. "$BIN_DIR"/
chmod +x "$BIN_DIR"/llama-cli 2>/dev/null || true
find "$BIN_DIR" -type f -name 'llama-*' -exec chmod +x {} + 2>/dev/null || true

# Smoke check
export LD_LIBRARY_PATH="$BIN_DIR:${LD_LIBRARY_PATH:-}"
if "$BIN_DIR/llama-cli" --version >/dev/null 2>&1; then
  echo "==> OK: $($BIN_DIR/llama-cli --version 2>&1 | head -1)"
else
  echo "==> Installed binaries to $BIN_DIR (version check skipped)"
fi

echo ""
echo "Runtime ready."
echo "  export DISKCHAT_LLAMA_CLI=$BIN_DIR/llama-cli"
echo "  export DISKCHAT_LIB_DIR=$BIN_DIR"
echo "  export LD_LIBRARY_PATH=$BIN_DIR:\$LD_LIBRARY_PATH"
