#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-$ROOT/.venv}"
PYTHON="${PYTHON:-python3}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. Install it first: https://docs.astral.sh/uv/"
  exit 1
fi

echo "Creating venv at: $VENV"
uv venv "$VENV" --python "$PYTHON"

echo "Installing requirements..."
uv pip install -r "$ROOT/requirements_uv.txt"

cat <<'EOF'
Done.
Note: install torch separately for your CUDA/Metal runtime.
Example (CUDA 12.1):
  uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
EOF
