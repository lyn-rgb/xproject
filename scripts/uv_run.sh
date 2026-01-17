#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-$ROOT/.venv}"

if [ ! -d "$VENV" ]; then
  echo "Virtualenv not found at $VENV. Run scripts/uv_setup.sh first."
  exit 1
fi

source "$VENV/bin/activate"

MODE="${1:-help}"
shift || true

case "$MODE" in
  avatar)
    python "$ROOT/src/avatar/webrtc_stream_server.py" "$@"
    ;;
  thinker)
    python "$ROOT/src/thinker/web_demo.py" "$@"
    ;;
  help|*)
    cat <<'EOF'
Usage:
  scripts/uv_run.sh avatar [args...]
  scripts/uv_run.sh thinker [args...]

Examples:
  scripts/uv_run.sh avatar --ckpt_dir models/SoulX-FlashTalk-14B --wav2vec_dir models/chinese-wav2vec2-base
  scripts/uv_run.sh thinker --avatar-server http://127.0.0.1:7861 --avatar-cond-image examples/man.png
EOF
    ;;
esac
