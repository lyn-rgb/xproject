#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-$ROOT/.venv}"
GPU_NUM="${GPU_NUM:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

if [ ! -d "$VENV" ]; then
  echo "Virtualenv not found at $VENV. Run scripts/uv_setup.sh first."
  exit 1
fi

source "$VENV/bin/activate"

CKPT_DIR="${CKPT_DIR:-models/SoulX-FlashTalk-14B}"
WAV2VEC_DIR="${WAV2VEC_DIR:-models/chinese-wav2vec2-base}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-7861}"

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" torchrun --nproc_per_node="$GPU_NUM" \
  "$ROOT/src/avatar/webrtc_stream_server.py" \
  --ckpt_dir "$CKPT_DIR" \
  --wav2vec_dir "$WAV2VEC_DIR" \
  --host "$HOST" \
  --port "$PORT"
