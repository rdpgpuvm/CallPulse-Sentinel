#!/usr/bin/env bash
# ============================================================================
# STEP 0 of the 3-step assembly line — one-shot server runner (v4 real-time).
#
# Usage (AMD dev portal / ROCm Jupyter):
#   Tab 1: open a Terminal  ->  bash run_vllm_server.sh   (leave it running)
#   Tab 2: open call_moderator_v4_realtime.ipynb and run cells one by one
#
# What it does:
#   1. Installs every notebook + server python requirement
#      (torch is deliberately NOT touched — the lab ships a ROCm build)
#   2. Repairs the starlette/fastapi conflict that `pip install mcp` causes
#   3. Launches vLLM as an OpenAI-compatible server (Ctrl-C stops it)
#
# Override defaults via env, e.g.:  PORT=8001 bash run_vllm_server.sh
# ============================================================================
set -euo pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B-Instruct-2507}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-call-moderator-llm}"
API_KEY="${API_KEY:-local-key-123}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"

echo "==> [1/3] Installing requirements (torch untouched: ROCm build already present)"
python3 -m pip install -q --root-user-action=ignore \
    openai mcp httpx faster-whisper librosa soundfile numpy pandas
# MUST run after mcp: pin starlette back to what vLLM's fastapi supports
python3 -m pip install -q --root-user-action=ignore "starlette<0.49"

echo "==> [2/4] Sanity-checking vLLM imports (catches the Router on_startup crash early)"
python3 - <<'PY'
import fastapi, starlette, vllm
print(f"    vllm {vllm.__version__} | fastapi {fastapi.__version__} | "
      f"starlette {starlette.__version__} -> OK")
PY

echo "==> [3/4] Clearing stale vLLM processes + fitting GPU memory budget"
# Kill leftover 'vllm serve' processes from earlier attempts — they hold VRAM forever.
STALE_PIDS=$(pgrep -f "vllm serve" || true)
if [ -n "${STALE_PIDS}" ]; then
    echo "    killing stale vLLM PID(s): ${STALE_PIDS}"
    kill -9 ${STALE_PIDS} 2>/dev/null || true
    sleep 8   # give the driver a moment to reclaim the VRAM
fi
# Auto-fit: never request more than ~90% of the VRAM that is ACTUALLY free right now.
GPU_MEMORY_UTILIZATION=$(REQUESTED="${GPU_MEMORY_UTILIZATION}" python3 - <<'PY'
import os, torch
free_bytes, total_bytes = torch.cuda.mem_get_info()          # works on ROCm (HIP) too
requested = float(os.environ["REQUESTED"])
fitted = max(0.05, min(requested, round((free_bytes * 0.9) / total_bytes, 2)))
print(f"{fitted:.2f}")
import sys
print(f"    free VRAM {free_bytes/2**30:.1f}/{total_bytes/2**30:.1f} GiB -> "
      f"gpu-memory-utilization {fitted:.2f} (requested {requested:.2f})", file=sys.stderr)
PY
)

echo "==> [4/4] Launching vLLM — leave this terminal open"
echo "    First run downloads the model (~8 GB). Wait for 'Application startup complete',"
echo "    then run the notebook in your other tab."
echo
exec env VLLM_USE_TRITON_FLASH_ATTN=0 vllm serve "${MODEL_ID}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --api-key "${API_KEY}" \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
