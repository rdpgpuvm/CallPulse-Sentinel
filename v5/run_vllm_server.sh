#!/usr/bin/env bash
# ============================================================================
# v5 one-shot runner: deps repair + LIVE GUI DASHBOARD + vLLM server.
#
# Usage (AMD dev portal / ROCm Jupyter):
#   Tab 1: open a Terminal  ->  bash run_vllm_server.sh   (leave it running)
#   Browser: open  <your Jupyter base URL>/proxy/7860/    (the live dashboard)
#   Tab 2: run call_moderator_v5_gui.ipynb cells one by one
#
# Steps: [1] pip requirements (ROCm torch untouched) + starlette pin
#        [2] vLLM import sanity check
#        [3] start the GUI dashboard (gui_server.py) in the background
#        [4] kill stale vLLM processes + fit GPU budget to free VRAM
#        [5] launch vLLM (Ctrl-C stops it; the GUI keeps running)
# ============================================================================
set -euo pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B-Instruct-2507}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-call-moderator-llm}"
API_KEY="${API_KEY:-local-key-123}"
PORT="${PORT:-8000}"
GUI_PORT="${GUI_PORT:-7860}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "==> [1/5] Installing requirements (torch untouched: ROCm build already present)"
python3 -m pip install -q --root-user-action=ignore \
    openai mcp httpx faster-whisper librosa soundfile numpy pandas scipy uvicorn
python3 -m pip install -q --root-user-action=ignore "starlette<0.49"

echo "==> [2/5] Sanity-checking vLLM imports"
python3 - <<'PY'
import fastapi, starlette, vllm
print(f"    vllm {vllm.__version__} | fastapi {fastapi.__version__} | "
      f"starlette {starlette.__version__} -> OK")
PY

echo "==> [3/5] Starting the live GUI dashboard on port ${GUI_PORT}"
pkill -f "gui_server.py" 2>/dev/null || true
sleep 1
nohup python3 "${SCRIPT_DIR}/gui_server.py" --port "${GUI_PORT}" \
    > /tmp/call_moderator_gui.log 2>&1 &
sleep 2
if curl -s "http://localhost:${GUI_PORT}/" > /dev/null; then
    echo "    GUI is up. Open it in your browser at:"
    echo "        <your Jupyter base URL>/proxy/${GUI_PORT}/"
    echo "    (e.g. https://<lab-host>/user/<you>/proxy/${GUI_PORT}/ — or use the"
    echo "     IFrame cell in the notebook, which computes the URL for you)"
else
    echo "    WARNING: GUI failed to start — see /tmp/call_moderator_gui.log"
fi

echo "==> [4/5] Clearing stale vLLM processes + fitting GPU memory budget"
STALE_PIDS=$(pgrep -f "vllm serve" || true)
if [ -n "${STALE_PIDS}" ]; then
    echo "    killing stale vLLM PID(s): ${STALE_PIDS}"
    kill -9 ${STALE_PIDS} 2>/dev/null || true
    sleep 8
fi
GPU_MEMORY_UTILIZATION=$(REQUESTED="${GPU_MEMORY_UTILIZATION}" python3 - <<'PY'
import os, sys, torch
free_bytes, total_bytes = torch.cuda.mem_get_info()
requested = float(os.environ["REQUESTED"])
# PERF: reserve ~7 GiB of the free VRAM for the GPU Whisper STT (loaded by the notebook)
# plus headroom — without this, vLLM grabs ~90% of free memory and Whisper OOMs to CPU.
reserve_bytes = 7 * 2**30
usable_bytes = max(free_bytes - reserve_bytes, free_bytes * 0.4)
fitted = max(0.05, min(requested, round(usable_bytes / total_bytes, 2)))
print(f"{fitted:.2f}")
print(f"    free VRAM {free_bytes/2**30:.1f}/{total_bytes/2**30:.1f} GiB -> "
      f"gpu-memory-utilization {fitted:.2f} (requested {requested:.2f})", file=sys.stderr)
PY
)

echo "==> [5/5] Launching vLLM — leave this terminal open"
echo "    Wait for 'Application startup complete', then run the notebook in Tab 2."
echo
exec env VLLM_USE_TRITON_FLASH_ATTN=0 vllm serve "${MODEL_ID}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --api-key "${API_KEY}" \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
