#!/usr/bin/env bash
# ============================================================================
# One-shot runner for the AI Call Moderator v3 server side.
#
# Usage (AMD dev portal / ROCm Jupyter):
#   Tab 1: open a Terminal  ->  bash run_vllm_server.sh   (leave it running)
#   Tab 2: open call_moderator_v3.ipynb and run the cells
#
# Does three things:
#   1. Installs every notebook + server python requirement
#      (torch is deliberately NOT touched — the lab ships a ROCm build)
#   2. Repairs the starlette/fastapi conflict that `pip install mcp` causes
#      (mcp drags starlette to >=1.x; vLLM's fastapi needs <0.49)
#   3. Launches vLLM as an OpenAI-compatible server (Ctrl-C stops it)
#
# Override any default via env, e.g.:  PORT=8001 bash run_vllm_server.sh
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
    openai mcp httpx transformers librosa soundfile pandas
# MUST run after mcp: pin starlette back to what vLLM's fastapi supports
python3 -m pip install -q --root-user-action=ignore "starlette<0.49"

echo "==> [2/3] Sanity-checking vLLM imports (catches the Router on_startup crash early)"
python3 - <<'PY'
import fastapi, starlette, vllm
print(f"    vllm {vllm.__version__} | fastapi {fastapi.__version__} | "
      f"starlette {starlette.__version__} -> OK")
PY

echo "==> [3/3] Launching vLLM — leave this terminal open"
echo "    First run downloads the model (~8 GB). Wait for 'Application startup complete',"
echo "    then run the notebook in your other tab."
echo "    Health check: curl http://localhost:${PORT}/v1/models -H \"Authorization: Bearer ${API_KEY}\""
echo
exec env VLLM_USE_TRITON_FLASH_ATTN=0 vllm serve "${MODEL_ID}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --api-key "${API_KEY}" \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
