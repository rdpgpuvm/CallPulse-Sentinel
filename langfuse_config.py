"""
langfuse_config.py — optional real-time token & trace monitoring for CallModerator
====================================================================================
HOW TO ACTIVATE
  1. pip install langfuse                   (done automatically by run_vllm_server.sh)
  2. Copy .env.example -> .env and fill in your Langfuse keys
  3. The notebook's optional cell already imports this module — no other changes needed

HOW IT WORKS (zero pipeline impact)
  generate_json() is wrapped once at import time.
  lf.generation() enqueues trace data in a background thread and returns in <1 ms —
  the real-time judge path is never blocked, even if Langfuse cloud is unreachable.
  Flush is automatic on process exit, or call lf.flush() manually.

WHAT YOU SEE IN LANGFUSE DASHBOARD
  - Every LLM judge call grouped by stage (turn_analysis, etc.)
  - Prompt + completion token counts per call → cost estimate
  - Latency histogram (wall-clock ms from generate_json entry to return)
  - The actual JSON verdict returned for each turn (for quality review)
  - Session grouping: all turns from one pipeline run share a trace
"""

import os
import pathlib
import time
import uuid

# ── load .env if present (never required — env vars work too) ─────────────────
_env_path = pathlib.Path(__file__).parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())

# ── attempt Langfuse init ──────────────────────────────────────────────────────
_lf = None
_session_id = str(uuid.uuid4())[:8]     # ties all turns in one run to one Langfuse session

try:
    from langfuse import Langfuse
    _lf = Langfuse(
        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
        host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        release=os.environ.get("LANGFUSE_RELEASE", "call-moderator"),
    )
    print(f"[langfuse] monitoring active — project traces at {_lf.base_url}"
          f"  session={_session_id}")
except ImportError:
    print("[langfuse] SDK not installed — run: pip install langfuse")
except KeyError as missing:
    print(f"[langfuse] key {missing} not set — check .env or environment variables (no-op)")
except Exception as e:
    print(f"[langfuse] init failed ({e}) — monitoring disabled (no-op)")


def patch_generate_json(generate_json_fn, stage_token_usage: dict, served_model_name: str):
    """
    Call this once after define generate_json to wrap it with Langfuse tracing.
    Returns the wrapped coroutine (or the original if Langfuse is inactive).

    Usage in notebook optional cell:
        import langfuse_config
        generate_json = langfuse_config.patch_generate_json(
            generate_json, STAGE_TOKEN_USAGE, SERVED_MODEL_NAME)
    """
    if _lf is None:
        return generate_json_fn          # no-op: return original unchanged

    import functools

    @functools.wraps(generate_json_fn)
    async def _wrapped(stage, system_prompt, user_prompt, json_schema, max_tokens=64):
        t0 = time.perf_counter()
        result = await generate_json_fn(stage, system_prompt, user_prompt,
                                        json_schema, max_tokens)
        latency_ms = (time.perf_counter() - t0) * 1000
        usage = stage_token_usage.get(stage, {})
        try:
            _lf.generation(
                name=f"judge/{stage}",
                model=served_model_name,
                session_id=_session_id,
                usage={
                    "input":  usage.get("prompt", 0),
                    "output": usage.get("completion", 0),
                    "unit":   "TOKENS",
                },
                metadata={"latency_ms": round(latency_ms, 1)},
                input={"system": system_prompt[:300], "user": user_prompt[:300]},
                output=result,
            )
        except Exception:
            pass    # never let monitoring break the pipeline
        return result

    return _wrapped


def flush():
    """Manually flush all queued traces (called automatically on process exit)."""
    if _lf:
        _lf.flush()
