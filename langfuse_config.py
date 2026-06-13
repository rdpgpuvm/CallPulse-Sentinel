"""
langfuse_config.py — Langfuse v3 SDK integration for AI Call Moderator
────────────────────────────────────────────────────────────────────────
Modes (auto-detected):
  A — Langfuse cloud (LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY in .env)
  B — Self-hosted    (same keys + LANGFUSE_BASE_URL=http://localhost:3000)
  C — Local JSONL    (no keys — writes to langfuse_traces.jsonl, zero deps)

.env format (no quotes around values):
  LANGFUSE_PUBLIC_KEY=pk-lf-...
  LANGFUSE_SECRET_KEY=sk-lf-...
  LANGFUSE_BASE_URL=https://us.cloud.langfuse.com

Pipeline is completely UNCHANGED if this file is not imported.
"""

import os, json, time, pathlib
from datetime import datetime, timezone

# ── Load .env ────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass  # python-dotenv optional

# ── Try Langfuse v3 SDK ───────────────────────────────────────────────────────
_lf        = None
_mode      = "local"
_local_log = None
_session_id = f"sess-{int(time.time())}"

_pub = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
_sec = os.environ.get("LANGFUSE_SECRET_KEY", "")
_url = os.environ.get("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")

if _pub.startswith("pk-lf-") and _sec.startswith("sk-lf-"):
    try:
        from langfuse import get_client
        _lf   = get_client()   # reads LANGFUSE_PUBLIC_KEY / SECRET_KEY / BASE_URL automatically
        _mode = "cloud" if "localhost" not in _url else "self-hosted"
        print(f"[langfuse] {_mode} mode active — {_url}  session={_session_id}")
    except Exception as e:
        _lf = None
        print(f"[langfuse] Init failed ({e}) — using local file mode")
else:
    print("[langfuse] No API keys found — using local file mode (Mode C)")

if _lf is None:
    _local_log = open("langfuse_traces.jsonl", "a")
    _mode = "local"


# ── Local file helpers (Mode C) ───────────────────────────────────────────────
def _log_local(stage, system_prompt, user_prompt, result, elapsed_ms, usage):
    if _local_log is None:
        return
    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "session": _session_id,
        "stage": stage,
        "input_chars": len(system_prompt) + len(user_prompt),
        "elapsed_ms": round(elapsed_ms),
        "usage": usage,
        "output_keys": list(result.keys()) if isinstance(result, dict) else None,
    }
    _local_log.write(json.dumps(row) + "\n")
    _local_log.flush()


def show_local_traces(n=20):
    """Print last N traces from local JSONL file."""
    p = pathlib.Path("langfuse_traces.jsonl")
    if not p.exists():
        print("No local trace file yet.")
        return
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    rows = rows[-n:]
    if not rows:
        print("No traces recorded yet.")
        return
    print(f"\n{'ts':<26} {'stage':<22} {'ms':>6}  {'in_ch':>6}  {'usage'}")
    print("-" * 80)
    for r in rows:
        u = r.get("usage") or {}
        print(f"{r['ts'][:25]:<26} {r['stage']:<22} {r['elapsed_ms']:>6}  "
              f"{r['input_chars']:>6}  {u}")


# ── Core wrapper ──────────────────────────────────────────────────────────────
def patch_generate_json(generate_json_fn, stage_token_usage: dict, served_model_name: str):
    """
    Wraps generate_json() to send one Langfuse generation per LLM call.
    Works with both async and sync callers.
    """
    import asyncio, functools

    @functools.wraps(generate_json_fn)
    async def _wrapped(stage: str, system_prompt: str, user_prompt: str,
                       json_schema: dict, max_tokens: int = 64):
        t0 = time.perf_counter()
        result = await generate_json_fn(stage, system_prompt, user_prompt,
                                        json_schema, max_tokens)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        usage = stage_token_usage.get(stage, {})

        if _lf is not None:
            # ── Langfuse v3: one generation observation per LLM call ──────────
            try:
                with _lf.start_as_current_observation(
                    as_type="generation",
                    name=f"judge-{stage}",
                    model=served_model_name,
                    input=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                ) as gen:
                    gen.update(
                        output=result,
                        usage={
                            "input":  usage.get("prompt_tokens", 0),
                            "output": usage.get("completion_tokens", 0),
                            "total":  usage.get("total_tokens", 0),
                        },
                        metadata={
                            "stage":      stage,
                            "session_id": _session_id,
                            "elapsed_ms": round(elapsed_ms),
                        },
                    )
            except Exception as e:
                print(f"[langfuse] trace error (non-fatal): {e}")
        else:
            _log_local(stage, system_prompt, user_prompt, result, elapsed_ms, usage)

        return result

    base_url_display = _url if _lf is not None else getattr(_local_log, 'name', 'langfuse_traces.jsonl')
    print(f"[langfuse] generate_json patched — mode: {_mode}  "
          + (f"dashboard → {_url}" if _lf is not None else f"traces → {base_url_display}"))
    return _wrapped


def flush():
    """Call after pipeline run to push any buffered traces to Langfuse."""
    if _lf is not None:
        _lf.flush()
        print("[langfuse] flushed.")
    elif _local_log is not None:
        _local_log.flush()
        print("[langfuse] local file flushed.")
