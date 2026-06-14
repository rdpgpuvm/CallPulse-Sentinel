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

import os, json, time, pathlib, contextvars
from datetime import datetime, timezone

# ── Load .env ────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass

# ── Context var so brain_worker can tag each generation with its call_id ─────
_current_call_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    'langfuse_call_id', default='unknown')

def set_current_call_id(call_id: str):
    """Call this in brain_worker before generate_json so traces are tagged by call."""
    _current_call_id.set(call_id)

# ── In-memory trace cache: call_id -> list of generation records ─────────────
# This is the source for the GUI info panel regardless of cloud/local mode.
_call_traces: dict = {}

def get_call_traces(call_id: str) -> list:
    """Return all generation records for a call_id (used by GUI /langfuse/<call_id>)."""
    return _call_traces.get(call_id, [])

def get_all_call_ids() -> list:
    return list(_call_traces.keys())

def clear_traces():
    """Call between pipeline runs to reset the cache."""
    _call_traces.clear()

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
        _lf   = get_client()
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
def _log_local(record: dict):
    if _local_log is None:
        return
    _local_log.write(json.dumps(record) + "\n")
    _local_log.flush()


def show_local_traces(n=20):
    p = pathlib.Path("langfuse_traces.jsonl")
    if not p.exists():
        print("No local trace file yet.")
        return
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    rows = rows[-n:]
    if not rows:
        print("No traces recorded yet.")
        return
    print(f"\n{'ts':<26} {'call_id':<36} {'stage':<22} {'ms':>6}  {'in':>5}  {'out':>5}")
    print("-" * 100)
    for r in rows:
        u = r.get("usage") or {}
        print(f"{r['ts'][:25]:<26} {r.get('call_id','?'):<36} {r['stage']:<22} "
              f"{r['elapsed_ms']:>6}  {u.get('input',0):>5}  {u.get('output',0):>5}")


# ── Core wrapper ──────────────────────────────────────────────────────────────
def patch_generate_json(generate_json_fn, stage_token_usage: dict, served_model_name: str):
    """
    Wraps generate_json() to:
      1. Send one Langfuse generation per LLM call (cloud or local file)
      2. Cache the record in _call_traces[call_id] for the GUI info panel
    """
    import asyncio, functools

    @functools.wraps(generate_json_fn)
    async def _wrapped(stage: str, system_prompt: str, user_prompt: str,
                       json_schema: dict, max_tokens: int = 64):
        call_id = _current_call_id.get()
        usage = stage_token_usage.get(stage, {})

        if _lf is not None:
            # wrap the LLM call INSIDE the context manager so Langfuse measures
            # the real inference latency, not just the time to call gen.update()
            try:
                # A parent span carries session_id + user_id so all generations from
                # this run appear grouped under one session in the Langfuse Sessions tab.
                # Without this wrapper, session_id would only live in metadata and the
                # Sessions view would stay empty.
                with _lf.start_as_current_span(
                    name=f"call-{call_id}",
                    session_id=_session_id,   # groups every trace from this run in Sessions tab
                    user_id=call_id,          # lets you filter by call in the Traces view
                ):
                    with _lf.start_as_current_observation(
                        as_type="generation",
                        name=f"judge-{stage}",
                        model=served_model_name,
                        input=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user",   "content": user_prompt},
                        ],
                    ) as gen:
                        t0 = time.perf_counter()
                        result = await generate_json_fn(stage, system_prompt, user_prompt,
                                                        json_schema, max_tokens)
                        elapsed_ms = (time.perf_counter() - t0) * 1000
                        gen.update(
                            output=result,
                            usage={
                                "input":  usage.get("prompt_tokens", 0),
                                "output": usage.get("completion_tokens", 0),
                                "total":  usage.get("total_tokens", 0),
                            },
                            metadata={
                                "stage":      stage,
                                "call_id":    call_id,
                                "elapsed_ms": round(elapsed_ms),
                            },
                        )
            except Exception as e:
                print(f"[langfuse] trace error (non-fatal): {e}")
                t0 = time.perf_counter()
                result = await generate_json_fn(stage, system_prompt, user_prompt,
                                                json_schema, max_tokens)
                elapsed_ms = (time.perf_counter() - t0) * 1000
        else:
            t0 = time.perf_counter()
            result = await generate_json_fn(stage, system_prompt, user_prompt,
                                            json_schema, max_tokens)
            elapsed_ms = (time.perf_counter() - t0) * 1000

        record = {
            "ts":         datetime.now(timezone.utc).isoformat(),
            "call_id":    call_id,
            "session_id": _session_id,
            "stage":      stage,
            "model":      served_model_name,
            "elapsed_ms": round(elapsed_ms),
            "usage": {
                "input":  usage.get("prompt_tokens", 0),
                "output": usage.get("completion_tokens", 0),
                "total":  usage.get("total_tokens", 0),
            },
            "output": result,
        }
        _call_traces.setdefault(call_id, []).append(record)
        if _lf is None:
            _log_local(record)

        return result

    print(f"[langfuse] generate_json patched — mode: {_mode}  "
          + (f"dashboard -> {_url}" if _lf is not None
             else f"traces -> {getattr(_local_log, 'name', 'langfuse_traces.jsonl')}"))
    return _wrapped


def flush():
    if _lf is not None:
        _lf.flush()
        print("[langfuse] flushed.")
    elif _local_log is not None:
        _local_log.flush()
        print("[langfuse] local file flushed.")
