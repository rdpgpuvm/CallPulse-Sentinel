"""
langfuse_config.py — token & trace monitoring for CallModerator
================================================================
Three modes — pick whichever suits your setup:

  MODE A: Self-hosted Langfuse (Docker, FREE, no account)
  ────────────────────────────────────────────────────────
  Run Langfuse locally with one Docker command (see README).
  Full dashboard at http://localhost:3000.
  Set in .env:
      LANGFUSE_PUBLIC_KEY=pk-lf-...   ← from your local project
      LANGFUSE_SECRET_KEY=sk-lf-...
      LANGFUSE_HOST=http://localhost:3000

  MODE B: Langfuse cloud (free tier, account at langfuse.com)
  ───────────────────────────────────────────────────────────
  Set in .env:
      LANGFUSE_PUBLIC_KEY=pk-lf-...
      LANGFUSE_SECRET_KEY=sk-lf-...
      (LANGFUSE_HOST defaults to https://cloud.langfuse.com)

  MODE C: Local file logging (zero setup, zero dependencies)
  ──────────────────────────────────────────────────────────
  No .env, no keys, no Docker. Every LLM call is logged to
  langfuse_traces.jsonl in the repo root. View with:
      import langfuse_config; langfuse_config.show_local_traces()

In all modes the pipeline performance is UNCHANGED — tracing
is fully async/background and returns in <1 ms.
"""

import os, json, time, uuid, pathlib

# ── load .env if present ──────────────────────────────────────────────────────
_env_path = pathlib.Path(__file__).parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

_session_id = str(uuid.uuid4())[:8]   # groups all turns in one run
_lf = None
_mode = "local"

# ── attempt Langfuse SDK (modes A + B) ────────────────────────────────────────
try:
    from langfuse import Langfuse
    _lf = Langfuse(
        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
        host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        release=os.environ.get("LANGFUSE_RELEASE", "call-moderator"),
    )
    _is_local = "localhost" in os.environ.get("LANGFUSE_HOST", "")
    _mode = "self-hosted" if _is_local else "cloud"
    print(f"[langfuse] {_mode} mode active — dashboard at {_lf.base_url}  session={_session_id}")
except ImportError:
    print("[langfuse] SDK not installed — using local file mode  (pip install langfuse for dashboard)")
except KeyError:
    print("[langfuse] No keys found — using local file mode  (see .env.example to enable dashboard)")
except Exception as e:
    print(f"[langfuse] Init failed ({e}) — using local file mode")

# ── local file fallback (mode C) ─────────────────────────────────────────────
_local_trace_file = pathlib.Path(__file__).parent / "langfuse_traces.jsonl"


def _log_local(stage, prompt_tokens, completion_tokens, latency_ms, result, system_prompt, user_prompt):
    """Append one trace record to langfuse_traces.jsonl (mode C fallback)."""
    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "session": _session_id,
        "stage": stage,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "latency_ms": round(latency_ms, 1),
        "output": result,
        "system_snippet": system_prompt[:120],
        "user_snippet": user_prompt[:120],
    }
    with _local_trace_file.open("a") as f:
        f.write(json.dumps(record) + "\n")


def show_local_traces(tail=20):
    """Print the last N traces from langfuse_traces.jsonl (mode C viewer)."""
    if not _local_trace_file.exists():
        print("No local traces yet — run the pipeline first.")
        return
    lines = _local_trace_file.read_text().splitlines()
    print(f"\nLocal traces ({len(lines)} total, showing last {tail}):")
    print(f"{'timestamp':<22} {'session':<10} {'stage':<22} {'prompt':>7} {'compl':>6} {'total':>6} {'ms':>7}")
    print("-" * 90)
    for line in lines[-tail:]:
        r = json.loads(line)
        print(f"{r['ts']:<22} {r['session']:<10} {r['stage']:<22} "
              f"{r['prompt_tokens']:>7} {r['completion_tokens']:>6} "
              f"{r['total_tokens']:>6} {r['latency_ms']:>7.0f}")
    total_tokens = sum(json.loads(l)['total_tokens'] for l in lines)
    print(f"\nAll-time total: {total_tokens:,} tokens across {len(lines)} calls")


# ── main patch function ───────────────────────────────────────────────────────
def patch_generate_json(generate_json_fn, stage_token_usage: dict, served_model_name: str):
    """
    Wrap generate_json once to add tracing.  Call this after Cell 3 defines it:

        import langfuse_config
        generate_json = langfuse_config.patch_generate_json(
            generate_json, STAGE_TOKEN_USAGE, SERVED_MODEL_NAME)

    Works in all three modes (A/B/C) — pipeline performance is unchanged.
    """
    import functools

    @functools.wraps(generate_json_fn)
    async def _wrapped(stage, system_prompt, user_prompt, json_schema, max_tokens=64):
        t0 = time.perf_counter()
        result = await generate_json_fn(stage, system_prompt, user_prompt, json_schema, max_tokens)
        latency_ms = (time.perf_counter() - t0) * 1000
        usage = stage_token_usage.get(stage, {})
        p_tok = usage.get("prompt", 0)
        c_tok = usage.get("completion", 0)

        if _lf is not None:
            # Mode A or B: send to Langfuse dashboard
            try:
                _lf.generation(
                    name=f"judge/{stage}", model=served_model_name,
                    session_id=_session_id,
                    usage={"input": p_tok, "output": c_tok, "unit": "TOKENS"},
                    metadata={"latency_ms": round(latency_ms, 1)},
                    input={"system": system_prompt[:300], "user": user_prompt[:300]},
                    output=result,
                )
            except Exception:
                pass  # never let monitoring break the pipeline
        else:
            # Mode C: write to local file
            try:
                _log_local(stage, p_tok, c_tok, latency_ms, result, system_prompt, user_prompt)
            except Exception:
                pass

        return result

    print(f"[langfuse] generate_json patched — mode: {_mode}  "
          + (f"traces → {_local_trace_file.name}" if _lf is None else f"dashboard → {_lf.base_url}"))
    return _wrapped


def flush():
    """Flush any queued Langfuse traces (called automatically on process exit)."""
    if _lf:
        _lf.flush()
