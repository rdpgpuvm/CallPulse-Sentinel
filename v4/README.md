# AI Call Moderator v4 — Real-Time 3-Step Assembly Line

v4 turns the moderator into the architecture document's **3-step assembly line**: audio is ingested in 5-second chunks, transcribed locally, handed across zero-copy `asyncio.Queue`s to a vLLM-served judge, and flagged **the instant** a rule trips — with the *transcript-ready → flag-raised* latency measured in milliseconds for every flag. Test input is 1–3 real call-center recordings pulled from Kaggle via MCP OAuth.

## How to run (the 3-step flow)

**Tab 1 — Terminal** (one command, leave it running):

```bash
git clone -b v4-realtime https://github.com/rdpgpuvm/Project1.git /workspace/CallModV4
cd /workspace/CallModV4/v4 && bash run_vllm_server.sh
```

The runner installs all pip requirements (never touching the lab's ROCm torch), repairs the known `mcp`→starlette→fastapi conflict, sanity-checks the vLLM import, and serves `Qwen3-4B-Instruct-2507` as `call-moderator-llm`. Wait for `Application startup complete` (first run downloads ~8 GB). If it reports low free GPU memory, an old process holds the card: `rocm-smi`, kill the stale `vllm` PID, rerun.

**Tab 2 — Notebook:** open `v4/call_moderator_v4_realtime.ipynb` and run the cells **one by one**. Every cell has a markdown explainer above it and per-line comments inside it. The only interactive moment is the Kaggle OAuth consent (CELL 4a): open the printed URL in your browser, approve, paste the localhost redirect URL back.

## Architecture document → v4 mapping

| Document | v4 | Why |
|---|---|---|
| Faster-Whisper, 5 s chunks ("The Ears") | ✅ kept — `small`, int8, **CPU** | CTranslate2 has no ROCm backend; CPU int8 transcribes a 5 s chunk faster than real time, and the GPU stays 100 % dedicated to vLLM |
| zero-copy in-memory `asyncio.Queue` | ✅ kept — one FIFO **per call** | order is preserved within a call (escalation rule 3 reads *consecutive* sentiment) while calls run in parallel |
| Qwen 2.5 32B Q6 ("The Brain") | Qwen3-4B-Instruct via vLLM | 8× fewer weights → lower latency and tokens; per-turn compliance judging doesn't need 32 B |
| PyQt6 red-flash + pyqtSignal + OS alarm ("The Alarm") | flushed alert print + sqlite `:memory:` audit DB | Jupyter has no desktop GUI; the handshake and ms-precise latency measurement are identical |
| WebRTC/WebSocket ingress | chunked file streaming (optional 1× pacing) | same dataflow — swap the producer's source for a socket later; nothing downstream changes |
| LangChain / LangGraph (considered) | **not used** | a fixed 3-stage line needs no graph framework; it would add layers (and ms) between transcript and flag |

## The pipeline, snippet by snippet

**Step 1 — The Ears.** One producer per call walks the recording in 5 s chunks (stereo channels transcribed separately = free speaker separation), skips silence for free, transcribes off the event loop, and stamps each transcript with the moment the flag clock starts:

```python
text = await asyncio.to_thread(transcribe_chunk, piece)   # blocking ASR off the event loop
await transcript_queue.put({
    "call_id": call_id, "speaker": f"speaker_{channel_index}",
    "text": text, "transcribed_at": time.perf_counter(),  # flag-latency clock starts here
})
```

**Handshake 1→2.** The `asyncio.Queue` lives in the same process — putting a dict on it copies a pointer, not data: a ~0 ms hop, exactly the document's "zero-copy" requirement.

**Step 2 — The Brain.** One worker per call `await queue.get()`s, guesses the speaker's role with a zero-token marker scoreboard, pre-screens with regex (0 tokens), then makes a single schema-constrained judge call with only the last 3 utterances as context:

```python
analysis = await generate_json("turn_moderation", MODERATOR_SYSTEM_PROMPT,
                               f'Context:\n{context}\n\nLATEST {role.upper()} utterance: "{item["text"]}"',
                               TURN_ANALYSIS_SCHEMA, max_tokens=64)
```

Escalation is deterministic code, not the model: critical violation → flag; two high-severity violations → flag; customer sentiment ≤ −2 twice in a row → flag. First trigger latches and pushes to the alert queue (handshake 2→3).

**Step 3 — The Alarm.** The consumer prints an unmissable flushed alert *the instant* the flag arrives and records it in the in-memory DB:

```python
flag_latency_ms = (alert["flagged_at"] - alert["transcribed_at"]) * 1000
print(f"!! ESCALATE {alert['call_id']} turn {alert['turn_number']} ({alert['rule']})", flush=True)
```

## Speed, tokens, and the design choices behind them

The bottleneck between a spoken violation and a raised flag is exactly **one judge call** on a 4B model — typically a few hundred ms — because everything around it is free: silence is dropped before ASR, role guessing and pre-screening are regex/string scoring, the queue hops are in-process, and the alarm is a flushed print. Guided JSON (vLLM grammar-constrained decoding) makes malformed output impossible, so there are never retry tokens; `temperature=0` keeps verdicts reproducible. The prompt diet (policy codes + last 3 utterances, `max_tokens=64`) holds per-turn cost near ~300 prompt / ~40 completion tokens. A shared `asyncio.Semaphore(16)` is the client-side concurrency gate — vLLM's continuous batching does the real scheduling on the GPU.

CELL 9 reports it all from in-memory data: every raised flag with its rule and ms latency, per-stage latency (avg / p95 / max), exact per-stage and combined token usage from vLLM's `usage` field, and a per-call SQL summary from the `:memory:` audit DB.

## Going truly live

Replace `ears_producer`'s file loop with a WebSocket/WebRTC receiver that pushes 5 s chunks — every line downstream of `transcript_queue.put(...)` is already streaming-shaped. Set `REALTIME_PACING = True` in CELL 5 to rehearse at 1× speed today. The audit DB swaps from `:memory:` to a file (or Redis) with one line when persistence is needed.
