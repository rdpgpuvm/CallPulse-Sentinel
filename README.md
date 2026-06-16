# CallPulse Sentinel — Real-Time Compliance Pipeline

A two-model pipeline that monitors live call center calls for policy violations.
**Whisper** (ASR) transcribes audio; **Qwen3-4B-Instruct** (LLM judge via vLLM on AMD GPU)
flags violations in under one second. A live FastAPI/WebSocket dashboard gives supervisors
override controls, audio seeking, and escalation review.

---

## Quick Start

```bash
# Clone the repo
git clone https://github.com/rdpgpuvm/CallPulse-Sentinel.git
cd CallPulse-Sentinel

# (optional) enable Langfuse monitoring — copy the template and add your keys
cp .env.example .env

# Start vLLM + the live GUI dashboard (leave this terminal running)
bash run_vllm_server.sh

# Then run call_moderator_gui.ipynb cell by cell in Tab 2
```

---

## ASR modes (runtime toggle — single branch, no branches to pick)

One ASR path, two runtime modes, chosen by `REALTIME_SINGLE_CALL` in Cell 8:

| Mode | ASR | When to use |
|---|---|---|
| Real-time (`REALTIME_SINGLE_CALL = True`) | whisper-large-v3-turbo **fp16 on GPU** (~150–400 ms/chunk), 5s live pacing | Live monitoring; auto-falls back to CPU int8 if no GPU |
| Bulk / file (`REALTIME_SINGLE_CALL = False`) | faster-whisper large-v3-turbo **CPU int8**, full-file with confidence gates | Offline, max-accuracy batch |

The LLM judge (Qwen3-4B via vLLM) always runs on the AMD GPU.

---

## Supervisor Override UI

When a call escalates, the **⚡ OVERRIDE** button appears:
1. Click it → escalation list drops down with audio timestamps
2. Click any item → audio seeks to that exact moment
3. Expand `+ show` → see the exact transcript text that triggered the flag
4. Click **✅ JOIN CALL** → supervisor indicator activates globally

In **Simple mode** the escalation panel auto-opens on escalation — no click required.

**Audio sync:** enable the **Sync audio** checkbox so the audio seeker follows each incoming turn automatically.

---

## Skipped-Segment Panel

Every chunk the ASR pipeline silently dropped appears in a collapsible panel:

| Tag | Meaning |
|---|---|
| `silence` | RMS below threshold — almost certainly fine |
| `beep` | Tone-dominated spectrum — censor bleep / redacted PII |
| `no_speech` | Whisper P(no_speech) > 0.5 |
| `low_confidence` | avg_logprob < −1.0 (uncertain decode) |
| `repetition` | compression_ratio > 2.4 (hallucination loop) |

Click any item to seek audio and validate by ear.

---

## Monitoring with Langfuse (optional — three modes, no account required)

Langfuse tracks token usage, latency, and LLM verdicts per call.
**You do not need a Langfuse account.** Choose any of the three modes below.

### Mode A — Self-hosted Docker (recommended, no account, full dashboard)

Run Langfuse on your own machine. Completely free and private.

```bash
# One-time setup — starts Langfuse at http://localhost:3000
docker run -d --name langfuse \
  -p 3000:3000 \
  -e NEXTAUTH_SECRET=change-me-secret \
  -e SALT=change-me-salt \
  -e DATABASE_URL=file:/data/langfuse.db \
  -v langfuse_data:/data \
  langfuse/langfuse:latest
```

Then open **http://localhost:3000** → Create project → Settings → API Keys.
Copy the public + secret key into `.env`:

```bash
cp .env.example .env
# Edit .env and fill in your keys + set:
# LANGFUSE_BASE_URL="http://localhost:3000"
```

### Mode B — Langfuse cloud (free tier, account at langfuse.com)

Create `.env` using `printf` (works in Jupyter terminals — do **not** use `cat <<EOF` as it breaks when pasted on one line):

```bash
printf 'LANGFUSE_PUBLIC_KEY=pk-lf-REPLACE_ME\nLANGFUSE_SECRET_KEY=sk-lf-REPLACE_ME\nLANGFUSE_BASE_URL=https://us.cloud.langfuse.com\n' > .env
```

Verify it wrote correctly:

```bash
cat .env
```

Then fill in your real keys from **langfuse.com → Project Settings → API Keys**.

### Mode C — Local file logging (zero setup, zero dependencies, always works)

**No Docker, no account, no `.env` file needed.** Just run the optional Cell 7 in the notebook
and every LLM call is logged to `langfuse_traces.jsonl` in the repo root.

To view traces in the notebook:
```python
import langfuse_config
langfuse_config.show_local_traces()   # prints last 20 calls with token counts
```

### Activating monitoring (all modes)

Run the optional **Cell 7** in the notebook after Cell 3:

```python
import langfuse_config
generate_json = langfuse_config.patch_generate_json(
    generate_json, STAGE_TOKEN_USAGE, SERVED_MODEL_NAME)
```

The pipeline is **unchanged** whether you run Cell 7 or not.
`langfuse_config.py` detects which mode to use automatically:
- Keys found in `.env` → Mode A or B (dashboard)
- No keys → Mode C (local file)

### Sessions (Modes A and B)

Every pipeline run is automatically assigned a unique `session_id` (e.g. `sess-1781397200`), generated at import time in `langfuse_config.py`. All LLM judge calls in that run are attached to this session as a **first-class Langfuse field** — not just metadata — via a parent span that carries `session_id` and `user_id=call_id`.

**To view a session in the cloud UI:**
1. Open `https://us.cloud.langfuse.com` (or your self-hosted URL)
2. Click **Sessions** in the left sidebar
3. Select the session ID printed at startup (e.g. `sess-1781397200`)

The Sessions view shows aggregate token usage, total cost, and a latency timeline across every call processed in that run. Individual calls are also filterable in the Traces view by `user_id=call_id`.


---

## Recordings included

| Folder | Files | Notes |
|---|---|---|
| `scam_call/` | `call_recording1.mp3`, `call_recording2.mp3` | Two demo/test calls (stereo scam-awareness scenarios) |

Leave `SELECTED_CALL_IDS = ""` in Cell 8 to pick one at random, or set it to a
stem (e.g. `"call_recording1"`) to target a specific recording. Drop more audio
into `scam_call/` and it is picked up automatically.

---

## Architecture

```
Audio file / live stream
        │
        ▼
  ┌─────────────┐    asyncio.Queue    ┌──────────────────┐   asyncio.Queue   ┌──────────┐
  │  THE EARS   │ ─────────────────▶  │   THE BRAIN      │ ────────────────▶ │THE ALARM │
  │  ASR model  │  (zero-copy µs)     │  Qwen3-4B-Instr  │  (zero-copy µs)  │ + GUI    │
  │  CPU/GPU    │                     │  vLLM AMD ROCm   │                   │ sqlite   │
  └─────────────┘                     │  guided JSON     │                   └──────────┘
                                      │  Semaphore(16)   │
                                      └──────────────────┘
```

- **asyncio.Queue** — zero-copy ~µs handshakes between stages
- **guided JSON** — grammar-constrains LLM output at decode time, eliminates JSON retries
- **Semaphore(16)** — caps concurrent LLM requests, keeps GPU saturated without socket thrash
- **temperature=0** — greedy decode, reproducible verdicts for compliance audit

---

## Token usage (what the numbers mean)

Each LLM judge call costs roughly:
- **~300 prompt tokens** — system prompt (policy definitions) + 3-turn context
- **~25 completion tokens** — the JSON verdict: `{"sentiment":-1,"violations":["C2"],"reason":"..."}`
- **~325 total per turn**

See Cell 9 (results) for exact counts after each run.


## Architecture & Pipeline Flow

```
AUDIO FILE (WAV/MP3/FLAC)
        |
        v
  ┌─────────────────────────────────────────────────────────┐
  │  CELL 4 — Audio Discovery                               │
  │  AUDIO_DIRS = [call_recordings/, scam_call/]            │
  │  collect_audio() scans all dirs, builds stem->path map  │
  └─────────────────┬───────────────────────────────────────┘
                    |
        ┌───────────┴───────────┐
        │ REALTIME_SINGLE_CALL  │  SELECTED_CALL_IDS
        │  True  = 1 call, 1x  │  ""  = random / all
        │  False = bulk, flat  │  "x" = that recording
        └───────────┬───────────┘
                    |
  ┌─────────────────v───────────────────────────────────────┐
  │  STEP 1 — EARS (ears_producer)                          │
  │                                                         │
  │  FILE MODE (bulk / accurate)                            │
  │    faster-whisper large-v3-turbo, CPU int8              │
  │    beam_size=5, temperature=[0.0,0.2,0.4]               │
  │    Full channel transcription, VAD-segmented            │
  │    quality_gated(): drops no_speech_prob>0.5,           │
  │      avg_logprob<-1.0, compression_ratio>2.4            │
  │                                                         │
  │  LIVE MODE (real-time simulation)                       │
  │    whisper-large-v3-turbo fp16 on GPU (ROCm)            │
  │    5s chunks, deadline pacing, ~150-400ms/chunk         │
  │    RMS gate: skip silence (<1e-4 abs mean)              │
  │    FFT gate: skip beep tones (spectral entropy<0.3)     │
  │    Fallback: CPU int8 if GPU unavailable                │
  │                                                         │
  │  Emits: skip events (silence/beep/low-conf/repetition)  │
  └────────────────────┬────────────────────────────────────┘
                       | asyncio.Queue (zero-copy, µs handoff)
                       | utterance: {call_id, speaker, text,
                       |             audio_start_s, asr_ms}
  ┌────────────────────v────────────────────────────────────┐
  │  STEP 2 — BRAIN (brain_worker, 1 per call)              │
  │                                                         │
  │  Semaphore(16): max concurrent LLM requests             │
  │  3-turn rolling context window (token budget)           │
  │                                                         │
  │  Fast-path (skip judge):                                │
  │    utterance <= 12 chars AND no regex hints             │
  │    ~20% of turns, saves ~350 tokens + ~300ms each       │
  │    role carried from last judge call for that channel   │
  │                                                         │
  │  LLM JUDGE — Qwen3-4B via vLLM                         │
  │  ┌──────────────────────────────────────────────────┐   │
  │  │  Model: Qwen/Qwen3-4B-Instruct-2507              │   │
  │  │  Served as: call-moderator-llm                   │   │
  │  │  Port: 8000, temperature=0 (greedy, reproducible)│   │
  │  │  max_tokens=48, guided JSON (grammar-constrained) │   │
  │  │  GPU memory utilization: auto-fitted              │   │
  │  │  max_model_len: 16384                             │   │
  │  │                                                   │   │
  │  │  Input:  system prompt (~300 tokens, fixed)       │   │
  │  │          + 3-turn context (~100 tokens)           │   │
  │  │          + latest utterance (~30 tokens)          │   │
  │  │  Output: {speaker, sentiment, violations, reason} │   │
  │  │          ~32-48 tokens, single pass, no retries   │   │
  │  │                                                   │   │
  │  │  speaker: "rep"|"customer"  (stereo: calibrate   │   │
  │  │           +lock channel; mono: regex → LLM)       │   │
  │  │  sentiment: -2..2  (customer only)                │   │
  │  │  violations: [code, ...]  (policy codes)          │   │
  │  │  reason: string, max 35 chars                     │   │
  │  └──────────────────────────────────────────────────┘   │
  │                                                         │
  │  Deterministic escalation rules (no LLM, instant):      │
  │    Rule 1: any critical violation                        │
  │    Rule 2: >= 2 high-severity violations                 │
  │    Rule 3: 2 consecutive customer sentiment <= -2        │
  └────────────────────┬────────────────────────────────────┘
                       | asyncio.Queue (zero-copy)
                       | alert: {call_id, rule, detail,
                       |         audio_start_s, latencies}
  ┌────────────────────v────────────────────────────────────┐
  │  STEP 3 — ALARM (alarm_consumer, shared)                │
  │  flush=True print (instant visual signal)               │
  │  sqlite :memory: audit trail (turns + alerts)           │
  │  emit_event() -> GUI dashboard (httpx, 2s timeout)      │
  └────────────────────┬────────────────────────────────────┘
                       |
  ┌────────────────────v────────────────────────────────────┐
  │  GUI DASHBOARD (gui_server.py, port 7860)               │
  │  Starlette + WebSocket, event-driven                    │
  │  Per-call tabs, speaker indicators, turn log            │
  │  Escalation panel: expand items, seek audio             │
  │  Skip panel: silence/PII-beep/low-conf segments         │
  │  Override flow: OVERRIDE -> review -> JOIN CALL          │
  │  Audio sync: seeks player to turn timestamp             │
  │  Info panel: Langfuse token usage per call              │
  │  Simple mode: minimal UI for non-technical supervisors  │
  └─────────────────────────────────────────────────────────┘
```

## LLM Usage — Patterns, Frequency and Settings

### When the LLM is called

The LLM judge (Qwen3-4B via vLLM) is called once per utterance that passes the fast-path gate. The fast-path skips utterances that are 12 characters or fewer with no keyword hits — typically acknowledgements like "Okay.", "Yes.", "Sure." — which account for roughly 20 % of turns in a real call.

For a typical 5-minute call with two speakers and 5-second chunks, expect 30-60 LLM calls per call. In bulk mode all calls run concurrently and vLLM batches their requests internally.

### Token budget per call

| Component            | Tokens (approx) |
|----------------------|-----------------|
| System prompt        | ~300 (fixed, cached by --enable-prefix-caching) |
| 3-turn context       | ~60-150          |
| Latest utterance     | ~10-50           |
| Total input          | ~370-500         |
| Output (verdict)     | ~32-48           |

With prefix caching enabled the 300-token system prompt is computed once and reused for every judge call, cutting prefill cost by ~60-70 %.

### vLLM server settings

| Setting                    | Value          | Reason |
|----------------------------|----------------|--------|
| Model                      | Qwen3-4B-Instruct-2507 | 4B fits in shared VRAM with Whisper |
| temperature                | 0              | Greedy decoding — same transcript always produces same verdict (audit reproducibility) |
| max_tokens                 | 48             | Reason maxLength=35 chars in schema; max plausible verdict ~35 tokens; 48 is lossless ceiling |
| guided_json                | schema         | Grammar-constrains output during decoding — eliminates JSON repair retries |
| gpu_memory_utilization     | auto-fitted    | Script reserves ~7 GiB for Whisper, vLLM gets the rest |
| max_model_len              | 16384          | Upper bound on context; actual prompts are ~500 tokens |
| Semaphore                  | 16             | Client-side concurrency cap matching vLLM scheduler budget |

### Speaker identification — 3-tier pipeline

Speaker role (`rep` / `customer`) is resolved differently depending on whether the audio carries clean channel separation.

**Stereo / multi-channel audio — action-anchored, self-correcting.** Each channel carries exactly one speaker for the whole call, but telephony does **not** guarantee channel 0 is the rep, and *who sounds like a rep* is unreliable — on a scam-awareness call the savvy customer talks like a compliance officer ("no legitimate rep would ask for your password"), which fools content-only labeling. So the Brain scores each channel **every turn and recomputes** which one is the rep (no permanent lock — it self-corrects mid-call). The dominant signal is **behaviour, not tone**: only an agent asks for a full SSN/card/password (`C1`), changes accounts (`R1`/`R3`), or offers off-the-books / over-policy deals (`C3`/`C4`). Those violation codes are detected from content regardless of the current role label, so they anchor the rep channel robustly; the channel that opened the call gets a small prior, and the LLM/regex "tone" lean is only a light tiebreaker for clean calls. This costs no extra model calls — the `speaker` field rides the judge pass that already runs every turn.

**Mono / single-channel audio — accurate LLM identification at zero extra cost.** When there is no channel separation the speaker rides the same judge forward pass that already produces sentiment and violations, so it adds no latency or tokens. Two tiers:

1. **Keyword regex pre-pass** (`_guess_mono_speaker`) — obvious lexical markers matched in microseconds at zero token cost. Customer markers: *"your agent"*, *"loyal member"*, *"you work for me"*, *"the customer is always right"*, *"get me your manager"*. Rep markers: *"how can I help"*, *"let me pull up"*, *"one moment"*, *"you were speaking with"*. Returns `'rep'`, `'customer'`, or `None`.
2. **LLM `speaker` field** — used when regex is ambiguous. The prompt now favours **speaker continuity** rather than strict alternation: because a 5-second chunk frequently splits one person's sentence across turns, the same speaker is assumed to continue unless the content clearly shows the other party is now talking. This fixes single sentences being mislabelled across the rep/customer boundary at chunk edges.

**In a live deployment** the telephony/CTI layer (Genesys, Avaya, Amazon Connect, Twilio, etc.) supplies channel metadata, so only the brief stereo calibration runs (or is skipped entirely if the layer also tags role) — speaker is otherwise a one-line lookup from stream metadata.

### Customer sentiment traffic-light (satisfied / neutral / dissatisfied)

Every judge verdict already includes a `sentiment` score in `-2..2` for the customer. That single number is mapped — at **render time only** — to a three-state traffic light, so the feature is **lossless and adds zero model cost** (no extra LLM call, no added latency):

| Sentiment score | State | Colour |
|---|---|---|
| `>= +1` | satisfied | 🟢 green |
| `0` | neutral | 🟡 yellow |
| `<= -1` | dissatisfied | 🔴 red |

**In the dashboard (`gui_server.py`):** customer chat bubbles get a small square-outlined badge containing the word *"sentiment"*. The outline and the text share one colour (via `currentColor`), set the instant the turn renders according to the judgement — green / yellow / red. The badge appears only under **customer** turns (rep turns carry no customer-sentiment signal). Hovering shows the exact state and score. The mapping lives in `sentBand()` in `gui_server.py` and is mirrored by `_sentiment_band()` in the notebook.

**In the notebook cell output:** the same three-state band is printed as a colour-coded `[ sentiment ]` token (ANSI green / yellow / red) next to each customer turn, so the live run is readable without opening the dashboard.

Because both surfaces read the score the judge already returns, the traffic light never disagrees with the deterministic Rule 3 escalation (two consecutive customer turns at `<= -2`).

### Langfuse observability

Every LLM call is instrumented via `langfuse_config.py`. Token counts, latency, stage, and verdict are recorded per call. The GUI info panel fetches this from an in-memory cache (zero network latency) and displays token usage, per-stage breakdown, and a per-generation table. The same data is forwarded to the Langfuse cloud dashboard asynchronously without affecting pipeline latency.

## Performance & Efficiency Design

The pipeline was built to be resource-conscious without sacrificing accuracy. Every architectural choice has a specific reason.

### CPU/GPU split

In **file mode** (`REALTIME_PACING=False`), Whisper runs on CPU (faster-whisper int8). This frees the AMD GPU entirely for vLLM and gives access to per-segment confidence metadata (`no_speech_prob`, `avg_logprob`, `compression_ratio`) needed by `quality_gated()`. CPU int8 on large-v3-turbo runs in ~1–2s per chunk — fine for offline batch processing.

In **live mode** (`REALTIME_PACING=True`), Whisper switches to GPU fp16 via HF transformers (~150–400 ms per chunk). This is mandatory to keep up with the 5s live deadline and also unlocks `num_beams=4` beam search for better accuracy. GPU VRAM is shared: ~1.5 GB for Whisper fp16, remainder for vLLM.

### Silence and beep gating

Every audio chunk passes an RMS energy check and an FFT spectral-entropy check before it reaches Whisper. Chunks below the RMS floor (silence) or dominated by a single frequency bin (censor beeps / PII tones) are dropped immediately. This means the ASR model — and downstream the LLM judge — never sees non-speech audio. On typical redacted call datasets this eliminates 30–50 % of chunks before any model is invoked.

### asyncio.Queue stage handoffs

The three pipeline stages (EARS → transcript queue → BRAIN → alarm queue → GUI) are connected by in-process asyncio queues. Handoff latency is microseconds with zero serialisation overhead. There are no threads, no IPC sockets, and no copies of the audio or transcript buffers between stages.

### Semaphore(16) concurrency cap

The LLM judge is gated by a `Semaphore(16)` on the client side. This matches vLLM's practical batching capacity for a 4B model on the available VRAM, prevents queue pile-ups under burst load from multiple simultaneous calls, and avoids GPU memory spikes that would cause context eviction. Requests beyond the cap wait in the asyncio queue at zero GPU cost.

### Sliding 3-turn context window

Each judge invocation receives only the last 3 utterances, not the full call transcript. The compliance rules that require sequential context (e.g., repeated negative sentiment, escalating language) operate on a 3-turn window by design. Keeping the prompt short holds token counts to ~150–300 input tokens per call, which directly drives the sub-500 ms judge latency seen in practice.

### Guided JSON decoding

The vLLM server grammar-constrains every judge response to the exact JSON schema at decode time. This eliminates retry loops, JSON repair passes, and any ambiguity in the output format. A single forward pass always produces a valid structured response.

### GPU memory budget auto-fitting

`run_vllm_server.sh` reserves ~7 GiB of free VRAM for Whisper before computing `--gpu-memory-utilization`. vLLM never grabs memory that Whisper needs, and Whisper never OOMs onto CPU mid-call. The fitted value is printed at startup so the actual budget is always visible.

### KV cache recommendations

For further gains without accuracy impact:

- `--enable-prefix-caching` — the `MODERATOR_SYSTEM_PROMPT` is identical across every judge call. vLLM computes its KV representation once and reuses it for all subsequent calls, reducing prefill cost on the largest fixed portion of every prompt.
- `--kv-cache-dtype fp8` — halves KV cache memory on MI300X hardware, allowing more concurrent sequences within the same VRAM budget.
- `--max-num-seqs 32` — aligns vLLM's pre-allocated concurrency slots with the `Semaphore(16)` cap, reducing idle KV cache reservation.
- `--block-size 32` — larger token blocks reduce memory management overhead for the multi-turn context window.
