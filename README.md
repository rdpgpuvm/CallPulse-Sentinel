# AI Call Moderator — Real-Time Compliance Pipeline

A two-model pipeline that monitors live call center calls for policy violations.
**Whisper** (ASR) transcribes audio; **Qwen3-4B-Instruct** (LLM judge via vLLM on AMD GPU)
flags violations in under one second. A live FastAPI/WebSocket dashboard gives supervisors
override controls, audio seeking, and escalation review.

---

## Quick Start

```bash
# Clone the branch you want
git clone -b v7      https://github.com/rdpgpuvm/Project1.git /workspace/CallModV7
git clone -b vosktest https://github.com/rdpgpuvm/Project1.git /workspace/CallModVosk

# Start vLLM + GUI dashboard (leave this terminal running)
cd /workspace/CallModV7
bash run_vllm_server.sh

# Then run call_moderator_v5_gui.ipynb cell by cell in Tab 2
```

---

## Branch Differences

| | v7 | vosktest |
|---|---|---|
| ASR | whisper-large-v3-turbo (CPU int8 file / GPU fp16 live) | Vosk/Kaldi CPU-only |
| GPU for ASR | Optional (GPU fp16 live mode) | Never — 100% CPU |
| GPU for LLM | Always (vLLM AMD ROCm) | Always (vLLM AMD ROCm) |
| When to use | Best accuracy | GPU-constrained / CPU-only fallback |

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

---

## Recordings included

| Folder | Files | Source |
|---|---|---|
| `call_recordings/` | 5 FLAC files (Amazon, Ubereats, Prezzee, Paramount+, Spin) | [Kaggle unidpro/call-center-audio](https://www.kaggle.com/datasets/unidpro/call-center-audio) |
| `scam_call/` | 4 WAV files | Committed scam call recordings |

Best for testing override UI: `CA769e290725c8cb356344c837470375f2` (Amazon, 26 min) —
set `SELECTED_CALL_ID` in Cell 8 to target it.

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

## Performance & Efficiency Design

The pipeline was built to be resource-conscious without sacrificing accuracy. Every architectural choice has a specific reason.

### CPU/GPU split

Whisper (ASR) runs on CPU in int8 quantization. This frees the AMD GPU entirely for the vLLM judge — the only stage that actually requires GPU compute. Running both on GPU simultaneously causes memory contention and causes Whisper to OOM-evict vLLM's KV cache mid-call. The CPU int8 transcription runs at ~200–250 ms per 5 s chunk, comfortably within the pacing window.

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
