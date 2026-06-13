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
# LANGFUSE_HOST=http://localhost:3000
```

### Mode B — Langfuse cloud (free tier, account at langfuse.com)

```bash
cp .env.example .env
# Fill in LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY from langfuse.com
# Leave LANGFUSE_HOST commented out (defaults to cloud)
```

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
