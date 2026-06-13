# AI Call Moderator v7 — Supervisor Override + Skip Validation

v7 builds on the v6 deadline-pacing pipeline with two new supervisor tools:

## New in v7

### ⚡ Supervisor Override with Escalation Jump

When a call is escalated, an **OVERRIDE** button appears in the call header. Clicking it:
1. Reveals a list of all escalated segments with their **audio timestamps** (`mm:ss`)
2. Clicking any item **seeks the audio player** to that exact moment — so the supervisor can hear the context before joining
3. After clicking override, a **👥👥 SUPERVISING** badge appears in the center of the global header, indicating three parties are on the call (simulation)

### ⚠️ Skipped Segment Validation Panel

Every chunk the ASR pipeline silently dropped is now listed in a collapsible panel at the bottom of each call:
- **silence** — RMS below threshold (almost certainly fine)
- **beep** — tone-dominated spectrum (censor bleep / redacted PII)
- **no_speech** — Whisper's own confidence gate said P(no_speech) > 0.5
- **low_confidence** — avg_logprob < -1.0 (very uncertain decode)
- **repetition** — compression_ratio > 2.4 (hallucination loop)

Clicking any item seeks the audio to that timestamp so a supervisor can **validate by ear** whether the skip was genuine silence/beep or potentially missed speech.

### Architecture additions

- `audio_start_s` added to all `turn`, `alert`, and new `skip` events — the audio offset in seconds so the UI can seek without any server-side mapping
- `quality_gated()` now returns `(good_segs, skipped_info)` — the skipped list flows to the GUI rather than being silently discarded
- Skip events emitted inline in `ears_producer` for live-mode silence/beep/no_speech, and from `quality_gated` for file-mode per-segment rejects

## Recordings included

Five real call-center recordings from the [Unidata Call Center Audio Dataset](https://www.kaggle.com/datasets/unidpro/call-center-audio) (CC BY-NC-ND 4.0):

| File | Company | Duration | Escalation signals |
|---|---|---|---|
| `CA769e290725c8cb356344c837470375f2.flac` | Amazon | 26 min | 🔴 Repeated refund denials, customer frustration → **sentiment rule** |
| `CA4950c1c8c305cc85c5f5f040229fe608.flac` | Ubereats | 10 min | 🟡 Refund dispute |
| `CA0fe99171c6dec5c26dbe5fa5d10c863a.flac` | Prezzee | 15 min | 🟡 Cancellation |
| `CA5f229fc25030bdc650d548bdcf95780f.flac` | Paramount Plus | 7 min | 🟡 Cancellation |
| `CA3e0c1114f78be6bd450860973c404dba.flac` | Spin | 5 min | 🟡 Refund |

**Best for testing override UI:** use `CA769e290725c8cb356344c837470375f2` (Amazon, 26 min) — the customer's persistent frustration about missing refunds across multiple orders is likely to trigger the sentiment-based escalation rule (rule 3: sentiment ≤ −2 two turns in a row).

For additional recordings with escalatable content (angry customers, rep misconduct), the original dataset is at:
- **Kaggle**: https://www.kaggle.com/datasets/unidpro/call-center-audio (CC BY-NC-ND 4.0)
- **PissedConsumer**: https://www.pissedconsumer.com/call-recordings.html (source of transcripts)

## How to run

Same as v5/v6 — one terminal for vLLM, one for the notebook:

```bash
git clone -b v7 https://github.com/rdpgpuvm/Project1.git /workspace/CallModV7
cd /workspace/CallModV7/v5 && bash run_vllm_server.sh
```

Then run `call_moderator_v5_gui.ipynb` cell by cell. Set `SELECTED_CALL_ID = "CA769e290725c8cb356344c837470375f2"` in CELL 8 to target the Amazon recording for the best chance of seeing the override panel.

## Architecture is unchanged (accuracy + speed preserved)

All v6 optimizations carry forward:
- Deadline pacing (chunk k releases at `stream_start + k*5s`)
- GPU STT warm-up at load (pre-pays the 24.8s ONNX kernel-compile cost)
- CPU faster-whisper int8 fallback

v7 adds zero overhead to the moderation pipeline — `audio_start_s` is a free field copy, and skip events are fire-and-forget (same `emit_event` path already used for turns/alerts).
