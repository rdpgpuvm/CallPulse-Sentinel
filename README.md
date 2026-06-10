# AI Call Moderator

A Jupyter notebook that uses a **local LLM** (`Qwen/Qwen3-4B-Instruct-2507`) as a real-time call-center moderator: it scores customer sentiment every turn, detects policy/SOP violations, escalates to a supervisor by deterministic rules, and produces an end-of-call QA report (CSAT 1–5, SOP checklist, summary, coaching tip).

## Run it

1. Open `call_moderator.ipynb` in Jupyter (works on AMD GPUs — PyTorch ROCm exposes the GPU via the `cuda` API).
2. Run all cells top to bottom. Cell 1 installs `torch / transformers / accelerate` if missing; the model (~8 GB bf16) downloads on first load.
3. Section 7 streams five simulated calls through the moderator; Section 8 grades the run against expected labels and prints token usage.

## Design

- **Hybrid pipeline**: a zero-token regex pre-screener flags keyword *hints* → a single compact LLM call per turn (last 4 turns of context, strict minified-JSON output, greedy decoding) judges in context → a rule-based controller decides escalation. Rules, not the model, pull the trigger — auditable and cheap (~300 prompt / ~40 output tokens per turn).
- **Escalation rules**: any critical violation → immediate; 2+ high violations → escalate; customer sentiment ≤ −2 two turns in a row → supervisor assist.
- **Violation codes**: C1 sensitive-data abuse, C2 threats/abuse, C3 unethical conduct (bribes, off-the-books deals), C4 unauthorized promises, R1 skipped verification, R2 rudeness, R3 internal-data disclosure. SOP items S1–S4 (greeting, verification, recap, closing) are scored at call end.
- **Test set**: 5 labeled scenarios — clean call, furious customer (sentiment escalation), rep misconduct, a false-positive trap (rep correctly refuses an improper request — must NOT escalate), and an unethical rep (immediate escalation).
- **Why not Qwen-Audio 8B**: transcripts are text; a 4B instruct model is more accurate at schema-following, half the memory, no thinking-token overhead. Section 9 has a Whisper stub for real audio input.

## Customize

All enforcement lives in the `POLICY`, `SOP`, `ALLOWED`, and `ESCALATION_RULES` definitions in Section 2 — edit and re-run. Swap `MODEL_ID` in Section 4 for a larger judge (e.g. `Qwen/Qwen2.5-7B-Instruct`).
