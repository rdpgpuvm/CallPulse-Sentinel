# AI Call Moderator v3 — vLLM Serving, Guided JSON, Batched Turns, Staged Token Accounting

Continuation of v2 with the inference layer rebuilt on **vLLM**, following the serving pattern from the AMD `build_airbnb_agent_mcp` workshop notebook. The moderation logic, Kaggle MCP OAuth ingestion, and speaker identification carry over from v2 — what changes is *how* the model is run and *how precisely* tokens are measured.

## What was adopted from the workshop, and what wasn't

Adopted: launching vLLM in a Jupyter terminal as an OpenAI-compatible server (`--api-key`, `--served-model-name`, reduced `--max-model-len` for the lab), the `openai` client against `localhost:8000/v1`, and MCP for external data. Not adopted: **PydanticAI and the agent/tool-calling loop**. A compliance moderator is a deterministic pipeline — every turn must be judged, every call must get a report — so an agent deciding *whether* to call tools adds round-trips and scaffolding tokens with no accuracy benefit. Since performance, accuracy, and efficiency are graded, direct schema-constrained calls win on all three.

## 1. Serve the model once (terminal)

```bash
VLLM_USE_TRITON_FLASH_ATTN=0 \
vllm serve Qwen/Qwen3-4B-Instruct-2507 \
    --served-model-name call-moderator-llm \
    --api-key local-key-123 \
    --port 8000 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.85
```

Instruct variant (not the thinking model the workshop served) so no completion tokens are burned on `<think>` blocks. `--gpu-memory-utilization 0.85` leaves room for Whisper on the same GPU. No tool-call parser flags — there are no agent tool calls.

## 2. Guided JSON: accuracy by construction

v2 hoped the model emitted valid JSON and regex-extracted it. v3 makes invalid output impossible: vLLM's structured-output mode constrains token sampling to a JSON-schema grammar.

```python
TURN_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "sentiment":  {"enum": [-2, -1, 0, 1, 2]},                    # can't go off-scale
        "violations": {"type": "array", "items": {"enum": list(POLICY)}},  # only real codes
        "reason":     {"type": "string", "maxLength": 90},
    },
    "required": ["sentiment", "violations", "reason"],
}

response = await async_client.chat.completions.create(
    model="call-moderator-llm",
    messages=[...],
    temperature=0,                                  # greedy -> reproducible
    max_tokens=72,
    extra_body={"guided_json": TURN_ANALYSIS_SCHEMA},  # vLLM structured output
)
```

The same applies to the end-of-call report (`rating` constrained to 1–5, each standard-procedure item to 0/1), the mono-transcript speaker split, and the role tiebreak. Hallucinated violation codes and unparseable replies are eliminated, and no tokens are wasted on retries.

## 3. Performance: batched turn analysis

Each turn's judgment needs only the *transcript* of the preceding turns — fully known upfront for a recorded call — and per-turn judgments never feed each other. So all turns are dispatched concurrently and vLLM's continuous batching processes them together; multiple calls are batched too:

```python
analyses = await asyncio.gather(*[
    analyze_turn(turns[:index], speaker_role, turn_text, call_id)
    for index, (speaker_role, turn_text) in enumerate(turns)
])
```

Correctness is preserved because escalation is decided afterward by a sequential, deterministic controller that scans the analyses in order and finds the *earliest* rule trigger (critical violation → immediate; two high-severity → escalate; customer sentiment ≤ −2 twice in a row → supervisor assist). Batched results are identical to one-at-a-time results — just several times faster. For live streaming, the same `analyze_turn` runs once per incoming turn with the controller applied incrementally.

## 4. Token accounting: per stage, per call, combined

Every response from vLLM carries exact `usage` counts — no tokenizer re-counting. Each LLM call is tagged with its pipeline stage and the call being moderated:

```python
STAGE_TOKEN_USAGE    = defaultdict(lambda: {"calls": 0, "prompt": 0, "completion": 0})
PER_CALL_TOKEN_USAGE = defaultdict(lambda: {"calls": 0, "prompt": 0, "completion": 0})

def record_usage(stage, usage, call_id=None):
    for bucket in [STAGE_TOKEN_USAGE[stage]] + ([PER_CALL_TOKEN_USAGE[call_id]] if call_id else []):
        bucket["calls"]      += 1
        bucket["prompt"]     += usage.prompt_tokens
        bucket["completion"] += usage.completion_tokens
```

Section 8 prints three views: tokens **by stage** (`speaker_split`, `role_identification`, `turn_moderation`, `end_of_call_report`) with calls / prompt / completion / total / average-per-call columns, tokens **by moderated call** including tokens-per-turn, and the **combined** grand total. Expected shape: `turn_moderation` dominates (~300 prompt / ~40 completion per turn); speaker logic stays near zero because stereo separation is free and the keyword-first role classifier only falls back to a ~24-token LLM tiebreak on ambiguous openings.

## 5. Everything else

Kaggle ingestion is unchanged from v2: official Kaggle MCP server, OAuth 2.0 with the manual paste flow for headless labs, runtime tool discovery, automatic audio/CSV routing. The policy content (`POLICY`, `STANDARD_PROCEDURE`, `COMPANY_POLICY_ALLOWANCES`), the zero-token regex pre-screener, and the escalation rules are also unchanged — edit them and re-run. Caps: `MAX_CALLS_TO_MODERATE`, `MAX_TURNS_PER_CALL`. To try a bigger judge, change only the `vllm serve` command; the notebook code stays identical.
