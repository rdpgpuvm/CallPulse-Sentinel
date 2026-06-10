# AI Call Moderator v2 — Real Recordings via Kaggle MCP (OAuth 2.0)

Continuation of v1. The simulated scenarios are gone: v2 pulls real call-center recordings from Kaggle through its **official MCP server** with **OAuth 2.0**, figures out **who is the rep and who is the customer**, then runs the same turn-by-turn moderation engine. Renames in this version: `SOP` → `STANDARD_PROCEDURE`, `CSAT` → `rating` (the predicted 1–5 customer satisfaction rating), and all variables are self-explanatory (`tokenizer`, `language_model`, `TOKEN_USAGE`, `escalation_reason`, …).

## Run it

Open `call_moderator_v2.ipynb` and run top to bottom. Cell 1 installs `mcp`, `httpx`, `librosa`, `soundfile` alongside the v1 stack. The OAuth consent happens in your own browser on first connect (Section 2); pick the dataset in Section 3.

## 1. Kaggle authentication is OAuth, not API keys

The notebook is an MCP client. The Python `mcp` SDK handles the whole OAuth 2.0 dance (dynamic client registration → browser consent → authorization code → access token), and tokens are held in RAM only:

```python
kaggle_oauth = OAuthClientProvider(
    server_url="https://www.kaggle.com/mcp",
    client_metadata=OAuthClientMetadata(
        client_name="call-moderator-v2-notebook",
        redirect_uris=["http://localhost:8765/callback"],
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
    ),
    storage=token_storage,                    # in-memory, nothing written to disk
    redirect_handler=show_authorization_url,  # prints the consent URL for you to open
    callback_handler=wait_for_pasted_callback # you paste the redirect URL back in
)
```

Because the AMD lab is headless, the redirect can't be caught by a local server. The standard manual flow is used instead: the notebook prints the consent URL, you approve in your browser, the browser lands on a `localhost:8765/...` page that fails to load (expected — nothing listens there), and you paste that full URL back into the notebook. The SDK extracts the `code` and exchanges it for tokens automatically. Refresh tokens are handled by the provider, and the same access token is reused for the dataset file download.

## 2. Tool discovery instead of hardcoded tool names

Kaggle's MCP schema can evolve, so the notebook never hardcodes tool names. It reads the live tool list and picks by keyword, then tries a few plausible argument spellings:

```python
def find_tool(*name_keywords):
    for tool in KAGGLE_TOOLS:                      # from kaggle_session.list_tools()
        if all(k in tool.name.lower() for k in name_keywords):
            return tool.name

search_tool_name   = find_tool("search", "dataset")
download_tool_name = find_tool("download", "dataset")

search_results = await call_kaggle(search_tool_name, [
    {"query": "call center audio english"}, {"search": ...}, {"q": ...},  # tried in order
])
```

The printed `list_tools()` output is the source of truth — if Kaggle renames something, you adjust one keyword, not the pipeline. Set `DATASET_REFERENCE` (e.g. `unidpro/call-center-audio`) after inspecting the search results; transcript-only datasets (CSV) work too via an automatic text fallback.

## 3. Speaker separation — cheapest mechanism first

Telephony recorders usually write **stereo** files with one party per channel, so separation is physical and free: transcribe each channel independently with Whisper and interleave by timestamp.

```python
samples, sampling_rate = librosa.load(audio_path, sr=16000, mono=False)
if samples.ndim == 2 and samples.shape[0] == 2:        # stereo: agent on ch0, caller on ch1 (or vice versa)
    for channel_index in (0, 1):
        for utterance in transcribe_samples(samples[channel_index], sampling_rate):
            utterances.append({**utterance, "speaker": f"speaker_{channel_index}"})
    utterances.sort(key=lambda u: u["start"])           # interleave into conversation order
```

For **mono** files no physical separation exists, so exactly one LLM call splits the raw transcript into labeled turns by context — greetings, account questions and requests make the roles obvious to a context-aware model:

```python
TRANSCRIPT_SPLIT_PROMPT = (
    "You segment a raw call-center transcript into speaker turns and label each turn "
    "'rep' or 'customer'. Keep the original wording. "
    'Reply with ONLY minified JSON: {"turns":[{"role":"rep","text":"..."}, ...]}'
)
```

## 4. Role identification — who is the rep?

Stereo channels arrive anonymous (`speaker_0`/`speaker_1`). Roles are assigned in two stages, cheapest first. Stage 1 costs zero tokens: count near-scripted marker phrases per speaker (+1 for rep markers like *"thank you for calling"*, *"is there anything else"*; −1 for customer markers like *"my bill"*, *"I want a refund"*). A score margin ≥ 2 settles it. Only on a tie does stage 2 spend one ~24-token LLM call on the call opening:

```python
score_margin = abs(marker_scores[speakers[0]] - marker_scores[speakers[1]])
if score_margin >= 2:                                   # stage 1: keywords decide, 0 tokens
    rep_speaker = max(speakers, key=lambda s: marker_scores[s])
else:                                                   # stage 2: tiny LLM tiebreak
    answer = generate_json(ROLE_TIEBREAK_PROMPT, f"Call opening:\n{call_opening}\nJSON:",
                           max_new_tokens=24)
    rep_speaker = answer.get("rep", speakers[0])
```

Since reps open with near-scripted greetings, stage 1 resolves almost every call — average role-identification cost is close to zero tokens. Consecutive same-role utterances are then merged into single turns, which also cuts the number of moderation LLM calls.

## 5. Moderation (unchanged logic, clearer names)

One compact LLM call per turn returns `{"sentiment", "violations", "reason"}`; deterministic rules in code — not the model — decide escalation (any critical violation; two high-severity violations; customer sentiment ≤ −2 two turns in a row). The end-of-call report now uses the new names:

```python
REPORT_SYSTEM_PROMPT = (
    'Reply with ONLY minified JSON: {"rating":<1-5 predicted customer satisfaction rating>,'
    '"procedure":{"S1":0 or 1,"S2":0 or 1,"S3":0 or 1,"S4":0 or 1},'
    '"summary":"<=25 words","coaching":"<=18 words of advice for the rep"}'
)
```

`STANDARD_PROCEDURE` items: S1 greeting, S2 identity verification, S3 resolution recap, S4 polite closing. All enforcement lives in `POLICY`, `STANDARD_PROCEDURE`, and `COMPANY_POLICY_ALLOWANCES` — edit and re-run. `MAX_CALLS_TO_MODERATE` and `MAX_TURNS_PER_CALL` cap the token budget; Section 9 prints exact prompt/output token spend. Real recordings carry no ground-truth labels, so v1's labeled scenarios remain the accuracy benchmark while v2 reports cost.
