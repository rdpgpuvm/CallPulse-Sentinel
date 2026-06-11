# AI Call Moderator v5 — Live GUI Dashboard

v5 adds a **real-time web dashboard** on top of the v4 pipeline: rep and customer rendered as opposite-side chat bubbles with their identities labeled, violation turns highlighted red with policy-code chips and the judge's reason, per-turn timing chips (asr / queue / judge ms), and a flashing escalation banner the instant a rule trips.

## Why a web GUI (and not a desktop window)

The AMD dev portal is a browser-based JupyterLab — there is no desktop to pop a PyQt window on. So the GUI is a tiny **FastAPI + WebSocket** server (`gui_server.py`, ~one file) that the runner starts *alongside vLLM*. The notebook pipeline pushes events to it over localhost; every connected browser renders them instantly. This maps the architecture document's "The Alarm" UI onto what the lab can actually display — and it's the same pattern you'd ship for a real supervisor dashboard.

## How to run

**Tab 1 — Terminal:**

```bash
git clone -b v5-gui https://github.com/rdpgpuvm/Project1.git /workspace/CallModV5
cd /workspace/CallModV5/v5 && bash run_vllm_server.sh
```

The runner now has 5 steps: deps + starlette pin → vLLM import check → **start GUI on port 7860** → kill stale vLLM + auto-fit GPU budget → launch vLLM. The GUI keeps running even if you Ctrl-C vLLM.

**Browser — open the dashboard.** JupyterLab proxies local ports via jupyter-server-proxy:

```
<your Jupyter base URL>/proxy/7860/
```

CELL 3b in the notebook prints the exact URL for your lab (it reads `JUPYTERHUB_SERVICE_PREFIX`) **and embeds the dashboard as an IFrame inside the notebook**, so even if you never open a separate tab, the GUI is right there under the cell.

**Tab 2 — Notebook:** run `call_moderator_v5_gui.ipynb` cells one by one (same flow as v4: health check → policy → Kaggle OAuth → download → Ears → Brain → Alarm → run). As CELL 8 streams the calls, bubbles appear in the dashboard live.

## How the GUI is wired (zero impact on the pipeline)

The pipeline never waits on the GUI. `emit_event` is fire-and-forget with a 2-second budget, and if the dashboard isn't running it disables itself after the first failure:

```python
async def emit_event(event: dict):
    global GUI_AVAILABLE
    if not GUI_AVAILABLE: return
    try:    await gui_client.post(GUI_EVENT_URL, json=event)
    except Exception: GUI_AVAILABLE = False      # GUI down -> keep moderating, no errors
```

The Brain emits one `turn` event per judged utterance (role, text, sentiment, violations, reason, asr/queue/judge ms); the Alarm emits one `alert` event per escalation. The server stores the full event log, so a browser opened mid-run replays everything it missed.

```
notebook pipeline ──POST /event──> gui_server.py ──WebSocket──> every open browser
   (Brain + Alarm)                  (port 7860)                  (bubbles + banners)
```

## What you see

Left/blue bubbles labeled **CUSTOMER REP**, right/green bubbles labeled **CUSTOMER** (identities from the v4 role classifier). Turns that broke policy turn red with code chips (`C1`, `R2`, …) and the judge's one-line reason. Every bubble carries timing chips. When a rule trips, the call panel gets a red border and a flashing `🚨 ESCALATED at turn N — rule …` banner, and the page title changes so it's visible even from another browser tab.

## Troubleshooting

If `/proxy/7860/` returns 404, jupyter-server-proxy may be absent — use the IFrame in CELL 3b (it goes through the same proxy; if both fail, ask the lab admins to enable jupyter-server-proxy). If the dashboard loads but stays empty, check `GUI_AVAILABLE` in the notebook and `/tmp/call_moderator_gui.log` on the box. The GUI process can be restarted alone with `python3 gui_server.py --port 7860 &`.
