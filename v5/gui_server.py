#!/usr/bin/env python3
"""Live dashboard for the AI Call Moderator (v5).

The notebook pipeline POSTs events here (/event); every connected browser gets them
pushed over a WebSocket and renders them instantly:
  - two-sided conversation bubbles, labeled CUSTOMER REP (left/blue) vs CUSTOMER (right/green)
  - violation turns highlighted red with policy-code chips + the judge's reason
  - a flashing escalation banner the moment a rule trips, with judge latency
  - per-turn timing chips (asr / queue / judge ms)

Runs on fastapi+uvicorn (already installed for vLLM). Started automatically by
run_vllm_server.sh; open it at <your Jupyter base URL>/proxy/7860/
"""
import argparse
import json

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

app = FastAPI()
EVENT_LOG = []          # full history -> late-joining browsers replay everything
CONNECTED = set()       # live websocket clients

PAGE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>AI Call Moderator — LIVE</title>
<style>
  :root { --bg:#0d1117; --panel:#161b22; --line:#30363d; --text:#e6edf3; --dim:#8b949e;
          --rep:#1f6feb; --cust:#238636; --bad:#da3633; --warn:#d29922; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--text);
         font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; }
  header { padding:14px 22px; border-bottom:1px solid var(--line); display:flex;
           align-items:center; gap:14px; position:sticky; top:0; background:var(--bg); z-index:5; }
  header h1 { font-size:17px; margin:0; letter-spacing:.5px; }
  #status { color:var(--dim); font-size:13px; }
  .dot { width:9px; height:9px; border-radius:50%; background:var(--warn); display:inline-block; }
  .dot.live { background:#3fb950; }
  #calls { padding:18px; display:grid; gap:18px; }
  .call { background:var(--panel); border:1px solid var(--line); border-radius:10px; overflow:hidden; }
  .call.escalated { border-color:var(--bad); box-shadow:0 0 0 1px var(--bad); }
  .call h2 { margin:0; padding:10px 16px; font-size:13px; color:var(--dim);
             border-bottom:1px solid var(--line); font-weight:600; }
  .banner { background:var(--bad); color:#fff; padding:10px 16px; font-weight:700;
            font-size:13px; animation:flash 1s linear 6; }
  @keyframes flash { 50% { filter:brightness(1.6);} }
  .turns { padding:14px 16px; display:flex; flex-direction:column; gap:10px; }
  .turn { max-width:72%; padding:9px 13px; border-radius:12px; font-size:14px; line-height:1.45;
          position:relative; }
  .turn .who { font-size:10.5px; font-weight:700; letter-spacing:.8px; opacity:.85;
               margin-bottom:3px; text-transform:uppercase; }
  .turn.rep      { align-self:flex-start; background:rgba(31,111,235,.16);
                   border:1px solid rgba(31,111,235,.45); }
  .turn.rep .who { color:#79b8ff; }
  .turn.customer { align-self:flex-end;   background:rgba(35,134,54,.16);
                   border:1px solid rgba(35,134,54,.5); }
  .turn.customer .who { color:#56d364; }
  .turn.violation { background:rgba(218,54,51,.18); border:1.5px solid var(--bad); }
  .chips { margin-top:6px; display:flex; gap:6px; flex-wrap:wrap; }
  .chip { font-size:10.5px; padding:2px 8px; border-radius:10px; font-weight:700; }
  .chip.code   { background:var(--bad); color:#fff; }
  .chip.reason { background:rgba(218,54,51,.25); color:#ffa198; font-weight:500; }
  .chip.sent   { background:rgba(110,118,129,.3); color:var(--dim); }
  .chip.time   { background:rgba(110,118,129,.18); color:var(--dim); font-weight:500; }
</style></head><body>
<header><h1>&#128737; AI CALL MODERATOR &mdash; LIVE</h1>
  <span class="dot" id="dot"></span><span id="status">connecting&hellip;</span></header>
<div id="calls"></div>
<script>
const wsProto = location.protocol === 'https:' ? 'wss://' : 'ws://';
let base = location.pathname; if (!base.endsWith('/')) base += '/';
const ws = new WebSocket(wsProto + location.host + base + 'ws');
const calls = {};
ws.onopen  = () => { dot.classList.add('live'); status.textContent = 'live — waiting for the pipeline'; };
ws.onclose = () => { dot.classList.remove('live'); status.textContent = 'disconnected'; };
ws.onmessage = (e) => render(JSON.parse(e.data));

function panel(callId) {                       // one panel per call, created on first event
  if (!calls[callId]) {
    const div = document.createElement('div');
    div.className = 'call'; div.id = 'call-' + callId;
    div.innerHTML = '<h2>CALL ' + callId + '</h2><div class="turns"></div>';
    document.getElementById('calls').appendChild(div);
    calls[callId] = div;
  }
  return calls[callId];
}
function render(ev) {
  if (ev.type === 'turn') {
    const p = panel(ev.call_id);
    const t = document.createElement('div');
    t.className = 'turn ' + ev.role + (ev.violations.length ? ' violation' : '');
    const who = ev.role === 'rep' ? 'Customer Rep' : 'Customer';
    let chips = '<span class="chip sent">sent ' + (ev.sentiment >= 0 ? '+' : '') + ev.sentiment + '</span>' +
                '<span class="chip time">asr ' + ev.asr_ms + 'ms &middot; queue ' + ev.queue_ms +
                'ms &middot; judge ' + ev.judge_ms + 'ms</span>';
    ev.violations.forEach(v => chips = '<span class="chip code">' + v + '</span>' + chips);
    if (ev.violations.length && ev.reason)
      chips += '<span class="chip reason">' + ev.reason + '</span>';
    t.innerHTML = '<div class="who">' + who + ' &middot; t' + ev.turn_number + '</div>' +
                  ev.text + '<div class="chips">' + chips + '</div>';
    p.querySelector('.turns').appendChild(t);
    t.scrollIntoView({behavior:'smooth', block:'end'});
    status.textContent = 'live — last turn t' + ev.turn_number + ' (' + ev.call_id + ')';
  } else if (ev.type === 'alert') {
    const p = panel(ev.call_id);
    p.classList.add('escalated');
    const b = document.createElement('div');
    b.className = 'banner';
    b.textContent = '\\u{1F6A8} ESCALATED at turn ' + ev.turn_number + ' — ' + ev.rule +
                    '  (judge ' + ev.judge_ms + ' ms)';
    p.insertBefore(b, p.querySelector('.turns'));
    document.title = '\\u{1F6A8} ESCALATION — AI Call Moderator';
  } else if (ev.type === 'status') {
    status.textContent = ev.text;
  }
}
</script></body></html>"""


@app.get("/")
async def index():
    return HTMLResponse(PAGE)


# Some Jupyter proxies forward the FULL path (/proxy/7860/...) instead of stripping it.
# Catch-all: any GET that isn't an API route serves the dashboard, so the URL always works.
@app.get("/{full_path:path}")
async def index_any(full_path: str):
    return HTMLResponse(PAGE)


@app.post("/event")
async def receive_event(event: dict):
    """Notebook pipeline -> dashboard. Stores + broadcasts to every open browser."""
    EVENT_LOG.append(event)
    dead = []
    for ws in CONNECTED:
        try:
            await ws.send_text(json.dumps(event))
        except Exception:
            dead.append(ws)
    for ws in dead:
        CONNECTED.discard(ws)
    return {"ok": True, "clients": len(CONNECTED)}


@app.websocket("/ws")
@app.websocket("/{prefix:path}/ws")          # tolerate full-path forwarding for the socket too
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    for event in EVENT_LOG:                    # replay history for late joiners
        await ws.send_text(json.dumps(event))
    CONNECTED.add(ws)
    try:
        while True:
            await ws.receive_text()            # keepalive; we never expect client messages
    except WebSocketDisconnect:
        CONNECTED.discard(ws)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")
