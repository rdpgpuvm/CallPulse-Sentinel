#!/usr/bin/env python3
"""Live dashboard for the AI Call Moderator (v5).

The notebook pipeline POSTs events here (/event); browsers receive them over a
WebSocket, or — when the lab's proxy blocks WS upgrades — via 1s HTTP polling
(automatic fallback). Renders, per call:
  - a STAGE: two human figures (CUSTOMER REP left, CUSTOMER right) that light up
    while their person is speaking, with the live utterance shown between them
    (speaker-colored; red when a violation is detected)
  - the full conversation as two-sided bubbles with violation highlighting,
    policy-code chips, the judge's reason, and per-turn timing chips
  - a flashing escalation banner the instant a rule trips
  - CLEAN MODE toggle: history hidden, utterances fade out after a few seconds
  - a play/pause button per call that streams the original recording (unsynced
    with the analysis — playback is for debugging by ear)
  - smart scrolling: auto-follow only when the viewer is already at the bottom

Started automatically by run_vllm_server.sh on port 7860.
Open at <your Jupyter base URL>/proxy/7860/
"""
import argparse
import json
import pathlib

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

app = FastAPI()
EVENT_LOG = []                                   # full history -> late joiners replay everything
CONNECTED = set()                                # live websocket clients
AUDIO_DIRS = [pathlib.Path("call_recordings"),   # committed in the repo (teammate mode)
              pathlib.Path("kaggle_call_data")]  # left behind by a Kaggle pull
AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".m4a", ".ogg")


def find_audio_file(call_id: str):
    """Locate the recording whose filename stem matches the call id (either source dir)."""
    for directory in AUDIO_DIRS:
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            if path.stem == call_id and path.suffix.lower() in AUDIO_EXTENSIONS:
                return path
    return None


PAGE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>AI Call Moderator — LIVE</title>
<style>
  :root { --bg:#0d1117; --panel:#161b22; --line:#30363d; --text:#e6edf3; --dim:#8b949e;
          --rep:#1f6feb; --repglow:#79b8ff; --cust:#238636; --custglow:#56d364;
          --bad:#da3633; --warn:#d29922; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--text);
         font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; }
  header { padding:14px 22px; border-bottom:1px solid var(--line); display:flex;
           align-items:center; gap:14px; position:sticky; top:0; background:var(--bg); z-index:5; }
  header h1 { font-size:17px; margin:0; letter-spacing:.5px; }
  #status { color:var(--dim); font-size:13px; flex:1; }
  .dot { width:9px; height:9px; border-radius:50%; background:var(--warn); display:inline-block; }
  .dot.live { background:#3fb950; }
  .hbtn { background:var(--panel); color:var(--text); border:1px solid var(--line);
          border-radius:8px; padding:6px 14px; font-size:12.5px; cursor:pointer; }
  .hbtn:hover { border-color:var(--dim); }
  .hbtn.on { border-color:var(--repglow); color:var(--repglow); }
  #tabs { display:flex; gap:8px; padding:12px 18px 0; flex-wrap:wrap; }
  .tab { background:var(--panel); border:1px solid var(--line); color:var(--dim); cursor:pointer;
         border-radius:8px; padding:6px 12px; font-size:12px; font-weight:600; }
  .tab:hover { border-color:var(--dim); }
  .tab.selected { border-color:var(--repglow); color:var(--repglow); }
  .tab.alerted  { border-color:var(--bad); color:#ffa198; animation:flash 1s linear 6; }
  #calls { padding:18px; display:grid; gap:18px; }
  .call { display:none; }                 /* one call at a time — pick via the tab bar */
  .call.selected { display:block; }
  .call { background:var(--panel); border:1px solid var(--line); border-radius:10px; }
  .call.escalated { border-color:var(--bad); box-shadow:0 0 0 1px var(--bad); }
  .call h2 { margin:0; padding:10px 16px; font-size:13px; color:var(--dim);
             border-bottom:1px solid var(--line); font-weight:600;
             display:flex; align-items:center; gap:10px; }
  .play { background:none; border:1px solid var(--line); color:var(--text); cursor:pointer;
          border-radius:6px; width:30px; height:24px; font-size:12px; line-height:1; }
  .play:hover { border-color:var(--repglow); color:var(--repglow); }
  .banner { background:var(--bad); color:#fff; padding:10px 16px; font-weight:700;
            font-size:13px; animation:flash 1s linear 6; }
  @keyframes flash { 50% { filter:brightness(1.6);} }

  /* ===== THE STAGE: sticky under the header while the history scrolls beneath ===== */
  .stage { display:grid; grid-template-columns:120px 1fr 120px; align-items:center;
           gap:14px; padding:18px 20px 14px; position:sticky; top:53px; z-index:4;
           background:var(--panel); border-bottom:1px solid var(--line);
           border-radius:10px 10px 0 0; }
  .person { text-align:center; }
  .person .figure { font-size:52px; line-height:1; opacity:.35; filter:grayscale(.9);
                    transition:all .25s ease; display:inline-block; }
  .person .plabel { font-size:10px; font-weight:700; letter-spacing:1px; color:var(--dim);
                    margin-top:6px; text-transform:uppercase; transition:color .25s; }
  .person.speaking .figure { opacity:1; filter:none; transform:scale(1.18); }
  .person.rep.speaking .figure      { filter:drop-shadow(0 0 14px var(--rep)); }
  .person.customer.speaking .figure { filter:drop-shadow(0 0 14px var(--cust)); }
  .person.rep.speaking .plabel      { color:var(--repglow); }
  .person.customer.speaking .plabel { color:var(--custglow); }
  .speech { min-height:64px; border:1.5px solid var(--line); border-radius:12px;
            padding:12px 16px; font-size:15px; line-height:1.5; color:var(--dim);
            transition:all .25s ease; opacity:1; }
  .speech.rep      { border-color:var(--rep);  color:var(--text);
                     background:rgba(31,111,235,.10); }
  .speech.customer { border-color:var(--cust); color:var(--text);
                     background:rgba(35,134,54,.10); }
  .speech.violation { border-color:var(--bad); background:rgba(218,54,51,.14);
                      box-shadow:0 0 12px rgba(218,54,51,.35); }
  .speech .vio { margin-top:6px; font-size:12px; color:#ffa198; font-weight:600; }
  .speech.fading { opacity:0; transition:opacity 2s ease; }  /* clean-mode fade-out */

  /* ===== conversation history (hidden entirely in clean mode) ===== */
  body.clean .turns { display:none; }
  body.clean .stage { border-radius:10px; border-bottom:none; }
  .turns { padding:14px 16px; display:flex; flex-direction:column; gap:10px; }
  .turn { max-width:72%; padding:9px 13px; border-radius:12px; font-size:14px; line-height:1.45; }
  .turn .who { font-size:10.5px; font-weight:700; letter-spacing:.8px; opacity:.85;
               margin-bottom:3px; text-transform:uppercase; }
  .turn.rep      { align-self:flex-start; background:rgba(31,111,235,.16);
                   border:1px solid rgba(31,111,235,.45); }
  .turn.rep .who { color:var(--repglow); }
  .turn.customer { align-self:flex-end; background:rgba(35,134,54,.16);
                   border:1px solid rgba(35,134,54,.5); }
  .turn.customer .who { color:var(--custglow); }
  .turn.violation { background:rgba(218,54,51,.18); border:1.5px solid var(--bad); }
  .chips { margin-top:6px; display:flex; gap:6px; flex-wrap:wrap; }
  .chip { font-size:10.5px; padding:2px 8px; border-radius:10px; font-weight:700; }
  .chip.code   { background:var(--bad); color:#fff; }
  .chip.reason { background:rgba(218,54,51,.25); color:#ffa198; font-weight:500; }
  .chip.sent   { background:rgba(110,118,129,.3); color:var(--dim); }
  .chip.time   { background:rgba(110,118,129,.18); color:var(--dim); font-weight:500; }
</style></head><body>
<header><h1>&#128737; AI CALL MODERATOR &mdash; LIVE</h1>
  <span class="dot" id="dot"></span><span id="status">connecting&hellip;</span>
  <button class="hbtn" id="cleanToggle">&#10024; Clean mode</button></header>
<div id="tabs"></div>
<div id="calls"></div>
<script>
let base = location.pathname; if (!base.endsWith('/')) base += '/';
const calls = {};
let selectedCall = null;  // only ONE call is displayed; the rest live behind tabs
function selectCall(callId) {
  selectedCall = callId;
  document.querySelectorAll('.call').forEach(el =>
    el.classList.toggle('selected', el.dataset.cid === callId));
  document.querySelectorAll('.tab').forEach(el => {
    el.classList.toggle('selected', el.dataset.cid === callId);
    if (el.dataset.cid === callId) el.classList.remove('alerted');
  });
}
const players = {};       // call_id -> {audio, button}
const fadeTimers = {};    // call_id -> timeout id (clean-mode fade)
let cursor = 0;
let polling = false;

/* ---------- clean mode (persisted across reloads) ---------- */
let cleanMode = localStorage.getItem('cleanMode') === '1';
function applyClean() {
  document.body.classList.toggle('clean', cleanMode);
  cleanToggle.classList.toggle('on', cleanMode);
}
cleanToggle.onclick = () => { cleanMode = !cleanMode;
  localStorage.setItem('cleanMode', cleanMode ? '1' : '0'); applyClean(); };
applyClean();

/* ---------- transport: WebSocket first, 1s HTTP polling as automatic fallback ---------- */
function goLive(mode) { dot.classList.add('live'); status.textContent = 'live (' + mode + ')'; }
function startPolling() {
  if (polling) return; polling = true; goLive('polling');
  setInterval(async () => {
    try {
      const r = await fetch(base + 'events?since=' + cursor);
      const data = await r.json();
      data.events.forEach(render); cursor = data.next;
    } catch (e) { /* transient — retry next tick */ }
  }, 1000);
}
try {
  const wsProto = location.protocol === 'https:' ? 'wss://' : 'ws://';
  const ws = new WebSocket(wsProto + location.host + base + 'ws');
  ws.onopen    = () => goLive('websocket');
  ws.onmessage = (e) => { render(JSON.parse(e.data)); cursor++; };
  ws.onerror   = startPolling;
  ws.onclose   = () => { if (!polling) startPolling(); };
  setTimeout(() => { if (ws.readyState !== 1) startPolling(); }, 3000);
} catch (e) { startPolling(); }

/* ---------- smart scrolling: only follow if the viewer is already near the bottom ---------- */
function nearBottom() {
  return window.innerHeight + window.scrollY >= document.body.scrollHeight - 180;
}

/* ---------- audio play/pause (unsynced with analysis; for debugging by ear) ---------- */
function attachPlayer(callId, button) {
  button.onclick = () => {
    if (!players[callId]) {
      players[callId] = new Audio(base + 'audio/' + callId);     // streamed from the server
      players[callId].onended = () => { button.textContent = '\\u25B6'; };
      players[callId].onerror = () => { button.textContent = '\\u2715'; button.disabled = true; };
    }
    const a = players[callId];
    if (a.paused) { a.play(); button.textContent = '\\u23F8'; }
    else          { a.pause(); button.textContent = '\\u25B6'; }
  };
}

/* ---------- rendering ---------- */
function panel(callId) {
  if (!calls[callId]) {
    const div = document.createElement('div');
    div.className = 'call';
    div.innerHTML =
      '<h2><button class="play" title="play/pause the recording">\\u25B6</button>' +
      'CALL ' + callId + '</h2>' +
      '<div class="stage">' +
        '<div class="person rep"><span class="figure">&#129489;&#8205;&#128188;</span>' +
          '<div class="plabel">Customer Rep</div></div>' +
        '<div class="speech">waiting for audio&hellip;</div>' +
        '<div class="person customer"><span class="figure">&#128694;</span>' +
          '<div class="plabel">Customer</div></div>' +
      '</div>' +
      '<div class="turns"></div>';
    div.dataset.cid = callId;
    document.getElementById('calls').appendChild(div);
    attachPlayer(callId, div.querySelector('.play'));
    const tab = document.createElement('button');      // one tab per loaded call
    tab.className = 'tab'; tab.dataset.cid = callId;
    tab.textContent = 'CALL ' + callId.slice(0, 10) + '…';
    tab.onclick = () => selectCall(callId);
    document.getElementById('tabs').appendChild(tab);
    calls[callId] = div;
    if (!selectedCall) selectCall(callId);             // first call to arrive is shown
  }
  return calls[callId];
}
function render(ev) {
  if (ev.type === 'turn') {
    const p = panel(ev.call_id);
    /* light up the speaking figure, dim the other */
    p.querySelectorAll('.person').forEach(el => el.classList.remove('speaking'));
    p.querySelector('.person.' + ev.role).classList.add('speaking');
    /* live utterance between the figures */
    const sp = p.querySelector('.speech');
    sp.className = 'speech ' + ev.role + (ev.violations.length ? ' violation' : '');
    sp.innerHTML = '&ldquo;' + ev.text + '&rdquo;' +
      (ev.violations.length
        ? '<div class="vio">&#9888; ' + ev.violations.join(', ') + ' — ' + (ev.reason || '') + '</div>'
        : '');
    /* clean mode: fade the utterance away after a few seconds of stillness */
    clearTimeout(fadeTimers[ev.call_id]);
    if (cleanMode && !ev.violations.length) {          // violations stay visible until the next turn
      fadeTimers[ev.call_id] = setTimeout(() => {
        sp.classList.add('fading');
        p.querySelectorAll('.person').forEach(el => el.classList.remove('speaking'));
      }, 4500);
    }
    /* conversation history bubble (hidden by CSS in clean mode, still recorded) */
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
    if (!cleanMode && nearBottom())                    // follow only if the viewer was at the bottom
      t.scrollIntoView({behavior:'smooth', block:'end'});
    status.textContent = 'live — last turn t' + ev.turn_number + ' (' + ev.call_id + ')';
  } else if (ev.type === 'alert') {
    const p = panel(ev.call_id);
    p.classList.add('escalated');
    const b = document.createElement('div');
    b.className = 'banner';
    b.textContent = '\\u{1F6A8} ESCALATED at turn ' + ev.turn_number + ' — ' + ev.rule +
                    '  (judge ' + ev.judge_ms + ' ms)';
    p.insertBefore(b, p.querySelector('.stage'));
    document.title = '\\u{1F6A8} ESCALATION — AI Call Moderator';
    if (ev.call_id !== selectedCall) {                 // escalation on a hidden call -> its tab screams
      const tab = document.querySelector('.tab[data-cid="' + ev.call_id + '"]');
      if (tab) tab.classList.add('alerted');
    }
  } else if (ev.type === 'status') {
    status.textContent = ev.text;
  }
}
</script></body></html>"""


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


@app.get("/")
async def index():
    return HTMLResponse(PAGE)


# Catch-all GET, because some Jupyter proxies forward the FULL /proxy/7860/... path:
#   */events           -> polling API (since=N)
#   */audio/<call_id>  -> stream the original recording for the play button
#   anything else      -> the dashboard page
@app.get("/{full_path:path}")
async def index_any(full_path: str, since: int = 0):
    trimmed = full_path.rstrip("/")
    if trimmed.endswith("events"):
        return JSONResponse({"events": EVENT_LOG[since:], "next": len(EVENT_LOG)})
    if "audio/" in trimmed:
        call_id = trimmed.rsplit("audio/", 1)[1]
        audio_path = find_audio_file(call_id)
        if audio_path is not None:
            return FileResponse(audio_path)
        return JSONResponse({"error": f"no recording found for {call_id}"}, status_code=404)
    return HTMLResponse(PAGE)


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
    parser.add_argument("--audio-dir", default=None,
                        help="extra directory containing recordings")
    args = parser.parse_args()
    if args.audio_dir:
        AUDIO_DIRS.insert(0, pathlib.Path(args.audio_dir))
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")
