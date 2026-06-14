#!/usr/bin/env python3
"""Live dashboard for the AI Call Moderator (v7).

New in v7:
  - Supervisor OVERRIDE: escalated calls show an ⚡ OVERRIDE button.  Clicking it
    reveals a list of every escalated segment (turn number, rule, timestamp in the
    recording).  Clicking any item seeks the audio player to that exact moment so the
    supervisor can hear the context before joining.  After override the global header
    shows a 3-person icon to indicate the call now has three parties.
  - Skipped-segment panel: every chunk the pipeline silently dropped (silence, censor
    beep, low-confidence ASR, repetition loop) is shown in a collapsible panel at the
    bottom of each call.  Clicking a row seeks the audio to that timestamp so the
    recording can be validated by ear — confirming whether the skip was correct.
  - audio_start_s on every turn + alert event so both panels know where in the
    recording to seek to.
  - X button on each call tab to dismiss it from the UI and remove it from the event
    log so it does not reappear on refresh or reconnect.
  - WebSocket ?since=N param: on reconnect only replay missing events, no duplicate
    turns.  Auto-reconnect loop keeps the live feed going after a brief blip.
  - EVENT_FILE stored in the working directory (not /tmp) so the log survives server
    restarts and kernel relaunches.

Unchanged from v5/v6:
  - WebSocket first, automatic 1s HTTP polling fallback
  - Full event-log replay for late joiners and GUI restarts
  - Per-call tab bar, stage figures, conversation bubbles, clean mode
"""
import argparse
import json
import pathlib

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

app = FastAPI()

# Stored in CWD (next to the notebook) so the log survives /tmp clears and
# kernel / server restarts.  On startup the file is replayed into EVENT_LOG so
# a browser refresh or reopen always gets the full session history.
EVENT_FILE = pathlib.Path("call_moderator_events.jsonl")
EVENT_LOG  = []
if EVENT_FILE.exists():
    for line in EVENT_FILE.read_text().splitlines():
        try:
            EVENT_LOG.append(json.loads(line))
        except json.JSONDecodeError:
            pass
CONNECTED  = set()
AUDIO_DIRS = [pathlib.Path("call_recordings"), pathlib.Path("scam_call"), pathlib.Path("kaggle_call_data")]
AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".m4a", ".ogg")


def find_audio_file(call_id: str):
    for directory in AUDIO_DIRS:
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            if path.stem == call_id and path.suffix.lower() in AUDIO_EXTENSIONS:
                return path
    return None


PAGE = r"""<!DOCTYPE html>
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

  #supervising { display:none; align-items:center; gap:8px;
                 background:rgba(35,134,54,.18); border:1px solid #238636;
                 border-radius:20px; padding:5px 18px;
                 font-size:13px; font-weight:700; color:#56d364;
                 animation:fadeIn .4s ease; }
  #supervising.active { display:flex; }
  @keyframes fadeIn { from{opacity:0;transform:scale(.9)} to{opacity:1;transform:scale(1)} }

  #tabs { display:flex; gap:8px; padding:12px 18px 0; flex-wrap:wrap; }
  .tab { background:var(--panel); border:1px solid var(--line); color:var(--dim); cursor:pointer;
         border-radius:8px; padding:6px 12px; font-size:12px; font-weight:600;
         display:flex; align-items:center; gap:6px; }
  .tab:hover { border-color:var(--dim); }
  .tab.selected { border-color:var(--repglow); color:var(--repglow); }
  .tab.alerted  { border-color:var(--bad); color:#ffa198; animation:flash 1s linear 6; }
  .tab-x { font-size:11px; line-height:1; opacity:.45; border:none; background:none;
            color:inherit; cursor:pointer; padding:0 2px; border-radius:3px; }
  .tab-x:hover { opacity:1; color:var(--bad); }

  #calls { padding:18px; display:grid; gap:18px; }
  .call { display:none; }
  .call.selected { display:block; }
  .call { background:var(--panel); border:1px solid var(--line); border-radius:10px; }
  .call.escalated { border-color:var(--bad); box-shadow:0 0 0 1px var(--bad); }
  .call h2 { margin:0; padding:10px 16px; font-size:13px; color:var(--dim);
             border-bottom:1px solid var(--line); font-weight:600;
             display:flex; align-items:center; gap:10px; flex-wrap:wrap; }
  .call h2 audio { height:30px; margin-left:auto; flex:1; max-width:480px;
                   filter:invert(.88) hue-rotate(180deg); border-radius:6px; }

  .override-btn { background:var(--warn); color:#000; border:none; border-radius:8px;
                  padding:5px 14px; font-size:12.5px; cursor:pointer; font-weight:700;
                  white-space:nowrap; transition:background .2s; }
  .override-btn:hover { filter:brightness(1.12); }
  .override-btn.supervising { background:#238636; color:#fff; cursor:default; }

  .esc-panel { display:none; background:rgba(218,54,51,.07);
               border-bottom:1px solid rgba(218,54,51,.3); padding:10px 16px; }
  .esc-panel.open { display:block; }
  .esc-panel-title { font-size:11px; font-weight:700; letter-spacing:.8px;
                     color:#ffa198; text-transform:uppercase; margin-bottom:8px; }
  .esc-item { display:flex; align-items:center; gap:12px; padding:7px 10px;
              border-radius:8px; cursor:pointer; font-size:13px;
              border:1px solid transparent; }
  .esc-item:hover { background:rgba(255,255,255,.06); border-color:rgba(218,54,51,.4); }
  .esc-item .esc-ts   { font-family:monospace; color:var(--warn); font-weight:700;
                        min-width:52px; }
  .esc-item .esc-turn { color:var(--dim); font-size:11px; min-width:30px; }
  .esc-item .esc-rule { color:#ffa198; flex:1; font-size:12.5px; }
  .esc-item .esc-seek { color:var(--dim); font-size:11px; }
  .esc-item:hover .esc-seek { color:var(--repglow); }
  .esc-expand { font-size:11px; color:var(--dim); cursor:pointer;
                padding:1px 7px; border:1px solid var(--line);
                border-radius:6px; background:none; white-space:nowrap; }
  .esc-expand:hover { color:var(--text); border-color:var(--dim); }
  .esc-text-block { display:none; padding:6px 10px 2px;
                    font-size:12px; color:var(--text); line-height:1.5;
                    border-left:2px solid var(--bad); margin:4px 0 2px 4px; }
  .esc-text-block.open { display:block; }
  body.clean .esc-panel.escalated-open { display:block; }
  .esc-join-btn { display:block; margin:10px 12px 6px;
    background:#238636; color:#fff; border:none; border-radius:8px;
    padding:8px 18px; font-size:13px; font-weight:700; cursor:pointer;
    letter-spacing:.4px; }
  .esc-join-btn:hover { filter:brightness(1.15); }
  .esc-join-btn:disabled { background:#30363d; color:var(--dim); cursor:default; }

  .banner { background:var(--bad); color:#fff; padding:10px 16px; font-weight:700;
            font-size:13px; animation:flash 1s linear 6; }
  @keyframes flash { 50% { filter:brightness(1.6);} }

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
            transition:all .25s ease; }
  .speech.rep      { border-color:var(--rep);  color:var(--text); background:rgba(31,111,235,.10); }
  .speech.customer { border-color:var(--cust); color:var(--text); background:rgba(35,134,54,.10); }
  .speech.violation { border-color:var(--bad); background:rgba(218,54,51,.14);
                      box-shadow:0 0 12px rgba(218,54,51,.35); }
  .speech .vio { margin-top:6px; font-size:12px; color:#ffa198; font-weight:600; }
  .speech.fading { opacity:0; transition:opacity 2s ease; }

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

  .lf-btn { margin-left:10px; padding:3px 10px; font-size:11px; font-weight:600;
    background:none; border:1px solid var(--line); border-radius:4px;
    color:var(--dim); cursor:pointer; vertical-align:middle; }
  .lf-btn:hover { color:var(--text); border-color:var(--dim); }
  .lf-btn.loading { opacity:.5; pointer-events:none; }
  .lf-panel { display:none; border-top:1px solid var(--line);
    padding:12px 16px; font-size:12px; }
  .lf-panel.open { display:block; }
  .lf-section { margin-bottom:14px; }
  .lf-section-title { font-size:10px; font-weight:700; letter-spacing:.8px;
    color:var(--dim); text-transform:uppercase; margin-bottom:6px; }
  .lf-kv { display:grid; grid-template-columns:180px 1fr;
    gap:3px 12px; color:var(--text); }
  .lf-kv .k { color:var(--dim); }
  .lf-table { width:100%; border-collapse:collapse; font-size:11px; }
  .lf-table th { text-align:left; color:var(--dim); font-weight:600;
    padding:3px 8px; border-bottom:1px solid var(--line); }
  .lf-table td { padding:3px 8px; border-bottom:1px solid rgba(255,255,255,.04); }
  .lf-table tr:hover td { background:rgba(255,255,255,.03); }
  .lf-verdict { font-family:monospace; font-size:10px; color:var(--dim);
    white-space:nowrap; overflow:hidden; text-overflow:ellipsis; max-width:220px; }
  .lf-no-data { color:var(--dim); font-size:12px; padding:8px 0; }

  .skip-panel { border-top:1px solid var(--line); }
  .skip-header { padding:9px 16px; font-size:12px; color:var(--dim); cursor:pointer;
                 display:flex; align-items:center; gap:8px; user-select:none;
                 transition:color .15s; }
  .skip-header:hover { color:var(--text); }
  .skip-caret { transition:transform .2s; display:inline-block; font-size:10px; }
  .skip-header.open .skip-caret { transform:rotate(90deg); }
  .skip-count-badge { border-radius:10px; padding:2px 9px; font-size:11px; font-weight:700;
                      background:rgba(110,118,129,.2); color:var(--dim); }
  .skip-count-badge.has-items { background:rgba(210,153,34,.2); color:var(--warn); }
  .skip-body { padding:0 12px 12px; display:none; }
  .skip-body.open { display:block; }
  .skip-item { display:flex; align-items:center; gap:10px; padding:6px 8px;
               border-radius:8px; cursor:pointer; font-size:12.5px; color:var(--dim);
               border:1px solid transparent; }
  .skip-item:hover { background:rgba(255,255,255,.05); border-color:var(--line); color:var(--text); }
  .skip-item .skip-ts { font-family:monospace; color:var(--repglow); min-width:52px;
                         font-weight:600; font-size:13px; }
  .skip-tag { padding:2px 8px; border-radius:8px; font-size:11px; font-weight:700;
              min-width:96px; text-align:center; }
  .skip-tag.silence        { background:rgba(110,118,129,.3);  color:var(--dim); }
  .skip-tag.beep           { background:rgba(210,153,34,.25);  color:var(--warn); }
  .skip-tag.no_speech      { background:rgba(218,54,51,.2);    color:#ffa198; }
  .skip-tag.low_confidence { background:rgba(218,54,51,.2);    color:#ffa198; }
  .skip-tag.repetition     { background:rgba(210,153,34,.2);   color:var(--warn); }
  .skip-item .skip-detail { color:var(--dim); font-size:11px; flex:1; }
  .skip-item .skip-seek   { color:var(--dim); font-size:11px; }
  .skip-item:hover .skip-seek { color:var(--repglow); }
</style></head><body>
<header>
  <h1>&#128737; AI CALL MODERATOR &mdash; LIVE</h1>
  <span class="dot" id="dot"></span>
  <span id="status">connecting&hellip;</span>
  <span id="supervising" title="Supervisor has joined the call">&#128101;&#128101; SUPERVISING</span>
  <label class="hbtn sync-label" title="Seek audio to match each incoming turn timestamp">
    <input type="checkbox" id="syncCheck" style="margin-right:5px">Sync audio
  </label>
  <button class="hbtn" id="cleanToggle">Simple</button>
</header>
<div id="tabs"></div>
<div id="calls"></div>
<script>
let base = location.pathname; if (!base.endsWith('/')) base += '/';

const calls    = {};
const callData = {};
let selectedCall = null, cursor = 0, polling = false;

/* ---------- audio sync ---------- */
let audioSync = false;
const syncCheck = document.getElementById('syncCheck');
syncCheck.onchange = () => { audioSync = syncCheck.checked; };

/* ---------- clean mode ---------- */
let cleanMode = localStorage.getItem('cleanMode') === '1';
function applyClean() {
  document.body.classList.toggle('clean', cleanMode);
  cleanToggle.classList.toggle('on', cleanMode);
}
cleanToggle.onclick = () => {
  cleanMode = !cleanMode;
  localStorage.setItem('cleanMode', cleanMode ? '1' : '0');
  applyClean();
};
applyClean();

/* ---------- tabs ---------- */
function selectCall(callId) {
  selectedCall = callId;
  document.querySelectorAll('.call').forEach(el =>
    el.classList.toggle('selected', el.dataset.cid === callId));
  document.querySelectorAll('.tab').forEach(el => {
    el.classList.toggle('selected', el.dataset.cid === callId);
    if (el.dataset.cid === callId) el.classList.remove('alerted');
  });
}

/* ---------- dismiss call (X button) ----------
   POSTs to the server which removes all events for this call from EVENT_LOG
   and rewrites the event file — the call will not reappear on refresh or
   reconnect.  DOM is cleaned up immediately on the client side.              */
async function dismissCall(callId, event) {
  event.stopPropagation();
  if (!confirm('Remove call "' + callId + '" from the dashboard?\nThis clears it from the session log and it will not reappear on refresh.')) return;
  try { await fetch(base + 'dismiss/' + callId, { method: 'POST' }); } catch(e) {}
  const tab = document.querySelector('.tab[data-cid="' + callId + '"]');
  if (tab) tab.remove();
  const p = calls[callId];
  if (p) p.remove();
  delete calls[callId];
  delete callData[callId];
  if (selectedCall === callId) {
    selectedCall = null;
    const firstTab = document.querySelector('.tab');
    if (firstTab) selectCall(firstTab.dataset.cid);
  }
}

/* ---------- transport: WebSocket with auto-reconnect → 1s HTTP polling fallback
   ?since=N tells the server to replay only events the client hasn't seen yet.
   On a fresh page load cursor=0 so the full session history is replayed and the
   chat is restored.  On reconnect cursor=N so only missed events are delivered —
   no duplicate turns appended to the conversation.                              */
function goLive(m) { dot.classList.add('live'); status.textContent = 'live (' + m + ')'; }

function startPolling() {
  if (polling) return; polling = true; goLive('polling');
  setInterval(async () => {
    try {
      const r = await fetch(base + 'events?since=' + cursor);
      const d = await r.json();
      d.events.forEach(render); cursor = d.next;
    } catch (e) {}
  }, 1000);
}

function connectWS() {
  try {
    const wsProto = location.protocol === 'https:' ? 'wss://' : 'ws://';
    const ws = new WebSocket(wsProto + location.host + base + 'ws?since=' + cursor);
    let opened = false;
    ws.onopen    = () => { opened = true; goLive('websocket'); };
    ws.onmessage = (e) => { render(JSON.parse(e.data)); cursor++; };
    ws.onerror   = () => { if (!opened) startPolling(); };
    ws.onclose   = () => { setTimeout(connectWS, 2000); };
    /* if WS hasn't opened within 3s start polling as a fallback */
    setTimeout(() => { if (ws.readyState !== 1) startPolling(); }, 3000);
  } catch (e) { startPolling(); setTimeout(connectWS, 3000); }
}
connectWS();

function nearBottom() {
  return window.innerHeight + window.scrollY >= document.body.scrollHeight - 180;
}

/* ---------- audio seek ---------- */
function fmtTime(s) {
  if (s == null || !isFinite(s)) return '--:--';
  return Math.floor(s / 60) + ':' + String(Math.floor(s % 60)).padStart(2, '0');
}
function seekTo(callId, seconds) {
  const cd = callData[callId];
  if (!cd || !cd.audioEl || !isFinite(+seconds)) return;
  cd.audioEl.currentTime = +seconds;
}

/* ---------- override / join ---------- */
function handleOverride(callId) {
  const p     = calls[callId];
  const btn   = p.querySelector('.override-btn');
  const panel = p.querySelector('.esc-panel');
  const isOpen = panel.classList.toggle('open');
  if (!callData[callId].overridden)
    btn.textContent = isOpen ? '🔍 REVIEWING...' : '⚡ OVERRIDE';
}

function joinCall(callId) {
  const cd = callData[callId];
  if (cd.overridden) return;
  cd.overridden = true;
  const p   = calls[callId];
  const btn = p.querySelector('.override-btn');
  btn.classList.add('supervising');
  btn.textContent = '✅ SUPERVISING';
  const joinBtn = p.querySelector('.esc-join-btn');
  if (joinBtn) { joinBtn.disabled = true; joinBtn.textContent = '✅ Joined'; }
  document.getElementById('supervising').classList.add('active');
}

/* ---------- escalation list ---------- */
function refreshEscList(callId) {
  const items = calls[callId].querySelector('.esc-items');
  items.innerHTML = '';
  callData[callId].escalations.forEach((esc, idx) => {
    const item = document.createElement('div');
    item.className = 'esc-item';
    const textId = 'esc-txt-' + callId + '-' + idx;
    item.innerHTML =
      '<span class="esc-ts">'  + fmtTime(esc.audio_start_s) + '</span>' +
      '<span class="esc-turn">t' + esc.turn_number + '</span>' +
      '<span class="esc-rule">' + (esc.rule || '') + '</span>' +
      '<button class="esc-expand" onclick="' +
        'var b=document.getElementById(\'' + textId + '\');' +
        'b.classList.toggle(\'open\');' +
        'this.textContent=b.classList.contains(\'open\')?\'+\' hide\':\'+ show\';' +
        'event.stopPropagation();">+ show</button>' +
      '<span class="esc-seek" onclick="seekTo(\'' + callId + '\',' + esc.audio_start_s + ');event.stopPropagation()">▶ seek</span>';
    const textBlock = document.createElement('div');
    textBlock.className = 'esc-text-block';
    textBlock.id = textId;
    textBlock.textContent = esc.detail || '';
    const wrap = document.createElement('div');
    wrap.style.borderBottom = '1px solid var(--line)';
    wrap.style.padding = '2px 0';
    wrap.appendChild(item);
    wrap.appendChild(textBlock);
    items.appendChild(wrap);
  });
}

/* ---------- skipped-segment panel ---------- */
function refreshSkipPanel(callId) {
  const p     = calls[callId];
  const cd    = callData[callId];
  const badge = p.querySelector('.skip-count-badge');
  const body  = p.querySelector('.skip-body');
  const n     = cd.skips.length;
  badge.textContent = n + ' skipped segment' + (n !== 1 ? 's' : '');
  badge.classList.toggle('has-items', n > 0);
  if (!body.classList.contains('open')) return;
  body.innerHTML = '';
  cd.skips.forEach(sk => {
    const item = document.createElement('div');
    item.className = 'skip-item';
    item.title = 'Seek to ' + fmtTime(sk.audio_start_s) + ' and listen to validate this skip';
    const rc = (sk.reason || 'silence').replace(/[^a-z_]/g, '');
    const SKIP_LABELS = {silence:'silence',beep:'🔇 PII/Sensitive',
      no_speech:'no speech',low_confidence:'low confidence',repetition:'repetition'};
    item.innerHTML =
      '<span class="skip-ts">' + fmtTime(sk.audio_start_s) + '</span>' +
      '<span class="skip-tag ' + rc + '">' + (SKIP_LABELS[rc] || rc) + '</span>' +
      '<span class="skip-detail">' + (sk.detail || '') + '</span>' +
      '<span class="skip-seek">▶ seek</span>';
    item.onclick = () => seekTo(callId, sk.audio_start_s);
    body.appendChild(item);
  });
}

const fadeTimers = {};

/* ---------- langfuse info panel ---------- */
async function toggleLangfuse(callId) {
  const callDiv = calls[callId];
  if (!callDiv) return;
  const panel = callDiv.querySelector('.lf-panel');
  const btn   = callDiv.querySelector('.lf-btn');
  if (panel.classList.contains('open')) {
    panel.classList.remove('open'); btn.textContent = 'info'; return;
  }
  btn.classList.add('loading'); btn.textContent = 'loading...';
  try {
    const r   = await fetch(base + 'langfuse/' + callId);
    const dat = await r.json();
    renderLangfuse(panel, dat.records || []);
  } catch(e) {
    panel.innerHTML = '<div class="lf-no-data">Could not load Langfuse data: ' + e + '</div>';
  }
  btn.classList.remove('loading'); btn.textContent = 'info';
  panel.classList.add('open');
}

function renderLangfuse(panel, records) {
  if (!records.length) {
    panel.innerHTML = '<div class="lf-no-data">No Langfuse data yet — run a call first.</div>';
    return;
  }
  const totalIn  = records.reduce((s,r) => s + (r.usage?.input  || 0), 0);
  const totalOut = records.reduce((s,r) => s + (r.usage?.output || 0), 0);
  const totalMs  = records.reduce((s,r) => s + (r.elapsed_ms || 0), 0);
  const avgMs    = records.length ? Math.round(totalMs / records.length) : 0;
  const stages = {};
  records.forEach(r => {
    const s = stages[r.stage] || {count:0, ms:0, input:0, output:0};
    s.count++; s.ms += r.elapsed_ms||0;
    s.input += r.usage?.input||0; s.output += r.usage?.output||0;
    stages[r.stage] = s;
  });
  let stageRows = '';
  Object.entries(stages).forEach(([stage, s]) => {
    stageRows += '<tr><td>' + stage + '</td><td>' + s.count + '</td>' +
      '<td>' + Math.round(s.ms/s.count) + ' ms avg</td>' +
      '<td>' + s.input + '</td><td>' + s.output + '</td></tr>';
  });
  let genRows = '';
  records.forEach((r, i) => {
    const verdict = r.output ? JSON.stringify(r.output).slice(0,80) : '-';
    genRows += '<tr><td>' + (i+1) + '</td><td>' + r.stage + '</td>' +
      '<td>' + (r.elapsed_ms||0) + ' ms</td>' +
      '<td>' + (r.usage?.input||0) + '</td><td>' + (r.usage?.output||0) + '</td>' +
      '<td class="lf-verdict" title="' + verdict.replace(/"/g,'&quot;') + '">' + verdict + '</td></tr>';
  });
  panel.innerHTML =
    '<div class="lf-section"><div class="lf-section-title">Token Usage</div>' +
      '<div class="lf-kv">' +
        '<span class="k">Input tokens</span><span>' + totalIn + '</span>' +
        '<span class="k">Output tokens</span><span>' + totalOut + '</span>' +
        '<span class="k">Total tokens</span><span>' + (totalIn+totalOut) + '</span>' +
        '<span class="k">LLM calls</span><span>' + records.length + '</span>' +
        '<span class="k">Avg judge latency</span><span>' + avgMs + ' ms</span>' +
      '</div></div>' +
    '<div class="lf-section"><div class="lf-section-title">By Stage</div>' +
      '<table class="lf-table"><thead><tr>' +
        '<th>Stage</th><th>Calls</th><th>Latency</th><th>In tokens</th><th>Out tokens</th>' +
      '</tr></thead><tbody>' + stageRows + '</tbody></table></div>' +
    '<div class="lf-section"><div class="lf-section-title">Generations</div>' +
      '<table class="lf-table"><thead><tr>' +
        '<th>#</th><th>Stage</th><th>Latency</th><th>In</th><th>Out</th><th>Verdict</th>' +
      '</tr></thead><tbody>' + genRows + '</tbody></table></div>';
}

/* ---------- panel factory ---------- */
function panel(callId) {
  if (!calls[callId]) {
    const div = document.createElement('div');
    div.className = 'call';
    div.dataset.cid = callId;
    div.innerHTML =
      '<h2>CALL ' + callId +
        '<button class="lf-btn" onclick="toggleLangfuse(\''+callId+'\')">info</button>' +
        '<button class="override-btn" style="display:none"' +
          ' onclick="handleOverride(\'' + callId + '\')">⚡ OVERRIDE</button>' +
        '<audio controls preload="metadata" src="' + base + 'audio/' + callId + '"' +
          ' title="play/pause/seek — unsynced with live analysis"></audio>' +
      '</h2>' +
      '<div class="esc-panel">' +
        '<div class="esc-panel-title">⚠️ Escalated segments — click any to seek &amp; review audio before joining</div>' +
        '<div class="esc-items"></div>' +
        '<button class="esc-join-btn" onclick="joinCall(\'' + callId + '\')">✅ JOIN CALL — take over</button>' +
      '</div>' +
      '<div class="stage">' +
        '<div class="person rep"><span class="figure">🧑‍💼</span>' +
          '<div class="plabel">Customer Rep</div></div>' +
        '<div class="speech">waiting for audio&hellip;</div>' +
        '<div class="person customer"><span class="figure">🚶</span>' +
          '<div class="plabel">Customer</div></div>' +
      '</div>' +
      '<div class="turns"></div>' +
      '<div class="lf-panel"><div class="lf-no-data">Run a call to see Langfuse data.</div></div>' +
      '<div class="skip-panel">' +
        '<div class="skip-header" onclick="' +
            'this.classList.toggle(\'open\');' +
            'var b=this.nextElementSibling;b.classList.toggle(\'open\');' +
            'if(b.classList.contains(\'open\')){var cid=this.closest(\'.call\').dataset.cid;refreshSkipPanel(cid);}">' +
          '<span class="skip-caret">▶</span>' +
          '<span class="skip-count-badge">0 skipped segments</span>' +
          '<span style="color:var(--dim);font-size:11px;margin-left:4px">' +
            '&mdash; expand to validate skipped audio</span>' +
        '</div>' +
        '<div class="skip-body"></div>' +
      '</div>';

    document.getElementById('calls').appendChild(div);

    /* Tab with X dismiss button */
    const tab = document.createElement('button');
    tab.className = 'tab'; tab.dataset.cid = callId;
    const label = callId.length > 16 ? callId.slice(-16) : callId;
    tab.innerHTML =
      '<span>' + label + '</span>' +
      '<span class="tab-x" title="Remove this call from dashboard"' +
        ' onclick="dismissCall(\'' + callId + '\', event)">&#10005;</span>';
    tab.onclick = (e) => { if (!e.target.classList.contains('tab-x')) selectCall(callId); };
    document.getElementById('tabs').appendChild(tab);

    calls[callId]    = div;
    callData[callId] = { audioEl: div.querySelector('audio'),
                         escalations: [], skips: [], overridden: false };
    if (!selectedCall) selectCall(callId);
  }
  return calls[callId];
}

/* ---------- render ---------- */
function render(ev) {
  if (ev.type === 'turn') {
    const p = panel(ev.call_id);
    p.querySelectorAll('.person').forEach(el => el.classList.remove('speaking'));
    p.querySelector('.person.' + ev.role).classList.add('speaking');
    const sp = p.querySelector('.speech');
    sp.className = 'speech ' + ev.role + (ev.violations.length ? ' violation' : '');
    sp.innerHTML = '"' + ev.text + '"' +
      (ev.violations.length
        ? '<div class="vio">⚠ ' + ev.violations.join(', ') + ' — ' + (ev.reason||'') + '</div>'
        : '');
    clearTimeout(fadeTimers[ev.call_id]);
    if (cleanMode && !ev.violations.length) {
      fadeTimers[ev.call_id] = setTimeout(() => {
        sp.classList.add('fading');
        p.querySelectorAll('.person').forEach(el => el.classList.remove('speaking'));
      }, 4500);
    }
    const t = document.createElement('div');
    t.className = 'turn ' + ev.role + (ev.violations.length ? ' violation' : '');
    const who = ev.role === 'rep' ? 'Customer Rep' : 'Customer';
    let chips = '<span class="chip sent">sent ' + (ev.sentiment >= 0 ? '+' : '') + ev.sentiment + '</span>' +
                '<span class="chip time">asr ' + ev.asr_ms + 'ms &middot; queue ' + ev.queue_ms +
                'ms &middot; judge ' + ev.judge_ms + 'ms</span>';
    ev.violations.forEach(v => chips = '<span class="chip code">' + v + '</span>' + chips);
    if (ev.violations.length && ev.reason)
      chips += '<span class="chip reason">' + ev.reason + '</span>';
    t.innerHTML = '<div class="who">' + who + ' · t' + ev.turn_number + '</div>' +
                  ev.text + '<div class="chips">' + chips + '</div>';
    p.querySelector('.turns').appendChild(t);
    if (!cleanMode && nearBottom()) t.scrollIntoView({behavior:'smooth', block:'end'});
    status.textContent = 'live — last turn t' + ev.turn_number + ' (' + ev.call_id + ')';
    if (audioSync && ev.call_id === selectedCall && ev.audio_start_s !== undefined)
      seekTo(ev.call_id, ev.audio_start_s);

  } else if (ev.type === 'alert') {
    const p = panel(ev.call_id);
    p.classList.add('escalated');
    const b = document.createElement('div');
    b.className = 'banner';
    b.textContent = '🚨 ESCALATED at turn ' + ev.turn_number + ' — ' + ev.rule +
                    '  (judge ' + ev.judge_ms + ' ms)';
    p.insertBefore(b, p.querySelector('.esc-panel'));
    p.querySelector('.override-btn').style.display = '';
    callData[ev.call_id].escalations.push(ev);
    refreshEscList(ev.call_id);
    if (cleanMode) {
      const escP = p.querySelector('.esc-panel');
      escP.classList.add('open', 'escalated-open');
    }
    document.title = '🚨 ESCALATION — AI Call Moderator';
    if (ev.call_id !== selectedCall) {
      const tab = document.querySelector('.tab[data-cid="' + ev.call_id + '"]');
      if (tab) tab.classList.add('alerted');
    }

  } else if (ev.type === 'call_end') {
    const tab = document.querySelector('.tab[data-cid="' + ev.call_id + '"]');
    if (tab) { tab.style.opacity = '0.55'; tab.title = 'Call finished'; }
    const p = panel(ev.call_id);
    const st = p.querySelector('.call-status');
    if (st) st.textContent = 'done';

  } else if (ev.type === 'skip') {
    panel(ev.call_id);
    callData[ev.call_id].skips.push(ev);
    refreshSkipPanel(ev.call_id);

  } else if (ev.type === 'status') {
    status.textContent = ev.text;
  }
}
</script></body></html>"""


@app.post("/event")
async def receive_event(event: dict):
    EVENT_LOG.append(event)
    with EVENT_FILE.open("a") as f:
        f.write(json.dumps(event) + "\n")
    dead = []
    for ws in CONNECTED:
        try:
            await ws.send_text(json.dumps(event))
        except Exception:
            dead.append(ws)
    for ws in dead:
        CONNECTED.discard(ws)
    return {"ok": True, "clients": len(CONNECTED)}


@app.post("/dismiss/{call_id}")
async def dismiss_call(call_id: str):
    """Remove all events for a call from the in-memory log and rewrite the event
    file so the call does not reappear on refresh or server restart."""
    global EVENT_LOG
    EVENT_LOG = [e for e in EVENT_LOG if e.get("call_id") != call_id]
    if EVENT_LOG:
        EVENT_FILE.write_text("\n".join(json.dumps(e) for e in EVENT_LOG) + "\n")
    else:
        EVENT_FILE.write_text("")
    return {"ok": True, "call_id": call_id, "remaining_events": len(EVENT_LOG)}


@app.get("/langfuse/{call_id}")
async def langfuse_data(call_id: str):
    """Return cached Langfuse generation records for a call."""
    try:
        import langfuse_config as _lfc
        records = _lfc.get_call_traces(call_id)
    except Exception:
        records = []
    return JSONResponse({"call_id": call_id, "records": records})


@app.get("/")
async def index():
    return HTMLResponse(PAGE)


@app.get("/{full_path:path}")
async def index_any(full_path: str, since: int = 0):
    trimmed = full_path.rstrip("/")
    if trimmed.endswith("events"):
        return JSONResponse({"events": EVENT_LOG[since:], "next": len(EVENT_LOG)})
    if "audio/" in trimmed:
        call_id    = trimmed.rsplit("audio/", 1)[1]
        audio_path = find_audio_file(call_id)
        if audio_path is not None:
            return FileResponse(audio_path)
        return JSONResponse({"error": f"no recording found for {call_id}"}, status_code=404)
    return HTMLResponse(PAGE)


@app.websocket("/ws")
@app.websocket("/{prefix:path}/ws")
async def websocket_endpoint(ws: WebSocket):
    """Accept a WebSocket connection.

    ?since=N replays only events from index N onward — lets reconnecting clients
    catch up without receiving duplicates.  Fresh page loads pass since=0 (default)
    and get the full session history replayed instantly.
    """
    since = int(ws.query_params.get("since", 0))
    await ws.accept()
    for event in EVENT_LOG[since:]:
        await ws.send_text(json.dumps(event))
    CONNECTED.add(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        CONNECTED.discard(ws)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--audio-dir", default="kaggle_call_data")
    args = parser.parse_args()
    AUDIO_DIRS[2] = pathlib.Path(args.audio_dir)
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")
