"""
Command-center web dashboard (Flask): live multi-camera MJPEG wall, alert feed, acknowledgement,
PDF incident reports. Run with ``python run.py --serve``.
"""

import os
import time
from typing import Any, Dict, List, Optional

from flask import Flask, Response, abort, jsonify, request, send_from_directory

from .alerts import AlertStore
from .pipeline import CameraWorker
from .report import build_incident_report, build_summary_report

PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>IBVAP Command Center</title><style>
:root{--bg:#0b111b;--panel:#121b2a;--line:#22304a;--txt:#dbe5f3;--mut:#7c8ca5;--ok:#2ecc71;--warn:#f39c12;--crit:#e74c3c}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--txt);font:14px/1.4 system-ui,Segoe UI,Roboto,sans-serif}
header{display:flex;align-items:center;gap:16px;padding:10px 18px;background:var(--panel);border-bottom:1px solid var(--line)}
header h1{font-size:16px;margin:0;letter-spacing:.08em}header .sp{flex:1}.pill{padding:3px 10px;border-radius:99px;font-size:12px;font-weight:600}
.SECURE{background:#12351f;color:var(--ok)}.ELEVATED{background:#40300c;color:var(--warn)}.CRITICAL{background:#47130f;color:var(--crit)}
main{display:grid;grid-template-columns:1fr 360px;gap:14px;padding:14px}@media(max-width:1000px){main{grid-template-columns:1fr}}
#cams{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:12px}
.cam{background:var(--panel);border:1px solid var(--line);border-radius:8px;overflow:hidden}.cam img{width:100%;display:block;background:#000;aspect-ratio:4/3;object-fit:contain}
.cam .bar{display:flex;gap:8px;align-items:center;padding:6px 10px;font-size:12px;color:var(--mut)}.cam .bar b{color:var(--txt)}
button{background:#1b2a41;color:var(--txt);border:1px solid var(--line);border-radius:6px;padding:3px 9px;cursor:pointer;font-size:12px}button:hover{background:#26395a}
aside{background:var(--panel);border:1px solid var(--line);border-radius:8px;display:flex;flex-direction:column;max-height:calc(100vh - 80px)}
aside h2{margin:0;padding:10px 14px;font-size:13px;letter-spacing:.06em;border-bottom:1px solid var(--line)}#stats{padding:8px 14px;color:var(--mut);font-size:12px;border-bottom:1px solid var(--line)}
#feed{overflow:auto;flex:1}.al{padding:9px 14px;border-bottom:1px solid var(--line);border-left:4px solid var(--mut)}
.al.CRITICAL{border-left-color:var(--crit)}.al.HIGH{border-left-color:var(--warn)}.al.MEDIUM{border-left-color:#f1c40f}.al.ack{opacity:.5}
.al .m{font-size:13px}.al .s{font-size:11px;color:var(--mut);margin-top:3px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}.al a{color:#6ea8ff}
.al img{width:100%;margin-top:6px;border-radius:4px;display:none}.al.open img{display:block}
</style></head><body>
<header><h1>IBVAP COMMAND CENTER</h1><span id="thr" class="pill SECURE">SECURE</span><span class="sp"></span>
<a style="color:#6ea8ff" href="/api/report/summary.pdf" target="_blank">Shift report (PDF)</a></header>
<main><section id="cams"></section><aside><h2>ALERTS</h2><div id="stats"></div><div id="feed"></div></aside></main>
<script>
const $=s=>document.querySelector(s);const esc=t=>{const d=document.createElement('div');d.textContent=t??'';return d.innerHTML};
const camEls={};
async function cams(){const r=await (await fetch('/api/cameras')).json();let worst='SECURE';const rank={SECURE:0,ELEVATED:1,CRITICAL:2};
 r.forEach(c=>{ if(rank[c.threat_level]>rank[worst])worst=c.threat_level;
  let el=camEls[c.id];
  if(!el){el=camEls[c.id]=document.createElement('div');el.className='cam';
   el.innerHTML=`<img src="/video/${encodeURIComponent(c.id)}" alt="${esc(c.id)}"><div class="bar"><b>${esc(c.id)}</b><span>${esc(c.sector)}</span><span class="sp" style="flex:1"></span><span class="meta"></span>
   <button data-m="night">Night</button><button data-m="thermal">Thermal</button></div>`;$('#cams').appendChild(el);
   el.querySelectorAll('button').forEach(b=>b.onclick=async()=>{const cur=camEls[c.id]._c;await fetch(`/api/cameras/${encodeURIComponent(c.id)}/mode`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({[b.dataset.m]:!cur[b.dataset.m+'_mode']})})})}
  el._c=c;el.querySelector('.meta').innerHTML=`<span class="pill ${c.threat_level}">${c.threat_level}</span> ${c.fps} fps · ${c.persons} ppl · ${esc(c.backend)}${c.status!=='running'?' · '+esc(c.status):''}`});
 const t=$('#thr');t.textContent=worst;t.className='pill '+worst}
async function feed(){const [a,s]=await Promise.all([fetch('/api/alerts?limit=60&min_severity=MEDIUM').then(r=>r.json()),fetch('/api/stats').then(r=>r.json())]);
 $('#stats').textContent=`Total ${s.total} · `+Object.entries(s.by_severity).map(([k,v])=>k+' '+v).join(' · ');
 const open=new Set([...document.querySelectorAll('.al.open')].map(e=>e.dataset.id));
 $('#feed').innerHTML=a.map(x=>`<div class="al ${x.severity} ${x.acknowledged?'ack':''} ${open.has(x.id)?'open':''}" data-id="${x.id}">
  <div class="m">${esc(x.message)}</div><div class="s"><span>${new Date(x.created*1000).toLocaleTimeString()}</span><span>${esc(x.camera_id)}</span><span>${esc(x.type)}</span>
  ${x.snapshot?`<a href="#" data-t="1">snapshot</a>`:''}<a href="/api/alerts/${x.id}/report.pdf" target="_blank">report</a>${x.acknowledged?'<span>acknowledged</span>':`<a href="#" data-ack="1">acknowledge</a>`}</div>
  ${x.snapshot?`<img loading="lazy" src="/snapshots/${esc(x.snapshot)}">`:''}</div>`).join('')}
$('#feed').addEventListener('click',async e=>{const al=e.target.closest('.al');if(!al)return;
 if(e.target.dataset.t){e.preventDefault();al.classList.toggle('open')}
 if(e.target.dataset.ack){e.preventDefault();await fetch(`/api/alerts/${al.dataset.id}/ack`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({note:prompt('Operator note (optional)')||''})});feed()}});
cams();feed();setInterval(cams,2000);setInterval(feed,2000);
</script></body></html>"""


def create_app(cfg: Dict[str, Any], workers: List[CameraWorker], store: AlertStore) -> Flask:
    app = Flask(__name__)
    by_id = {w.camera_id: w for w in workers}
    snap_dir = os.path.abspath(cfg["alerts"]["snapshot_dir"])
    rep_dir = cfg["alerts"]["report_dir"]

    @app.get("/")
    def index():
        return PAGE

    @app.get("/api/cameras")
    def cameras():
        return jsonify([w.summary() for w in workers])

    @app.post("/api/cameras/<cam>/mode")
    def mode(cam):
        w = by_id.get(cam) or abort(404)
        body = request.get_json(silent=True) or {}
        if "night" in body:
            w.night_mode = bool(body["night"])
        if "thermal" in body:
            w.thermal_mode = bool(body["thermal"])
        return jsonify(w.summary())

    @app.get("/video/<cam>")
    def video(cam):
        w = by_id.get(cam) or abort(404)

        def gen():
            seq = -1
            while True:
                jpg, seq = w.wait_frame(seq)
                if jpg:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n"
        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.get("/api/alerts")
    def alerts():
        return jsonify(store.recent(min(int(request.args.get("limit", 50)), 500), request.args.get("camera"),
                                    request.args.get("min_severity")))

    @app.get("/api/stats")
    def stats():
        return jsonify(store.stats())

    @app.post("/api/alerts/<aid>/ack")
    def ack(aid):
        note = (request.get_json(silent=True) or {}).get("note", "")
        return jsonify({"ok": store.acknowledge(aid, note)})

    @app.get("/api/alerts/<aid>/report.pdf")
    def report(aid):
        a = store.get(aid) or abort(404)
        path = build_incident_report(a, os.path.join(rep_dir, f"incident_{aid}.pdf"), snap_dir)
        return send_from_directory(os.path.abspath(os.path.dirname(path)), os.path.basename(path), mimetype="application/pdf")

    @app.get("/api/report/summary.pdf")
    def summary():
        path = build_summary_report(store.recent(100), store.stats(), os.path.join(rep_dir, "summary.pdf"))
        return send_from_directory(os.path.abspath(os.path.dirname(path)), os.path.basename(path), mimetype="application/pdf")

    @app.get("/snapshots/<path:name>")
    def snapshot(name):
        return send_from_directory(snap_dir, name)

    return app


def serve(cfg: Dict[str, Any], host: Optional[str] = None, port: Optional[int] = None) -> int:
    store = AlertStore(cfg["alerts"])
    workers = [CameraWorker(c, cfg, store) for c in cfg["cameras"]]
    for w in workers:
        w.start()
    app = create_app(cfg, workers, store)
    host, port = host or cfg["server"]["host"], port or cfg["server"]["port"]
    print(f"[IBVAP] Dashboard on http://{host}:{port}  ({len(workers)} camera(s))")
    try:
        app.run(host=host, port=port, threaded=True, debug=False)
    finally:
        for w in workers:
            w.stop()
    return 0
