# -*- coding: utf-8 -*-
"""Tiny web page to listen to the pipeline's TTS: pick vi / en / zh / ko, type a sentence, hear it. Standard library only (no Flask/FastAPI).
Runs the same TTS front ends + graphs as the pipeline, here on the laptop CPU (ONNX fp32; on the chip the graphs are fp16 and zh/en/ko
waveforms differ slightly, the words do not: AI Hub check 10/11). Share it with Cloudflare quick tunnel:
  python tools/tts_web.py --models models.dev_beam.json --port 8765            (prints the access key)
  cloudflared tunnel --url http://localhost:8765                              (prints https://<random>.trycloudflare.com)
  open https://<random>.trycloudflare.com/?k=<access key>
Requests are served one at a time (the graphs are not shared between threads)."""
import argparse, io, json, os, secrets, sys, threading, time, wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from s2s_edge.registry import Registry

LANGS = {"vi": "Tiếng Việt (Piper)", "en": "English (Supertonic)", "zh": "中文 (MeloTTS)", "ko": "한국어 (Supertonic)"}
SAMPLE = {"vi": "Xin chào, hôm nay chúng ta họp lúc ba giờ chiều ở phòng 204.", "en": "Please wear your helmet before entering the factory.",
          "zh": "今天天气很好，我们一起去公园散步。", "ko": "회의는 세 시에 시작합니다. 늦지 마세요."}
PAGE = """<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Nghe thử TTS</title><style>
:root{--bg:#f6f7f9;--card:#fff;--fg:#1d1d1f;--mut:#667085;--acc:#2b5797;--bd:#d0d5dd}
@media (prefers-color-scheme:dark){:root{--bg:#111418;--card:#1b2027;--fg:#e8eaed;--mut:#9aa4b2;--acc:#7aa7e8;--bd:#333b45}}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:720px;margin:0 auto;padding:24px 16px}h1{font-size:22px;margin:0 0 4px}p.s{color:var(--mut);margin:0 0 20px;font-size:14px}
.card{background:var(--card);border:1px solid var(--bd);border-radius:12px;padding:16px}
.langs{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:12px}.langs button{flex:1 1 140px;padding:10px;border:1px solid var(--bd);border-radius:8px;background:transparent;color:var(--fg);cursor:pointer;font-size:15px}
.langs button.on{border-color:var(--acc);background:var(--acc);color:#fff}
textarea{width:100%;box-sizing:border-box;min-height:110px;padding:10px;border:1px solid var(--bd);border-radius:8px;background:transparent;color:var(--fg);font:inherit;resize:vertical}
.row{display:flex;gap:8px;align-items:center;margin-top:12px;flex-wrap:wrap}.go{padding:10px 18px;border:0;border-radius:8px;background:var(--acc);color:#fff;font-size:15px;cursor:pointer}
.go:disabled{opacity:.5;cursor:wait}#st{color:var(--mut);font-size:14px}audio{width:100%;margin-top:14px}
ul{padding-left:18px;color:var(--mut);font-size:14px}li a{color:var(--acc)}</style></head><body><main>
<h1>Nghe thử giọng đọc (TTS)</h1><p class="s">Chọn ngôn ngữ, nhập câu, bấm Đọc. Mô hình chạy trên máy của nhóm (CPU), mỗi câu mất vài giây.</p>
<div class="card"><div class="langs" id="langs"></div><textarea id="t" maxlength="300"></textarea>
<div class="row"><button class="go" id="go">Đọc</button><span id="st"></span></div><audio id="a" controls></audio></div>
<ul id="hist"></ul></main><script>
const K=new URLSearchParams(location.search).get('k')||'';const L=__LANGS__, S=__SAMPLE__;let lang='vi';
const lb=document.getElementById('langs'),t=document.getElementById('t'),go=document.getElementById('go'),st=document.getElementById('st'),au=document.getElementById('a'),hi=document.getElementById('hist');
for(const [k,v] of Object.entries(L)){const b=document.createElement('button');b.textContent=v;b.onclick=()=>{lang=k;t.value=S[k];draw()};b.dataset.k=k;lb.appendChild(b)}
function draw(){for(const b of lb.children)b.classList.toggle('on',b.dataset.k===lang)}t.value=S[lang];draw();
go.onclick=async()=>{const text=t.value.trim();if(!text)return;go.disabled=true;st.textContent='Đang tổng hợp...';const t0=performance.now();
 try{const r=await fetch('/api/tts',{method:'POST',headers:{'Content-Type':'application/json','X-Key':K},body:JSON.stringify({lang,text})});
  if(!r.ok){st.textContent='Lỗi: '+(await r.text());return}const b=await r.blob(),u=URL.createObjectURL(b);au.src=u;au.play();
  st.textContent=`Xong: audio ${r.headers.get('X-Audio-S')} s, tổng hợp ${r.headers.get('X-Synth-S')} s (CPU)`;
  const li=document.createElement('li');li.innerHTML=`[${lang}] ${text.replace(/</g,'&lt;')} — <a href="${u}" download="tts_${lang}.wav">tải về</a>`;hi.prepend(li)}
 catch(e){st.textContent='Lỗi kết nối: '+e}finally{go.disabled=false}};
</script></body></html>"""


def wav_bytes(x, sr):
    x = np.clip(np.asarray(x, np.float32), -1, 1); buf = io.BytesIO()
    with wave.open(buf, "wb") as f: f.setnchannels(1); f.setsampwidth(2); f.setframerate(int(sr)); f.writeframes((x * 32767).astype(np.int16).tobytes())
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--models", default="models.dev_beam.json"); ap.add_argument("--backend", default="cpu")
    ap.add_argument("--port", type=int, default=8765); ap.add_argument("--key", default=os.environ.get("TTS_WEB_KEY") or secrets.token_urlsafe(8)); a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8"); reg = Registry(a.models, a.backend); tts = {}; lock = threading.Lock()
    for lg in LANGS:                                                         # load + warm up every voice before serving
        t0 = time.perf_counter(); tts[lg] = reg.tts(lg); tts[lg].synthesize(SAMPLE[lg]); print(f"voice {lg} ready ({time.perf_counter() - t0:.1f} s)", flush=True)
    page = PAGE.replace("__LANGS__", json.dumps(LANGS, ensure_ascii=False)).replace("__SAMPLE__", json.dumps(SAMPLE, ensure_ascii=False)).encode("utf-8")

    class H(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype, extra=None):
            self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body))); self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items(): self.send_header(k, v)
            self.end_headers(); self.wfile.write(body)

        def log_message(self, fmt, *args): print(f"{self.address_string()} {fmt % args}", flush=True)

        def do_GET(self):
            if parse_qs(urlparse(self.path).query).get("k", [""])[0] != a.key: return self._send(403, "Cần đường link có mã truy cập (?k=...).".encode("utf-8"), "text/plain; charset=utf-8")
            self._send(200, page, "text/html; charset=utf-8")

        def do_POST(self):
            if urlparse(self.path).path != "/api/tts": return self._send(404, b"not found", "text/plain")
            if self.headers.get("X-Key") != a.key: return self._send(403, "sai mã truy cập".encode("utf-8"), "text/plain; charset=utf-8")
            try:
                n = min(int(self.headers.get("Content-Length", 0)), 10000); req = json.loads(self.rfile.read(n).decode("utf-8"))
                lg, text = req.get("lang"), str(req.get("text", "")).strip()[:300]
                if lg not in tts or not text: return self._send(400, "thiếu ngôn ngữ hoặc câu".encode("utf-8"), "text/plain; charset=utf-8")
                with lock:
                    t0 = time.perf_counter(); w, sr = tts[lg].synthesize(text); dt = time.perf_counter() - t0
                print(f"  [{lg}] {len(text)} chars -> {len(w) / sr:.1f} s audio in {dt:.1f} s", flush=True)
                self._send(200, wav_bytes(w, sr), "audio/wav", {"X-Audio-S": f"{len(w) / sr:.1f}", "X-Synth-S": f"{dt:.1f}"})
            except Exception as e:
                self._send(500, f"lỗi: {e!r}"[:300].encode("utf-8"), "text/plain; charset=utf-8")

    print(f"\nserving on http://localhost:{a.port}/?k={a.key}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()


if __name__ == "__main__": main()
