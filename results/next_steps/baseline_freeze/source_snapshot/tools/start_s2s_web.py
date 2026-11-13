"""Start the user-facing local server outside the tool's foreground session."""
from pathlib import Path
import os,sys,subprocess,time,urllib.request,json
root=Path(__file__).resolve().parents[1];url='http://localhost:8766'
def ready():
 try:
  with urllib.request.urlopen(url+'/api/health',timeout=2) as r:return json.load(r).get('ok') is True
 except Exception:return False
if not ready():
 env=os.environ.copy();env.update({'S2S_NEURAL_TN':'1','S2S_TTS_FORMAT':'hybrid','HF_HOME':'/private/tmp/s2s-hf'})
 with (root/'logs/s2s_web.log').open('a') as f:
  p=subprocess.Popen([sys.executable,'-u',str(root/'tools/s2s_web.py'),'--port','8766'],cwd=str(root),env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 (root/'results/mic_web/server.pid').write_text(str(p.pid))
 for _ in range(40):
  if ready():break
  if p.poll() is not None:raise RuntimeError('Web server exited; inspect logs/s2s_web.log')
  time.sleep(.25)
 if not ready():raise RuntimeError('Web server did not become ready')
print(url)
