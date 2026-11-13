"""Local mic/file web app -> actual integrated ASR/MT/normalization/TTS.
No browser speech recognition or reference transcripts. One inference worker.
"""
import argparse,gc,io,json,os,sys,time,traceback,uuid,threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlparse,parse_qs
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'s2s_edge'))
os.environ.setdefault('S2S_NEURAL_TN','1')
os.environ.setdefault('S2S_TTS_FORMAT','hybrid')
os.environ.setdefault('HF_HOME','/private/tmp/s2s-hf')
MAX_BYTES=2*1024*1024

class Service:
 def __init__(self,out):
  self.out=out;out.mkdir(parents=True,exist_ok=True);self.jobs={};self.lock=threading.RLock();self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='s2s')
 def update(self,key,**fields):
  with self.lock:self.jobs[key].update(fields)
 def submit(self,data,src):
  import soundfile as sf
  if len(data)>MAX_BYTES:raise ValueError('Bản thu quá lớn. Hãy nói tối đa 30 giây.')
  try:
   info=sf.info(io.BytesIO(data))
  except Exception:raise ValueError('Không đọc được audio. Hãy thu lại hoặc chọn một file âm thanh khác.')
  if info.frames/info.samplerate>30.1 or info.frames/info.samplerate<.25:raise ValueError('Audio cần dài từ 0,25 đến 30 giây.')
  if info.channels>2:raise ValueError('Chỉ nhận audio mono hoặc stereo.')
  with self.lock:
   if sum(j['status'] in ('queued','running') for j in self.jobs.values())>=3:raise ValueError('Máy đang xử lý các bản thu trước. Hãy thử lại sau.')
   key=uuid.uuid4().hex;folder=self.out/key;folder.mkdir();(folder/'input.wav').write_bytes(data)
   self.jobs[key]={'id':key,'status':'queued','stage':'Đang chờ xử lý','src':src,'tgt':'vi' if src=='en' else 'en','created':time.time()}
  self.pool.submit(self.run,key);return key
 def run(self,key):
  from s2s_edge.registry import Registry
  from s2s_edge.pipeline import Translator
  from s2s_edge.audio import load_audio,save_wav,segment,concat_results,SR
  from s2s_edge import neural_textnorm
  import numpy as np
  reg=None;folder=self.out/key;start=time.perf_counter()
  with self.lock:src,tgt=self.jobs[key]['src'],self.jobs[key]['tgt']
  try:
   self.update(key,status='running',stage='Đang chuẩn bị mô hình…')
   os.environ['S2S_TN_AUDIT']=str(folder/'normalization.jsonl')
   w=load_audio(str(folder/'input.wav'))
   if not len(w) or not np.isfinite(w).all() or float(np.max(abs(w)))<1e-5:raise ValueError('Bản thu không có tín hiệu. Hãy kiểm tra mic và nói lại.')
   reg=Registry(str(ROOT/'models/models.cpu.json'),'cpu');translator=Translator(reg,src,tgt)
   raw=[];service=self
   class ASR:
    def __init__(self,m):self.m=m
    def __getattr__(self,n):return getattr(self.m,n)
    def __call__(self,audio):
     service.update(key,stage='Đang nhận dạng giọng nói…');text=self.m(audio);raw.append(text);service.update(key,stage='Đang chuẩn hóa văn bản nguồn…');return text
   class MT:
    def __init__(self,m):self.m=m
    def __getattr__(self,n):return getattr(self.m,n)
    def translate_atomic(self,*a):service.update(key,stage='Đang dịch…');return self.m.translate_atomic(*a)
    def __call__(self,*a):service.update(key,stage='Đang dịch…');return self.m(*a)
   class TTS:
    def __init__(self,m):self.m=m
    def __getattr__(self,n):return getattr(self.m,n)
    def synthesize(self,text):service.update(key,stage='Đang chuẩn hóa và tạo giọng nói…');return self.m.synthesize(text)
   translator.e._asr[src]=ASR(translator.asr);translator.e._mt=MT(translator.e._mt);translator.e._tts[tgt]=TTS(translator.tts)
   spans=segment(w,max_s=min(translator.asr.max_s,12.0));results=[]
   if not spans:raise ValueError('Chưa nhận ra lời nói. Hãy nói rõ hơn và đưa mic lại gần.')
   trace_start=len(neural_textnorm.get().records) if neural_textnorm.enabled() else 0
   for i,(a,b) in enumerate(spans):
    self.update(key,segment=i+1,segments_total=len(spans))
    result=translator.translate_segment(w[a:b]);result['asr_raw']=raw[-1];result['span']=[a/SR,b/SR];results.append(result)
   audio,sr=concat_results(results)
   if not len(audio):raise ValueError('Không nhận được nội dung để dịch. Hãy nói lại một câu ngắn, rõ hơn.')
   if not np.isfinite(audio).all():raise RuntimeError('Mô hình trả về audio không hợp lệ.')
   save_wav(str(folder/'output.wav'),audio,sr)
   result={'id':key,'status':'done','stage':'Hoàn tất','src':src,'tgt':tgt,'input_seconds':len(w)/SR,'output_seconds':len(audio)/sr,'elapsed_seconds':time.perf_counter()-start,'input_url':f'/audio/{key}/input.wav','output_url':f'/audio/{key}/output.wav','detail_url':f'/audio/{key}/result.json','asr':' '.join(z['asr_raw'] for z in results),'translation':' '.join(z['mt'] for z in results),'tts_text':' '.join(z['tts_text'] for z in results),'segments':[{k:v for k,v in z.items() if k!='wav'} for z in results],'normalization':neural_textnorm.get().records[trace_start:] if neural_textnorm.enabled() else [],'backend':'integrated ONNX CPU + learned normalization on Apple Metal','input_kind':'uploaded speech audio; no transcript provided'}
   (folder/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));self.update(key,**{k:v for k,v in result.items() if k!='id'})
   print('DONE',key,src,tgt,round(result['elapsed_seconds'],2),flush=True)
  except Exception as e:
   traceback.print_exc();self.update(key,status='error',stage='Chưa xử lý được bản thu',error=str(e)[:500])
   with self.lock:(folder/'error.json').write_text(json.dumps(self.jobs[key],ensure_ascii=False,indent=2))
  finally:
   if reg is not None:
    for k in list(reg._cache):reg.release(k)
   gc.collect()
 def snapshot(self,key):
  with self.lock:
   j=self.jobs.get(key)
   return dict(j) if j else None

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--port',type=int,default=8766);args=ap.parse_args();service=Service(ROOT/'results/mic_web')
 class Handler(BaseHTTPRequestHandler):
  def send(self,status,body,ctype='application/json; charset=utf-8'):
   self.send_response(status);self.send_header('Content-Type',ctype);self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.send_header('Permissions-Policy','microphone=(self)');self.end_headers()
   try:self.wfile.write(body)
   except (BrokenPipeError,ConnectionResetError):pass
  def json(self,status,value):self.send(status,json.dumps(value,ensure_ascii=False).encode())
  def log_message(self,fmt,*args):
   if self.path!='/api/health' and not self.path.startswith('/api/jobs/'):print(fmt%args,flush=True)
  def do_GET(self):
   path=urlparse(self.path).path
   static={'/':('index.html','text/html; charset=utf-8'),'/app.js':('app.js','text/javascript; charset=utf-8'),'/style.css':('style.css','text/css; charset=utf-8'),'/recorder.js':('recorder.js','text/javascript; charset=utf-8')}
   if path in static:
    name,ctype=static[path];return self.send(200,(ROOT/'web'/name).read_bytes(),ctype)
   if path=='/api/health':
    with service.lock:pending=sum(j['status'] in ('queued','running') for j in service.jobs.values())
    return self.json(200,{'ok':True,'pending':pending,'max_seconds':30,'normalization':os.environ['S2S_NEURAL_TN']=='1','tts_format':os.environ['S2S_TTS_FORMAT']})
   if path.startswith('/api/jobs/'):
    j=service.snapshot(path.rsplit('/',1)[-1]);return self.json(200,j) if j else self.json(404,{'error':'Không tìm thấy bản thu.'})
   if path.startswith('/audio/'):
    parts=path.split('/')
    if len(parts)!=4 or parts[3] not in ('input.wav','output.wav','result.json'):return self.json(404,{'error':'Not found'})
    try:uuid.UUID(hex=parts[2])
    except ValueError:return self.json(404,{'error':'Not found'})
    p=service.out/parts[2]/parts[3]
    if not p.is_file():return self.json(404,{'error':'Audio chưa sẵn sàng.'})
    return self.send(200,p.read_bytes(),'audio/wav' if p.suffix=='.wav' else 'application/json; charset=utf-8')
   return self.json(404,{'error':'Not found'})
  def do_POST(self):
   if urlparse(self.path).path!='/api/jobs':return self.json(404,{'error':'Not found'})
   origin=self.headers.get('Origin')
   if origin and origin not in (f'http://localhost:{args.port}',f'http://127.0.0.1:{args.port}'):return self.json(403,{'error':'Hãy mở trang trên localhost.'})
   try:
    src=parse_qs(urlparse(self.path).query).get('src',[''])[0]
    if src not in ('en','vi'):raise ValueError('Hãy chọn chiều Anh → Việt hoặc Việt → Anh.')
    n=int(self.headers.get('Content-Length','0'))
    if n<=0 or n>MAX_BYTES:raise ValueError('Kích thước audio không hợp lệ; tối đa 30 giây.')
    data=self.rfile.read(n)
    if not data.startswith(b'RIFF') or data[8:12]!=b'WAVE':raise ValueError('Hãy gửi audio WAV; trình duyệt sẽ tự chuyển đổi bản thu.')
    key=service.submit(data,src);self.json(202,{'id':key})
   except ValueError as e:self.json(400,{'error':str(e)})
   except Exception:
    traceback.print_exc();self.json(500,{'error':'Không gửi được bản thu. Hãy thử lại.'})
 print(f'OneVoice mic web: http://localhost:{args.port}',flush=True)
 ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()

if __name__=='__main__':main()
