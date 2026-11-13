from pathlib import Path
import os,sys,onnx,subprocess,json
root=Path(__file__).resolve().parents[1];work=root/'build/supertonic';os.environ['ST_WORK']=str(work)
p=root/'src/step4_quantization/tts/supertonic/scripts/rot_fix.py'
ns={'__file__':str(p),'__name__':'rot_fix_library'}
exec(compile(p.read_text().split('if sys.argv[1]=="make":')[0],str(p),'exec'),ns)
m,count=ns['fix']('R');dst=work/'ve_rotR_L150.onnx';onnx.save(m,dst);print('REDUCESUM_FIX',count,flush=True)
subprocess.run([sys.executable,str(root/'src/step4_quantization/tts/supertonic/scripts/fuse.py'),'96','150','5'],check=True)
f=work/'supertonic_full_N96_L150_s5.onnx';print('READY',f,f.stat().st_size,flush=True)
