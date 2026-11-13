from pathlib import Path
import os,sys,pickle,json,time,subprocess
root=Path(__file__).resolve().parents[1];base=root/'src/step4_quantization/tts/piper';g=base/'g2p';(g/'build').mkdir(exist_ok=True)
vdir=base/'outputs/piper_vi_npu';vdir.mkdir(parents=True,exist_ok=True)
for name in ['vi_VN-vais1000-medium.onnx','vi_VN-vais1000-medium.onnx.json']:
 dst=vdir/name
 if not dst.exists():dst.symlink_to(Path('/private/tmp/tts-piper-model')/name)
sys.path.insert(0,str(g));import g2p_graph as G
from piper import PiperVoice
vp=vdir/'vi_VN-vais1000-medium.onnx';voice=PiperVoice.load(str(vp));idm=json.loads(Path(str(vp)+'.json').read_text())['phoneme_id_map']
letters,classes,pats,pcls=G.build_patterns();tab={};t=time.perf_counter()
for word in (root/'s2s_edge/assets/piper_vi_words.txt').read_text().splitlines():
 if len(word)>G.S:continue
 phones=voice.phonemize(word)
 if len(phones)!=1:continue
 try:ids=[idm[c][0] for c in phones[0]]
 except KeyError:continue
 if ids and len(ids)<=14:tab[word]=ids
for c in G.PUNCT_CLASSES:tab[c]=[idm[c][0]]
tab[' ']=[3];keys,phs,cp=G.build_tables(list(tab),lambda x:tab[x],classes,letters,idm)
pickle.dump((letters,classes,pats,pcls,keys,phs,tab),(g/'build/g2p_tables.pkl').open('wb'))
print('TABLES_READY',len(tab),time.perf_counter()-t,flush=True)
subprocess.run([sys.executable,'export_full.py','8'],cwd=g,check=True)
print('PIPER_READY',flush=True)
