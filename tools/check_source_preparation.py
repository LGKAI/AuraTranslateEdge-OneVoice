from pathlib import Path
import sys,json
r=Path(__file__).resolve().parents[1];sys.path.insert(0,str(r/'s2s_edge'))
from s2s_edge.neural_textnorm import get
m=get()
p=json.loads((r/'results/rerun_2026-11-12/vi_to_en.json').read_text())['segments'][0]['asr_raw']
print('SOURCE',p,flush=True);print('IDENTIFIERS',m.canonical_identifiers(p,'vi'),flush=True)
