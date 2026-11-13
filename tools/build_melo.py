from pathlib import Path
import sys,os,json,gc
import torch,numpy as np,onnx
from transformers import AutoTokenizer,AutoModelForMaskedLM
root=Path(__file__).resolve().parents[1];base=root/'src/step4_quantization/tts/melotts';out=root/'build/melo';out.mkdir(exist_ok=True)
os.chdir(base);sys.path.insert(0,str(base/'MeloTTS'));sys.path.insert(0,str(base));torch.set_num_threads(4)
from melo.models import SynthesizerTrn
from melo_static import MeloStatic
from melo_full import BertStatic
from melo_graphA import GraphA
from melo_front import load_tables,LB,N,L
cfg=json.loads((root/'models/source/melo/config.json').read_text());print('LOAD_MELO_VITS',flush=True)
m=SynthesizerTrn(len(cfg['symbols']),cfg['data']['filter_length']//2+1,cfg['train']['segment_size']//cfg['data']['hop_length'],n_speakers=cfg['data']['n_speakers'],num_tones=cfg['num_tones'],num_languages=cfg['num_languages'],**cfg['model']).eval()
ck=torch.load(root/'models/source/melo/checkpoint.pth',map_location='cpu',weights_only=False);m.load_state_dict(ck['model'],strict=True);del ck;gc.collect()
m.dec.remove_weight_norm()
for f in m.flow.flows:
 if hasattr(f,'enc') and hasattr(f.enc,'remove_weight_norm'):f.enc.remove_weight_norm()
g=MeloStatic(m,N=96,Y=384,group=5).eval();x=torch.zeros(1,96,dtype=torch.int32);args=(x,x,x,torch.zeros(1,1024,96),torch.tensor([10],dtype=torch.int32))
print('EXPORT_MELO_VITS',flush=True)
with torch.no_grad():torch.onnx.export(g,args,str(out/'melo_N96_Y384_g5.onnx'),input_names=['x','tone','lang','bert','x_len'],output_names=['audio','y_len'],opset_version=17,do_constant_folding=True,dynamo=False)
del m,g,args;gc.collect()
print('LOAD_BERT_ZH',flush=True)
bpath=root/'models/source/bert-zh';tok=AutoTokenizer.from_pretrained(str(bpath),local_files_only=True);mlm=AutoModelForMaskedLM.from_pretrained(str(bpath),local_files_only=True).eval()
g=GraphA(load_tables(),BertStatic(mlm,L),tok).eval();a=torch.zeros(1,LB,dtype=torch.int32);b='今天天气很好。'.encode();a[0,:len(b)]=torch.tensor(list(b),dtype=torch.int32)
print('EXPORT_GRAPH_A',flush=True)
with torch.no_grad():torch.onnx.export(g,(a,),str(out/'graphA.onnx'),input_names=['text_bytes'],output_names=['output_0','output_1','output_2','output_3','output_4'],opset_version=17,do_constant_folding=True,dynamo=False)
del g,mlm;gc.collect()
print('MERGE_MELO',flush=True)
from onnx import compose
a=compose.add_prefix(onnx.load(out/'graphA.onnx'),'A_');v=compose.add_prefix(onnx.load(out/'melo_N96_Y384_g5.onnx'),'V_')
# Graph A and acoustic model are exported together with identical opset/IR.
merged=compose.merge_models(a,v,io_map=[('A_output_0','V_x'),('A_output_1','V_tone'),('A_output_2','V_lang'),('A_output_4','V_bert'),('A_output_3','V_x_len')],outputs=['V_audio','V_y_len'])
for n in merged.graph.node:
 n.input[:]=['text_bytes' if z=='A_text_bytes' else z for z in n.input]
 n.output[:]=['audio' if z=='V_audio' else 'y_len' if z=='V_y_len' else z for z in n.output]
merged.graph.input[0].name='text_bytes';merged.graph.output[0].name='audio';merged.graph.output[1].name='y_len';del merged.graph.value_info[:]
onnx.save(merged,out/'melo_one.onnx');print('MELO_READY',(out/'melo_one.onnx').stat().st_size,flush=True)
