from pathlib import Path
import sys,types,time,gc
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'src'));import common
sys.path.insert(0,str(root/'src/step4_quantization/asr/sensevoice/export'))
import torch,numpy as np
from funasr import AutoModel
from step4_s1_export_sensevoice_e2e_unified import StaticSinusoidalPositionEncoder,StaticCTCCollapseAndDetokenizer,build_static_byte_table,MAX_SEQ_FRAMES,L_MAX
from step4_s1_export_sensevoice_split_npu import EncoderClassifierGraph
from sv_export_frontend_v3_perframe import TraceableFrontendV3,FrontendGraphV3
from funasr.models.sense_voice.export_meta import export_rebuild_model
out=root/'build/sensevoice';out.mkdir(exist_ok=True);torch.set_num_threads(4)
print('LOAD_LOCAL_SENSEVOICE',flush=True)
am=AutoModel(model='/Users/khoa/.cache/modelscope/models/iic--SenseVoiceSmall/snapshots/master',device='cpu',disable_update=True)
sp=am.kwargs['tokenizer'].sp;table=build_static_byte_table(sp,vocab_size=sp.get_piece_size(),l_max=L_MAX)
fe=FrontendGraphV3(TraceableFrontendV3(am.kwargs['frontend'].cmvn)).eval()
print('EXPORT_FRONTEND_V3',flush=True)
with torch.no_grad():torch.onnx.export(fe,(torch.randn(1,464000),torch.tensor([464000],dtype=torch.int32)),str(out/'sv_frontend_v3.onnx'),input_names=['wav','wav_len'],output_names=['fbank','speech_lengths'],opset_version=17,do_constant_folding=True,dynamo=False)
sv=export_rebuild_model(am.model.eval(),device='cpu',max_seq_len=512);sv.export_dynamic_axes=types.MethodType(lambda self:{},sv)
sv.encoder.embed=StaticSinusoidalPositionEncoder(timesteps=MAX_SEQ_FRAMES,depth=560)
dec=StaticCTCCollapseAndDetokenizer(table,max_frames=MAX_SEQ_FRAMES,l_max=L_MAX)
g=EncoderClassifierGraph(sv,dec).eval();print('EXPORT_ENCODER_CLASSIFIER',MAX_SEQ_FRAMES,flush=True)
args=(torch.zeros(1,500,560),torch.tensor([100],dtype=torch.int32),torch.tensor([4],dtype=torch.int32),torch.tensor([14],dtype=torch.int32))
with torch.no_grad():torch.onnx.export(g,args,str(out/'sv_enc_cls.onnx'),input_names=['fbank','speech_lengths','language','textnorm'],output_names=['byte_stream'],opset_version=17,do_constant_folding=True,dynamo=False)
print('SENSEVOICE_READY',flush=True)
