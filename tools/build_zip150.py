from pathlib import Path
import sys,gc
import numpy as np,torch,onnx
from onnx import helper,numpy_helper,TensorProto as TP
root=Path(__file__).resolve().parents[1];src=root/'models/source/zip150';out=root/'build/zip150';out.mkdir(exist_ok=True)
sys.path.insert(0,str(root/'src/step4_quantization/asr/zipformer/step4_hardware'))
from prepare_zipformer_for_qnn import find_bool_slices,wrap_bool_slice
print('LOAD_CTC_HEAD',flush=True)
ck=torch.load(src/'pretrained.pt',map_location='cpu',mmap=True,weights_only=False)
state=ck['model'];W=state['ctc_output.1.weight'].numpy().copy();b=state['ctc_output.1.bias'].numpy().copy();del state,ck;gc.collect()
assert W.shape==(2000,768);np.savez(out/'ctc_head.npz',W=W,b=b)
print('LOAD_ENCODER',flush=True);m=onnx.load(src/'encoder-epoch-11-avg-2.onnx')
for name in find_bool_slices(m):wrap_bool_slice(m.graph,name)
by={o:n for n in m.graph.node for o in n.output};mm=by['/encoder_proj/MatMul_output_0'];ad=by['encoder_out'];pre=mm.input[0]
remove={mm.input[1],'encoder_proj.bias'};m.graph.node.remove(mm);m.graph.node.remove(ad)
for x in list(m.graph.initializer):
 if x.name in remove:m.graph.initializer.remove(x)
m.graph.initializer.extend([numpy_helper.from_array(np.ascontiguousarray(W.T),'ctc_head.W'),numpy_helper.from_array(b,'ctc_head.b')])
m.graph.node.extend([helper.make_node('MatMul',[pre,'ctc_head.W'],['ctc_mm']),helper.make_node('Add',['ctc_mm','ctc_head.b'],['ctc_logits'])])
del m.graph.output[:];m.graph.output.extend([helper.make_tensor_value_info('ctc_logits',TP.FLOAT,[1,None,2000]),helper.make_tensor_value_info('encoder_out_lens',TP.INT64,[1])])
for x in m.graph.input:
 vals=[1,1500,80] if x.name=='x' else [1]
 for d,v in zip(x.type.tensor_type.shape.dim,vals):d.ClearField('dim_param');d.dim_value=v
p=out/'zip150_ctc_static_1500.onnx';onnx.save(m,p)
print('ZIP150_READY',p,p.stat().st_size,flush=True)
# Verify the shipped tokenizer agrees with the model being assembled.
assert (src/'bpe.model').read_bytes()==(root/'s2s_edge/assets/zipformer_bpe.model').read_bytes(),'Tokenizer mismatch'
