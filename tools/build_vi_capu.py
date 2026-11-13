from pathlib import Path
import sys,torch,gc
from onnxruntime.quantization import quantize_dynamic,QuantType
root=Path(__file__).resolve().parents[1];p=root/'models/source/vi-capu';sys.path.insert(0,str(p))
from configuration_seq2labels import Seq2LabelsConfig
from modeling_seq2labels import Seq2LabelsModel
cfg=Seq2LabelsConfig.from_pretrained(str(p));cfg.pretrained_name_or_path=str(root/'models/source/vibert-base-config');cfg.load_pretrained=False
torch.set_num_threads(4);m=Seq2LabelsModel(cfg).eval();state=torch.load(p/'pytorch_model.bin',map_location='cpu',weights_only=True);
if 'bert.embeddings.position_ids' in state:
 assert torch.equal(state['bert.embeddings.position_ids'],m.bert.embeddings.position_ids)
 state.pop('bert.embeddings.position_ids')
m.load_state_dict(state,strict=True);del state;gc.collect()
class Export(torch.nn.Module):
 def __init__(self,m):super().__init__();self.m=m
 def forward(self,input_ids,token_type_ids,attention_mask,input_offsets):return self.m(input_ids=input_ids,token_type_ids=token_type_ids,attention_mask=attention_mask,input_offsets=input_offsets,return_dict=False)[:2]
out=root/'build/vi-capu';out.mkdir(exist_ok=True);ids=torch.tensor([[101,38168,500,1000,102]],dtype=torch.long);args=(ids,torch.zeros_like(ids),torch.ones_like(ids),torch.tensor([[1,2,3]],dtype=torch.long))
print('EXPORT_VI_CAPU',flush=True)
with torch.no_grad():torch.onnx.export(Export(m),args,str(out/'vibert_capu.onnx'),input_names=['input_ids','token_type_ids','attention_mask','input_offsets'],output_names=['logits','detect_logits'],dynamic_axes={'input_ids':{0:'batch',1:'tokens'},'token_type_ids':{0:'batch',1:'tokens'},'attention_mask':{0:'batch',1:'tokens'},'input_offsets':{0:'batch',1:'words'},'logits':{0:'batch',1:'words'},'detect_logits':{0:'batch',1:'words'}},opset_version=17,do_constant_folding=True,dynamo=False)
quantize_dynamic(str(out/'vibert_capu.onnx'),str(root/'s2s_edge/assets/vi_capu/vibert_capu_int8.onnx'),weight_type=QuantType.QInt8,per_channel=True)
print('VI_CAPU_READY',flush=True)
