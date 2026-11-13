from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import hf_hub_download
import json
root=Path(__file__).resolve().parents[1]
specs=[('hynt/Zipformer-150M-CR-CTC-RNNT-6000h','8c3d474d3d74cb247f4b8c7b0f8b48598ac8c1c6','zip150',['bpe.model','config.json','encoder-epoch-11-avg-2.onnx','pretrained.pt']),('myshell-ai/MeloTTS-Chinese','af5d207a364ea4208c6f589c89f57f88414bdd16','melo',['config.json','checkpoint.pth']),('hfl/chinese-roberta-wwm-ext-large','a25cc9e05974bd9687e528edd516f2cfdb3f5db9','bert-zh',['config.json','vocab.txt','tokenizer.json','tokenizer_config.json','special_tokens_map.json','pytorch_model.bin'])]
def download(s):
 repo,rev,name,files=s
 for f in files:
  print('FETCH',name,f,flush=True);p=hf_hub_download(repo,f,revision=rev,local_dir=root/'models/source'/name);print('READY',name,f,Path(p).stat().st_size,flush=True)
with ThreadPoolExecutor(3) as ex:list(ex.map(download,specs))
(root/'models/source/revisions.json').write_text(json.dumps({n:{'repo':r,'revision':v} for r,v,n,f in specs},indent=2));print('DOWNLOADS_COMPLETE',flush=True)
