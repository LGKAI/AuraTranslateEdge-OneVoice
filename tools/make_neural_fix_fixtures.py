"""Synthetic speech INPUT fixtures; their text is never passed to ASR/MT.
Sentences are synthesized separately and joined with pauses to make clear speech.
"""
from pathlib import Path
import sys,json,numpy as np
r=Path(__file__).resolve().parents[1];sys.path.insert(0,str(r/'s2s_edge'))
from s2s_edge.registry import Registry
from s2s_edge.audio import save_wav
out=r/'results/neural_fix';(out/'audio').mkdir(exist_ok=True,parents=True)
lang=sys.argv[1];reg=Registry(str(r/'models/models.cpu.json'),'cpu');tts=reg.tts(lang)
cases={
'en':{
'written':'The meeting is at 3:30 PM on November 15, 2026. The ticket costs $15.50, with a 15% discount. The temperature is 37.5°C. My flight is VN123. My ZIP code is 67297. Call 0912345678.',
'sentences':['The meeting is at three thirty in the afternoon, on November fifteenth, twenty twenty six.','The ticket costs fifteen dollars and fifty cents, with a fifteen percent discount.','The temperature is thirty seven point five degrees Celsius.','My flight is V N one two three.','My ZIP code is six seven two nine seven.','Call zero nine one two three four five six seven eight.']},
'vi':{
'written':'Cuộc họp lúc 3:30 chiều ngày 15/11/2026. Giá vé 50.000 đồng, giảm 15%. Nhiệt độ 37,5°C. Xin gọi số 0912345678. Mã bưu chính là 67297.',
'sentences':['Cuộc họp lúc ba giờ ba mươi phút chiều.','Hôm nay là ngày mười lăm tháng mười một năm hai nghìn không trăm hai mươi sáu.','Giá vé năm mươi nghìn đồng.','Giảm giá mười lăm phần trăm.','Nhiệt độ ba mươi bảy phẩy năm độ xê.','Xin gọi số không chín một hai ba bốn năm sáu bảy tám.','Mã bưu chính là sáu bảy hai chín bảy.']}}
c=cases[lang];waves=[];audit=[]
for text in c['sentences']:
 w,sr=tts.synthesize(text);waves.extend([w,np.zeros(int(sr*1.0),np.float32)]);audit.append({'text':text,'seconds':len(w)/sr});print(text,len(w)/sr,flush=True)
p=out/'audio'/f'{lang}_input_clear.wav';save_wav(str(p),np.concatenate(waves[:-1]),sr)
(out/f'{lang}_input.json').write_text(json.dumps({**c,'audio':str(p),'kind':'synthetic speech input only','sentences_synthesized':audit},ensure_ascii=False,indent=2))
