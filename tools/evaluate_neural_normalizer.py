"""Development regression checks, not an independent generalization benchmark."""
from pathlib import Path
import sys,json,time,unicodedata
r=Path(__file__).resolve().parents[1];sys.path.insert(0,str(r/'s2s_edge'))
from s2s_edge.neural_textnorm import get
cases=[
('vi','time','Cuộc họp lúc 8:25 sáng.', ['tám giờ','hai mươi lăm']),
('vi','time','Chuyến xe khởi hành lúc 14h45.', ['mười bốn giờ','bốn mươi lăm']),
('vi','date','Hẹn ngày 21/12/2027.', ['hai mươi mốt','tháng mười hai','hai mươi bảy']),
('vi','decimal','Nhiệt độ là 36,8°C.', ['ba mươi sáu phẩy tám','độ']),
('vi','money','Giá vé là 75.000 đồng.', ['bảy mươi lăm nghìn','đồng']),
('vi','money','Giá là $18.25.', ['mười tám','đô','hai mươi lăm']),
('vi','percent','Giảm giá 12,5%.', ['mười hai phẩy năm phần trăm']),
('vi','phone','Số điện thoại là 0908 246 135.', ['không chín không tám hai bốn sáu một ba năm']),
('vi','code','Mã ZIP của tôi là 04782.', ['không bốn bảy tám hai']),
('vi','code','Chuyến bay VN247.', ['hai bốn bảy']),
('vi','context','Phòng 330 có giá 330 đồng.', ['ba trăm ba mươi']),
('en','time','The meeting is at 8:25 AM.', ['eight','twenty five']),
('en','date','The date is December 21, 2027.', ['december','twenty']),
('en','decimal','The temperature is 36.8 degrees Celsius.', ['thirty six point eight']),
('en','money','The price is $18.25.', ['eighteen dollars','twenty five cents']),
('en','percent','The discount is 12.5%.', ['twelve point five percent']),
('en','phone','My phone number is 0908246135.', ['zero nine zero eight two four six one three five']),
('en','code','My ZIP code is 04782.', ['zero four seven eight two']),
('en','code','The flight is VN247.', ['two four seven']),
('en','context','Room 330 costs 330 dollars.', ['three hundred thirty']),
]
def fold(s):return ' '.join(''.join(' ' if c in '-,.;:' else c.lower() for c in unicodedata.normalize('NFC',s)).split())
m=get();out=[]
for lang,kind,text,required in cases:
 y=m(text,lang);checks={v:fold(v) in fold(y) for v in required};passed=all(checks.values()) and not any(c.isdecimal() for c in y)
 row={'lang':lang,'kind':kind,'input':text,'output':y,'required_phrases':checks,'passed':passed,'seconds':m.records[-1]['seconds']};out.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
(r/'results/neural_normalizer_evaluation.json').write_text(json.dumps({'method':'learned tagger + pretrained task decoders; regression suite used during development; phrase checks are partial and do not prove semantic correctness','passed':sum(x['passed'] for x in out),'total':len(out),'cases':out},ensure_ascii=False,indent=2))
