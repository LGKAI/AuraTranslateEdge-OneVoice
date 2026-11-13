import sys,json
from pathlib import Path
r=Path(__file__).resolve().parents[1];sys.path.insert(0,str(r/'s2s_edge'))
from s2s_edge.neural_textnorm import get
samples=[
('vi','Cuộc họp lúc 3:30 chiều ngày 15/11/2026.'),
('vi','Giá vé 50.000 đồng, giảm 15%. Nhiệt độ 37,5°C.'),
('vi','Xin gọi số 0912 345 678. Chuyến bay VN123, mã ZIP 67297.'),
('en','The meeting is at 3:30 PM on November 15, 2026. The ticket costs $15.50, with a 15% discount. The temperature is 37.5 degrees Celsius.'),
('en','Call 0912345678. My flight is VN123 and my ZIP code is 67297.'),
('vi','Phòng 330 có giá 330 đồng. Cuộc họp lúc 14h45.'),
('en','The meeting is at 330 pm on November 15, 2026, the ticket costs $15.50 with a 15% discount.'),
]
m=get()
for lang,text in samples:
 print('INPUT',lang,text,flush=True)
 print('OUTPUT',m(text,lang),flush=True)
(r/'results/neural_normalizer_smoke.json').write_text(json.dumps(m.records,ensure_ascii=False,indent=2))
