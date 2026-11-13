# -*- coding: utf-8 -*-
import os, sys, re, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from s2s_edge.asr_clean import clean_mt

KEEP = [  # (mt output, source, tgt): real translations that must not change
 ("-5 độ C vào ban đêm.", "-5 degrees at night.", "vi"), ("Không.", "No.", "vi"),
 ("特朗普与土耳其总统雷杰普·塔伊普·埃尔多安通话。", "Trump đã điện đàm với Tổng thống Recep Tayyip Erdoğan.", "zh"),
 ("Q: Bạn tên gì?", "Q: What is your name?", "vi"), ("A: Tôi tên Lan.", "A: My name is Lan.", "vi"),
 ("그래, 그래.", "Vâng.", "ko"), ("아니, 아니.", "Không có.", "ko"), ("是的,是的.他是我的医生.", "Phải. Ông ta là bác sĩ của tôi.", "zh"),
 ("你好。", "Xin chào.", "zh"), ("\"Xin chào\", anh ấy nói.", "\"Hello,\" he said.", "vi"), ("15.000 đồng.", "15000원이에요.", "vi"), ("3.5 km.", "3,5 km.", "en"),
 ("Và rồi anh ấy đi.", "And then he left.", "vi"), ("Có.", "Yes.", "vi"),
]
CHANGE = [  # (mt output, source, tgt, expected, tag)
 ("❚ Ông có hơn 70 chiếc răng. ❚ Điều này có nghĩa là không thoát được. ‖", "他的…", "vi", "Ông có hơn 70 chiếc răng. Điều này có nghĩa là không thoát được.", "mt_symbol"),
 ("问:她借给你的那本中文书,你看完了吗?", "Sách tiếng Trung đó mà, cô ta cho bạn mượn, bạn đã xem xong chưa?", "zh", "她借给你的那本中文书,你看完了吗?", "mt_label"),
 ("- Yes, I did. I added a battery.", "Có. Tôi thêm một viên pin.", "en", "Yes, I did. I added a battery.", "mt_lead_punct"),
 (". 침대를 교환하고 싶지만, 자리가 없어?", "Tôi muốn đổi vé giường nằm.", "ko", "침대를 교환하고 싶지만, 자리가 없어?", "mt_lead_punct"),
 ("。 Vì vậy, có thể biểu tượng này chỉ được thêm vào như là một thẻ。", "因此…", "vi", "Vì vậy, có thể biểu tượng này chỉ được thêm vào như là một thẻ.", "mt_lead_punct"),
 ("Xin chào。Bạn khỏe không?", "你好。你好吗？", "vi", "Xin chào. Bạn khỏe không?", "mt_cjk_stop"),
]

def test_keep():
    bad = [(o, clean_mt(o, s, t)) for o, s, t in KEEP if clean_mt(o, s, t)[0] != o.strip() or clean_mt(o, s, t)[1]]
    assert not bad, f"damaged a real translation: {bad}"

def test_change():
    bad = []
    for o, s, t, exp, tag in CHANGE:
        r, tg = clean_mt(o, s, t)
        if r != exp or tag not in tg: bad.append((o, r, tg, "expected", exp, tag))
    assert not bad, f"wrong: {bad}"

def test_properties():
    random.seed(1); al = list("abc xyz đôơ 你好 그래 .,。-—❚‖·:") + ["问:", "Q:", "- ", "-5", "。 "]
    for _ in range(20000):
        o = "".join(random.choice(al) for _ in range(random.randint(0, 15))); t = random.choice(["vi", "en", "zh", "ko"])
        r, tg = clean_mt(o, "", t); r2, tg2 = clean_mt(r, "", t)
        assert r2 == r, (o, r, r2)                                                            # idempotent
        assert set(re.findall(r"\w", r)) <= set(re.findall(r"\w", o)), (o, r)                  # never invents a letter
        if "mt_label" not in tg: assert len(re.findall(r"\w", r)) == len(re.findall(r"\w", o)), (o, r)  # only the label rule may drop a letter                  
        if not tg: assert r == o.strip(), (o, r)

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("PASS", n)
