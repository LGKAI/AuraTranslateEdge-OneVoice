# -*- coding: utf-8 -*-
import os, sys, csv, re, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from s2s_edge.asr_clean import clean_asr

KEEP = [  # real speech that a careless filter would damage: must come back unchanged
 ("Có.", "vi"), ("Có. Tôi thêm một viên pin.", "vi"), ("Vâng.", "vi"), ("Dạ.", "vi"), ("Ừ.", "vi"), ("Ờ, được rồi.", "vi"), ("A, tôi hiểu rồi.", "vi"), ("À, vâng.", "vi"),
 ("Ô, đẹp quá.", "vi"), ("Y tế ở đâu?", "vi"), ("Không. Không. Không.", "vi"), ("Xin lỗi xin lỗi xin lỗi.", "vi"), ("Hà Nội.", "vi"), ("Hơn.", "vi"),
 ("Khách sạn K.O. ở đâu?", "vi"), ("Phòng H.264 ở đâu?", "vi"), ("Q1 tăng 5%.", "vi"), ("Số 5.", "vi"), ("Tầng 3.", "vi"), ("Tôi tên là Q.", "vi"),
 ("Ok.", "en"), ("OK, fine.", "en"), ("Okay.", "en"), ("Yes.", "en"), ("No.", "en"), ("I do.", "en"),
 ("Let me check.", "en"), ("Where is the exit?", "en"), ("Press Q to quit.", "en"), ("Room 12.", "en"), ("That that is true.", "en"), ("I had had enough.", "en"), ("very very very good", "en"),
 ("好。", "zh"), ("好的。", "zh"), ("嗯，我知道了。", "zh"), ("啊，是吗？", "zh"), ("哦。", "zh"), ("问题在哪里？", "zh"), ("答案是五。", "zh"), ("问你一个问题。", "zh"), ("哈哈哈哈", "zh"),
 ("没。", "zh"), ("没。我没去。", "zh"), ("没。", "zh"), ("嗯，好的。", "zh"), ("그.", "ko"), ("그, 그건 아니에요.", "ko"), ("그 사람은 누구예요?", "ko"), ("그건 뭐예요?", "ko"),
 ("Vâng. Tôi hiểu.", "vi"), ("Có. Có.", "vi"), ("네.", "ko"), ("예.", "ko"), ("응.", "ko"), ("여름.", "ko"), ("음, 알겠어요.", "ko"), ("어, 그래요?", "ko"), ("안녕하세요.", "ko"),
]
CHANGE = [  # (input, lang, expected clean, expected tag)
 ("Q. Sách tiếng Trung đó mà, cô ta cho bạn mượn", "vi", "Sách tiếng Trung đó mà, cô ta cho bạn mượn", "label"),
 ("H. Hàng ngày bạn đều dậy sớm như vậy à?", "vi", "Hàng ngày bạn đều dậy sớm như vậy à?", "label"),
 ("Q: Bạn tên gì?", "vi", "Bạn tên gì?", "label"),
 ("问：你叫什么名字？", "zh", "你叫什么名字？", "label"),
 ("答: 我叫小明。", "zh", "我叫小明。", "label"),
 ("Um. Tôi muốn đổi lại vé giường nằm.", "vi", "Tôi muốn đổi lại vé giường nằm.", "filler"),
 ("Ừm, tôi muốn đặt phòng.", "vi", "tôi muốn đặt phòng.", "filler"),
 ("Uh, where is the exit?", "en", "where is the exit?", "filler"),
 ("Um. Q. Sách tiếng Trung", "vi", "Sách tiếng Trung", "label"),
 ("Hmm, let me think about it.", "en", "let me think about it.", "filler"), ("Um.", "vi", "", "filler_only"), ("Hmm.", "en", "", "filler_only"), ("嗯。", "zh", "", "filler_only"), ("음.", "ko", "", "filler_only"),
 ("。", "zh", "", "no_text"), (".", "vi", "", "no_text"), ("...", "en", "", "no_text"), ("  ", "vi", "", None), ("I.", "en", "", "no_text"), ("A", "vi", "", "no_text"),
 ("嗯。 1963年大坝建成后，季节性洪水都控制出来。 嗯。", "zh", "1963年大坝建成后，季节性洪水都控制出来。", "edge_filler"),
 ("嗯。 因此，很可能这个符号只是作为一个标签添加的。", "zh", "因此，很可能这个符号只是作为一个标签添加的。", "edge_filler"),
 ("그. 지하철의 정규 안내 방송은 카탈로니아어로만 제공됩니다.", "ko", "지하철의 정규 안내 방송은 카탈로니아어로만 제공됩니다.", "edge_filler"),
 ("대서양에서 생성되었습니다. 그.", "ko", "대서양에서 생성되었습니다.", "edge_filler"),
 ("1963年大坝建成后，沉积物不再冲散到河流里。 没。", "zh", "1963年大坝建成后，沉积物不再冲散到河流里。", "edge_filler"),
 ("Tôi muốn đi. Ừm.", "vi", "Tôi muốn đi.", "edge_filler"),
 ("I want to go. Um.", "en", "I want to go.", "edge_filler"),
 ("the the the the the the exit is there", "en", "the the exit is there", "repeat"),
 ("tôi tôi tôi tôi tôi muốn đi", "vi", "tôi tôi muốn đi", "repeat"),
 ("xin lỗi xin lỗi xin lỗi xin lỗi xin lỗi bạn", "vi", "xin lỗi xin lỗi bạn", "repeat"),
]

KNOWN_TRADEOFFS = [  # real speech the filter DOES cut, accepted on purpose (the noise pattern is far more frequent in our data); keep this list honest
 ("你去了吗？没。", "zh", "你去了吗？"),             # a self-answered "没" at the very end: same shape as the noise artefact "…河流里。 没。"
 ("Hmm, let me think about it.", "en", "let me think about it."),   # leading hesitation dropped: no meaning lost
]

def test_known_tradeoffs_documented():
    for t, l, exp in KNOWN_TRADEOFFS: assert clean_asr(t, l)[0] == exp, (t, clean_asr(t, l))

def test_keep():
    bad = [(t, l, clean_asr(t, l)) for t, l in KEEP if clean_asr(t, l)[0] != t.strip() or clean_asr(t, l)[1]]
    assert not bad, f"damaged real speech: {bad}"

def test_change():
    bad = []
    for t, l, exp, tag in CHANGE:
        out, tags = clean_asr(t, l)
        if out != exp or (tag and tag not in tags): bad.append((t, l, out, tags, "expected", exp, tag))
    assert not bad, f"wrong: {bad}"

def test_properties():
    random.seed(0); alpha = list("abcdhkqmnôơàáãạ.,:!? 。，问答好음네") + ["Um.", "Q.", "H.", "xin lỗi", "the", "A.", "嗯。", "그.", "没。", "Có.", "。", "?"]
    for _ in range(20000):
        s = " ".join(random.choice(alpha) for _ in range(random.randint(0, 12))); lang = random.choice(["vi", "en", "zh", "ko"])
        o, tg = clean_asr(s, lang); o2, tg2 = clean_asr(o, lang)
        assert o2 == o, f"not idempotent: {s!r} -> {o!r} -> {o2!r}"                                      # cleaning twice never removes more
        assert len(o) <= len(s.strip()), (s, o)                                                          # never adds text
        assert set(re.findall(r"\w", o.lower())) <= set(re.findall(r"\w", s.lower())), (s, o)            # never invents characters
        if not tg: assert o == s.strip(), (s, o, tg)                                                     # no tag => unchanged

def test_clean_text_unchanged():
    """The human transcripts of the BTC clips are clean speech: the filter may only drop a leading filler, never anything else."""
    p = os.environ.get("BTC_RESULTS", r"C:\Users\Admin\Downloads\outputOnevoice\all_results.csv")
    if not os.path.exists(p): print("(skipped: no BTC results file)"); return
    R = list(csv.DictReader(open(p, encoding="utf-8-sig"))); bad = []
    for r in R:
        s = r["dir"].split("-")[0]; o, tg = clean_asr(r["text"], s)
        if o != r["text"].strip() and set(tg) != {"filler"}: bad.append((r["id"], r["text"], o, tg))        # a leading "um" in a reference transcript may go; nothing else may
    adv = os.environ.get("ADV_ASR", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "_keep", "results", "adv_asr_cpu.json"))
    if os.path.exists(adv):
        import json
        for d, rows in json.load(open(adv, encoding="utf-8")).items():
            for r in rows:
                o, tg = clean_asr(r["text"], d.split("-")[0])
                if o != r["text"].strip() and set(tg) != {"filler"}: bad.append((r["id"], r["text"], o, tg))
    assert not bad, f"{len(bad)} clean transcripts altered: {bad[:5]}"

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("PASS", n)
