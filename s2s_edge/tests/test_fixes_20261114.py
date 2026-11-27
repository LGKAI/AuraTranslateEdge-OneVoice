# -*- coding: utf-8 -*-
"""Fixes of 14/11 (from REPORT DATA CHECK + the review of every changed output on the 1,256 BTC/ADV rows).
Each block has CHANGE cases (what the fix must do) and KEEP cases (inputs that look similar and must NOT change), so a fix that
over-triggers fails here. Pattern tests, not sentence lookups."""
import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from s2s_edge.hybrid_textnorm import normalize, ordinal_en
from s2s_edge.mt import split_sentences, MtRouter, _comma_cut
from s2s_edge.asr_clean import clean_mt
from s2s_edge.conversation import fix_vi_colons


def N(t, l): return normalize(t, l)[0]


class UnitBoundary(unittest.TestCase):
    """A unit must not match the first letter of the next word ("5 lần" was "năm lítần", "15 mét" was "mười lăm métét")."""
    CHANGE = {("5 lần", "vi"): "năm lần", ("3 gói", "vi"): "ba gói", ("2 mẫu", "vi"): "hai mẫu", ("4 lượt", "vi"): "bốn lượt", ("10 lớp", "vi"): "mười lớp",
              ("7 mặt", "vi"): "bảy mặt", ("cao 15 mét", "vi"): "cao mười lăm mét", ("3 lít", "vi"): "ba lít", ("3 mètres", "en"): "three mètres",
              ("Mạng 5G", "vi"): "Mạng năm G", ("The 5G network.", "en"): "The five G network.", ("Over 5M users.", "en"): "Over five M users."}
    KEEP = {("dài 20 m.", "vi"): "dài hai mươi mét.", ("nặng 3 kg", "vi"): "nặng ba ki lô gam", ("1 l nước", "vi"): "một lít nước", ("2 L of water", "en"): "two liters of water",
            ("It weighs 5 g.", "en"): "It weighs five grams.", ("무게는 3.5kg입니다.", "ko"): "무게는 삼 점 오 킬로그램입니다.", ("5kg的大米", "zh"): "五千克的大米",
            ("가격은 15,000원입니다.", "ko"): "가격은 일만오천 원입니다."}

    def test_change(self):
        for (t, l), want in self.CHANGE.items(): self.assertEqual(N(t, l), want, t)

    def test_keep(self):
        for (t, l), want in self.KEEP.items(): self.assertEqual(N(t, l), want, t)


class NewFormats(unittest.TestCase):
    CHANGE = {("We offer 24/7 support.", "en"): "We offer twenty-four seven support.", ("Hỗ trợ 24/7.", "vi"): "Hỗ trợ hai mươi bốn trên bảy.",
              ("Ngày 15-11-2026.", "vi"): "Ngày mười lăm tháng mười một năm hai nghìn không trăm hai mươi sáu.",
              ("Born on 15.11.2026.", "en"): "Born on November fifteenth two thousand twenty-six.", ("15-11-2026", "zh"): "二零二六年十一月十五日",
              ("vào tháng 3/2017", "vi"): "vào tháng ba năm hai nghìn không trăm mười bảy", ("on 3/2017", "en"): "on March two thousand seventeen",
              ("Chờ 10-20 phút.", "vi"): "Chờ mười đến hai mươi phút.", ("Read pages 5-6.", "en"): "Read pages five to six.",
              ("From 2020-2022.", "en"): "From two thousand twenty to two thousand twenty-two.", ("等待10-20分钟。", "zh"): "等待十到二十分钟。",
              ("10-20분 걸립니다.", "ko"): "십에서 이십분 걸립니다.",
              ("The 21st century.", "en"): "The twenty first century.", ("On the 3rd floor.", "en"): "On the third floor.", ("11th grade", "en"): "eleventh grade",
              ("Xe buýt s403", "vi"): "Xe buýt s bốn trăm lẻ ba"}
    KEEP = {("tỉ số 6-6", "vi"): "tỉ số sáu gạch ngang sáu", ("The score was 3-1.", "en"): "The score was three dash one.",          # descending / equal = not a range
            ("Call 555-0123.", "en"): "Call five five five dash zero one two three.", ("2026-11-15出生。", "zh"): "二零二六年十一月十五日出生。",
            ("Số điện thoại 0912 345 678.", "vi"): "Số điện thoại không chín một hai ba bốn năm sáu bảy tám.", ("3-1로 이겼다", "ko"): "삼마이너스 일로 이겼다",
            ("1.500 người", "vi"): N("1.500 người", "vi")}

    def test_change(self):
        for (t, l), want in self.CHANGE.items(): self.assertEqual(N(t, l), want, t)

    def test_keep(self):
        for (t, l), want in self.KEEP.items(): self.assertEqual(N(t, l), want, t)

    def test_ordinal_words(self):
        for n, w in {1: "first", 2: "second", 3: "third", 4: "fourth", 11: "eleventh", 12: "twelfth", 20: "twentieth", 21: "twenty first", 100: "one hundredth",
                     101: "one hundred first", 1000: "one thousandth"}.items():
            self.assertEqual(ordinal_en(n), w, n)
        self.assertNotIn("ieth", ordinal_en(100))

    def test_range_needs_evidence(self):
        for t, l in [("thua 3-5 trước Nhật Bản", "vi"), ("They lost 3-5 to Japan.", "en"), ("The final score was 1-2 today.", "en"), ("3-5로 졌다", "ko"), ("比分1-3落后", "zh")]:
            self.assertFalse(any(s["kind"] == "range" for s in normalize(t, l)[1]["spans"]), t)
        self.assertTrue(normalize("3-5", "en")[1]["warnings"])                       # bare: ambiguous, reported (repo rule)
        for t, l in [("Wait 10-20 minutes.", "en"), ("Từ 2020-2022.", "vi"), ("Read pages 5-6.", "en"), ("10-20분", "ko")]:
            self.assertTrue(any(s["kind"] == "range" for s in normalize(t, l)[1]["spans"]), t)

    def test_identifier_touching_cjk(self):
        self.assertEqual(N("航班号是VN247", "zh"), "航班号是维恩二四七")
        self.assertEqual(N("riêng bạnwork2002", "vi"), "riêng bạnwork hai nghìn không trăm lẻ hai")   # inside a Vietnamese word: no identifier

    def test_ordinal_only_english(self):
        self.assertNotIn("first", N("1st", "vi"))


class KoreanNative(unittest.TestCase):
    CHANGE = {"8시간마다 2개의 약": "여덟 시간마다 두 개의 약", "3살의 쿠모는": "세 살의 쿠모는", "3시 30분입니다.": "세 시 삼십분입니다.", "20개": "스무 개", "21개": "스물한 개",
              "99개": "아흔아홉 개", "3번째 층입니다.": "세 번째 층입니다.", "2명": "두 명", "12시간": "열두 시간"}
    KEEP = {"13시에 만나요": "십삼시에 만나요", "100개": "백개", "2개월": "이개월", "3번 출구": "삼번 출구", "1만 명": "일만 명", "6월 초": "육월 초", "5달러": N("5달러", "ko"),
            "1,500명입니다.": "천오백명입니다.", "2026년 11월 15일": N("2026년 11월 15일", "ko"), "5전 연속 패배": "오전 연속 패배"}

    def test_change(self):
        for t, want in self.CHANGE.items(): self.assertEqual(N(t, "ko"), want, t)

    def test_keep(self):
        for t, want in self.KEEP.items(): self.assertEqual(N(t, "ko"), want, t)

    def test_5dollar_not_native(self):
        self.assertNotIn("다섯", N("5달러", "ko"))


class SplitInitials(unittest.TestCase):
    def test_change(self):
        self.assertEqual(split_sentences("Hi, my name is M. James Johnson.", "en"), ["Hi, my name is M. James Johnson."])
        self.assertEqual(split_sentences("Xin chào. Tôi là Bamela F. Fagj, giáo sư.", "vi"), ["Xin chào.", "Tôi là Bamela F. Fagj, giáo sư."])

    def test_keep(self):
        for t, want in {"Plan B. We go now.": ["Plan B.", "We go now."], "So did I. Then we left.": ["So did I.", "Then we left."],
                        "My name is Sarah Smith. Nice to meet you.": ["My name is Sarah Smith.", "Nice to meet you."],
                        "Đứng tập trung vào giữa. D. Tôi đi.": ["Đứng tập trung vào giữa. D.", "Tôi đi."]}.items():
            self.assertEqual(split_sentences(t, "vi" if "Đ" in t else "en"), want, t)


class VietColons(unittest.TestCase):
    def test_change(self):
        for t, want in {"Tôi: Anh không tiêu phải loại thuốc nào ạ?": "Tôi, anh không tiêu phải loại thuốc nào ạ?", "Nó vừa vặn với tôi:": "Nó vừa vặn với tôi.",
                        "Thư: Nam đến chưa?": "Thư, Nam đến chưa?", "Lưu ý: không hút thuốc.": "Lưu ý, không hút thuốc."}.items():
            self.assertEqual(fix_vi_colons(t), want)

    def test_keep(self):
        for t in ("Họp lúc 3:30 chiều.", "Tỉ số là 2:1.", "Không có dấu hai chấm.", "Mang theo: áo, mũ, giày.",
                  "Phòng đang dọn: Khoảng 2 phút nữa là xong.", "Tôi hỏi thêm một câu nữa: Bế xe ngoại tỉnh nằm ở đâu ạ?",   # not a label, not dangling (BTC VE-6066, VK-4594)
                  "thông báo bằng nhiều thứ tiếng gồm: tiếng Tây Ban Nha, tiếng Anh, tiếng Pháp."): self.assertEqual(fix_vi_colons(t), t)   # list colons (VK_ADV_006)


class MtParentheses(unittest.TestCase):
    def test_change(self):
        self.assertEqual(clean_mt("도니 대학교 (도니 대학교) 의 파멜라 파고 (파멜라 파고지) 교수", "x", "ko")[0], "도니 대학교 의 파멜라 파고 교수")
        self.assertEqual(clean_mt("唐尼大学的教授帕梅拉·法戈 (帕梅拉·法戈) 指出", "x", "zh")[0], "唐尼大学的教授帕梅拉·法戈 指出")

    def test_keep(self):
        for t, src, l in [("Đảng South West Africa People's Organization (SWAPO) thắng.", "x", "vi"), ('"시 (se)"라는 단어', "x", "ko"),
                          ("Giá (đã gồm thuế) là 5 đô.", "x", "vi"), ("도니 대학교 (도니 대학교)", "Đại học (Dony)", "ko")]:
            self.assertNotIn("mt_paren", clean_mt(t, src, l)[1], t)


class ParenAfterNames(unittest.TestCase):
    """Real pipeline order: NLLB writes "도니 대학교 (Dony University)"; the duplicate only exists after the name transliteration (VK_ADV_018)."""
    def _engine(self, sent_out, names):
        from s2s_edge import conversation as C
        class M:
            S = 64; last = "mt33"
            def __call__(self, x, s, t): return names.get(x, x) if s == "en" else sent_out
        class R:
            glossary = {}; profile = {}; source_mode = "0"
            def __init__(self): self.m = M()
            def mt(self): return self.m
        return C.Engine(R())

    def test_duplicate_after_transliteration_dropped(self):
        e = self._engine("도니 대학교 (Dony University) 의 교수는 말했다.", {"Dony University": "도니 대학교"})
        out, tags = e.translate_text_tagged("Giáo sư của Đại học Dony nói.", "vi", "ko")
        self.assertNotIn("(", out); self.assertIn("도니 대학교", out); self.assertIn("mt_paren", tags)

    def test_acronym_kept(self):
        e = self._engine("남서아프리카인민기구 (SWAPO) 가 이겼다.", {"SWAPO": "스와포"})
        out, _ = e.translate_text_tagged("Tổ chức SWAPO thắng.", "vi", "ko")
        self.assertIn("(", out)


class _Fake:
    def __init__(self, outs): self.outs = list(outs); self.lp = 1.0; self.seen_lp = []
    def __call__(self, text, src, tgt): self.seen_lp.append(self.lp); return self.outs.pop(0)


class CommaCut(unittest.TestCase):
    def test_detect(self):
        self.assertTrue(_comma_cut("Tôi đi học.", "我去上学,"))
        for s, o in [("Tôi đi học,", "我去上学,"), ("Tôi đi học.", "我去上学。"), ("", ","), ("Xin chào.", "")]: self.assertFalse(_comma_cut(s, o))

    def test_lp_retry_wins(self):
        big, small = _Fake(["我去上学,", "我去上学了。"]), _Fake(["X"])
        r = MtRouter(small, big, log=lambda m: None)
        self.assertEqual(r("Tôi đi học.", "vi", "zh"), "我去上学了。"); self.assertEqual(r.last, "mt33_lp1.6")
        self.assertEqual(big.seen_lp, [1.0, 1.6]); self.assertEqual(big.lp, 1.0); self.assertEqual(small.outs, ["X"])

    def test_600m_fallback(self):
        big, small = _Fake(["我去上学,", "我去,"]), _Fake(["我去上学了很久。"])
        r = MtRouter(small, big, log=lambda m: None)
        self.assertEqual(r("Tôi đi học.", "vi", "zh"), "我去上学了很久。"); self.assertEqual(r.last, "mt600_cut_fallback")

    def test_complete_but_not_longer_is_rejected(self):
        """VM_ADV_008: 600M was complete but shorter (dropped a clause, invented a date) -> keep the 3.3B output."""
        big, small = _Fake(["这份文件提及边界冲突, 巴莱蒂希望以边界为依据,", "这份文件提及边界冲突,"]), _Fake(["巴莱蒂希望在9月6日中东战争之前建立的边境冲突."])
        r = MtRouter(small, big, log=lambda m: None)
        self.assertEqual(r("Hồ sơ nhắc tới xung đột biên giới.", "vi", "zh"), "这份文件提及边界冲突, 巴莱蒂希望以边界为依据,"); self.assertEqual(r.last, "mt33")

    def test_all_cut_keeps_longest(self):
        big, small = _Fake(["我去上学,", "我,"]), _Fake(["我去,"])
        r = MtRouter(small, big, log=lambda m: None)
        self.assertEqual(r("Tôi đi học.", "vi", "zh"), "我去上学,")

    def test_no_retry_when_complete(self):
        big, small = _Fake(["我去上学。"]), _Fake([])
        r = MtRouter(small, big, log=lambda m: None)
        self.assertEqual(r("Tôi đi học.", "vi", "zh"), "我去上学。"); self.assertEqual(big.seen_lp, [1.0]); self.assertEqual(r.last, "mt33")


if __name__ == "__main__":
    unittest.main()
