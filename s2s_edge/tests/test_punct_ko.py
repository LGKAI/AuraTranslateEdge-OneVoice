# -*- coding: utf-8 -*-
"""Korean punctuation model (s2s_edge/punct_ko.py) and conversation.restore_ko.
CHANGE = what the model must add; KEEP = what must survive (the ASR's final '?', numbers, non-Korean text, missing model files)."""
import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from s2s_edge import conversation as C
from s2s_edge import punct_ko
from s2s_edge.registry import ASSETS
HAVE = os.path.exists(os.path.join(ASSETS, "ko_punct", "model_quantized.onnx"))


class Rules(unittest.TestCase):
    def test_strip_marks_keeps_numbers(self):
        self.assertEqual(punct_ko.strip_marks("3.5kg, 1,000원. 10:30?"), "3.5kg 1,000원 10:30")

    def test_missing_model_keeps_text(self):
        saved = punct_ko._INSTANCE; punct_ko._INSTANCE = None
        real = punct_ko.get
        try:
            punct_ko.get = lambda d: None
            self.assertEqual(C.restore_ko("남향방 없나요?"), "남향방 없나요?")
        finally:
            punct_ko.get = real; punct_ko._INSTANCE = saved

    def test_policy_with_fake_model(self):
        class Fake:
            def __init__(self, out): self.out = out
            def __call__(self, t): return self.out
        real = punct_ko.get
        try:
            for raw, model_out, want in [("뭐 하세요?", "뭐 하세요.", "뭐 하세요?"),                 # ASR question mark kept
                                         ("안녕하세요 저는 민수입니다", "안녕하세요, 저는 민수입니다", "안녕하세요, 저는 민수입니다."),
                                         ("회의 안건 예산", "회의 안건: 예산", "회의 안건, 예산."),     # ':' -> ','
                                         ("가 나 다", "가 - 나 다.", "가 나 다.")]:                  # '-' dropped
                punct_ko.get = lambda d, o=model_out: Fake(o)
                self.assertEqual(C.restore_ko(raw), want, raw)
        finally:
            punct_ko.get = real

    def test_non_korean_untouched(self):
        for t in ("hello world", "", "123"): self.assertEqual(C.restore_ko(t), t)


@unittest.skipUnless(HAVE, "assets/ko_punct not in this checkout")
class Model(unittest.TestCase):
    def test_adds_sentence_break_and_keeps_question(self):
        out = C.restore_ko("죄송합니다 시내에 가면 어디에 있는지 알 수 있을까요?")
        self.assertTrue(out.startswith("죄송합니다.") and out.endswith("?"), out)
        out = C.restore_ko("시 간이 있 있을 때 뭐 하 세하세요?")
        self.assertTrue(out.endswith("?"), out)

    def test_numbers_survive(self):
        out = C.restore_ko("무게는 3.5kg입니다 1,000원이에요")
        self.assertIn("3.5kg", out); self.assertIn("1,000원", out)

    def test_tokenizer_parity(self):
        try: from tokenizers import Tokenizer
        except ImportError: self.skipTest("tokenizers library not installed (box): parity checked on the laptop")
        ref = Tokenizer.from_file(os.path.join(ASSETS, "ko_punct", "tokenizer.json")); mine = punct_ko.get(ASSETS).tok
        for t in ["안녕하세요 😀 테스트 ㅋㅋ", "지금 바로 2번 생산 라인을 멈추고", "Xin chào các bạn", "你好世界", "3.5kg 1,000원"]:
            self.assertEqual(ref.encode(t, add_special_tokens=False).ids, [i for _, ids in mine.words(t) for i in ids], t)


if __name__ == "__main__":
    unittest.main()
