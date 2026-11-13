"""Identifier-policy regression checks, independent of an acoustic model."""
import unittest
from s2s_edge.identifiers import find_identifiers, prepare_for_tts, spell
from s2s_edge.textnorm import normalize

class IdentifierTests(unittest.TestCase):
    def test_spelling(self):
        for value, lang, expected in [
            ("VN123", "vi", "vê en một hai ba"),
            ("AB-0012", "vi", "a bê không không một hai"),
            ("67297", "vi", "sáu bảy hai chín bảy"),
            ("0912 345 678", "vi", "không chín một hai ba bốn năm sáu bảy tám"),
            ("+84 912 345 678", "vi", "cộng tám bốn chín một hai ba bốn năm sáu bảy tám"),
            ("VN123", "en", "v n one two three"),
            ("555-0123", "en", "five five five zero one two three"),
        ]:
            with self.subTest(value=value, lang=lang):
                self.assertEqual(spell(value, lang), expected)

    def test_recognizes_identifier_classes(self):
        for text, expected in [
            ("Chuyến bay VN123, mã bưu chính 67297.", ["VN123", "67297"]),
            ("Gọi 0912 345 678.", ["0912 345 678"]),
            ("Gọi 0912\u00a0345\u00a0678.", ["0912\u00a0345\u00a0678"]),
            ("Gọi số khẩn cấp 114.", ["114"]),
            ("Call emergency number 911.", ["911"]),
            ("Call +1 202 555 0123.", ["+1 202 555 0123"]),
            ("Flight VN123, ZIP code 67297.", ["VN123", "67297"]),
            ("The ZIP code is 67297.", ["67297"]),
            ("Mật khẩu là 001204.", ["001204"]),
            ("Order AB-0012.", ["AB-0012"]),
            ("Serial number X9Q7 and code 0042.", ["X9Q7", "0042"]),
            ("Flight number VN123.", ["VN123"]),
        ]:
            with self.subTest(text=text):
                self.assertEqual([s.value for s in find_identifiers(text)], expected)

    def test_does_not_spell_quantities_or_other_types(self):
        for text in [
            "Có 67297 người.", "There are 67297 participants.", "Phòng 204.",
            "Năm 1990.", "50.000 đồng, 15%, 37,5°C, 10km.", "14h45 và 3:30.",
            "21st visit; coefficient 1e-3.", "15/11/2026 và 2026-11-15.",
            "Email a1@example.com; https://example.com/a2.", "Giá trị 0.123456789.",
        ]:
            with self.subTest(text=text):
                self.assertEqual(find_identifiers(text), [])

    def test_foreign_name_step_cannot_damage_ids(self):
        seen = []
        def foreign(s):
            seen.append(s)
            return s.replace("Johnson", "giôn sơn")
        output = prepare_for_tts("Johnson, chuyến bay VN123.", "vi", normalize, foreign)
        self.assertEqual(output, "giôn sơn, chuyến bay vê en một hai ba.")
        self.assertFalse(any("VN123" in s for s in seen))

    def test_normalizer_cannot_change_protected_values(self):
        for method in (normalize, lambda s, lang: s):
            self.assertIn("zero zero one two zero four", prepare_for_tts("The PIN is 001204.", "en", method))

if __name__ == "__main__":
    unittest.main()
