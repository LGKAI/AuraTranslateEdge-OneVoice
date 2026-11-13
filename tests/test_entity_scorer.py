"""Regression tests for typed, boundary-aware semantic entity scorer (R5)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

import ablate_profiles
import entity_scorer


class TypedEntityScorerTests(unittest.TestCase):
    def test_boundary_prevents_32_matching_132_and_3_matching_330(self):
        res_32 = entity_scorer.score_text_entities(
            written="I am 32 years old.",
            asr_text="I am 132 years old.",
            expected_entities=["32"],
            lang="en",
        )
        self.assertEqual(res_32["correct"], 0)
        self.assertEqual(res_32["wrong"], 1)
        self.assertEqual(res_32["unknown"], 0)

        res_3 = entity_scorer.score_text_entities(
            written="Work till 3 o'clock.",
            asr_text="Work till 330 or 3:30 PM.",
            expected_entities=["3"],
            lang="en",
        )
        self.assertEqual(res_3["correct"], 0)
        self.assertEqual(res_3["wrong"], 1)

    def test_leading_zeros_preserved(self):
        ok = entity_scorer.score_text_entities(
            written="Call 0912345678 and ZIP 04782.",
            asr_text="Call 0912 345 678 and ZIP 04782.",
            expected_entities=["0912345678", "04782"],
            lang="en",
        )
        self.assertEqual(ok["correct"], 2)
        self.assertEqual(ok["wrong"], 0)

        dropped_zero = entity_scorer.score_text_entities(
            written="Call 0912345678 and ZIP 04782.",
            asr_text="Call 912345678 and ZIP 4782.",
            expected_entities=["0912345678", "04782"],
            lang="en",
        )
        self.assertEqual(dropped_zero["correct"], 0)
        self.assertEqual(dropped_zero["wrong"], 2)

    def test_sign_decimal_unit_currency_and_meridiem(self):
        # Sign mismatch (-12 vs 12)
        sign_bad = entity_scorer.score_text_entities(
            written="Temperature -12°C",
            asr_text="Temperature 12°C",
            expected_entities=["-12°C"],
            lang="en",
        )
        self.assertEqual(sign_bad["wrong"], 1)
        self.assertEqual(sign_bad["correct"], 0)

        # Decimal & unit equivalence vs unit mismatch
        temp_ok = entity_scorer.score_text_entities(
            written="Temperature 37.5°C",
            asr_text="Temperature 37.5 degrees Celsius",
            expected_entities=["37.5°C"],
            lang="en",
        )
        self.assertEqual(temp_ok["correct"], 1)

        temp_wrong_unit = entity_scorer.score_text_entities(
            written="Temperature 37.5°C",
            asr_text="Temperature 37.5°F",
            expected_entities=["37.5°C"],
            lang="en",
        )
        self.assertEqual(temp_wrong_unit["wrong"], 1)

        temp_truncated = entity_scorer.score_text_entities(
            written="Temperature 37.5°C",
            asr_text="Temperature 37°C",
            expected_entities=["37.5°C"],
            lang="en",
        )
        self.assertEqual(temp_truncated["wrong"], 1)

        # Meridiem mismatch (3:30 PM vs 3:30 AM)
        time_ok = entity_scorer.score_text_entities(
            written="Meeting at 3:30 PM.",
            asr_text="Meeting at 330 pm.",
            expected_entities=["3:30 PM"],
            lang="en",
        )
        self.assertEqual(time_ok["correct"], 1)

        time_bad = entity_scorer.score_text_entities(
            written="Meeting at 3:30 PM.",
            asr_text="Meeting at 3:30 AM.",
            expected_entities=["3:30 PM"],
            lang="en",
        )
        self.assertEqual(time_bad["wrong"], 1)

        # Currency value precision ($15.50 vs $15)
        cur_bad = entity_scorer.score_text_entities(
            written="Ticket costs $15.50.",
            asr_text="Ticket costs $15.",
            expected_entities=["$15.50"],
            lang="en",
        )
        self.assertEqual(cur_bad["wrong"], 1)

    def test_duplicate_entities_require_multiplicity(self):
        one_seen = entity_scorer.score_text_entities(
            written="Buy 3 apples and 3 oranges.",
            asr_text="Buy 3 apples and oranges.",
            expected_entities=["3", "3"],
            lang="en",
        )
        self.assertEqual(one_seen["correct"], 1)
        self.assertEqual(one_seen["wrong"], 1)

        two_seen = entity_scorer.score_text_entities(
            written="Buy 3 apples and 3 oranges.",
            asr_text="Buy 3 apples and 3 oranges.",
            expected_entities=["3", "3"],
            lang="en",
        )
        self.assertEqual(two_seen["correct"], 2)
        self.assertEqual(two_seen["wrong"], 0)

    def test_missing_flight_date_time_labels_and_unparsed_spoken_stay_unknown(self):
        # Missing flight, date, time labels must drop coverage and produce unknown, NEVER pass
        missing_labels = entity_scorer.score_text_entities(
            written="Flight VN247 is on 21/12/2027 at 3:30 PM for 20 passengers.",
            asr_text="Flight V and 247 is on December 21, 2027 at 3:30 PM for 20 passengers.",
            expected_entities=["20"],
            lang="en",
        )
        self.assertLess(missing_labels["coverage"], 0.5)
        self.assertEqual(missing_labels["correct"], 1)
        self.assertEqual(missing_labels["unknown"], 3)
        self.assertFalse(missing_labels["all_labeled_correct"])
        reasons = {it["reason"] for it in missing_labels["items"] if it["status"] == "unknown"}
        self.assertEqual(reasons, {"missing_flight_id_label", "missing_date_label", "missing_time_label"})

        # Unparsed spoken ASR must stay unknown, never forced pass/fail
        spoken_en = entity_scorer.score_text_entities(
            written="thirty two years old",
            asr_text="thirty two years old",
            expected_entities=["32"],
            lang="en",
        )
        self.assertEqual(spoken_en["correct"], 0)
        self.assertEqual(spoken_en["wrong"], 0)
        self.assertEqual(spoken_en["unknown"], 1)
        self.assertEqual(spoken_en["items"][0]["reason"], "unparsed_spoken")

        spoken_vi = entity_scorer.score_text_entities(
            written="Cuộc họp lúc 14h45, gọi số 0912345678.",
            asr_text="cuộc họp lúc mười bốn giờ bốn lăm gọi số không chín một hai ba bốn năm sáu bảy tám",
            expected_entities=["14h45", "0912345678"],
            lang="vi",
        )
        self.assertEqual(spoken_vi["correct"], 0)
        self.assertEqual(spoken_vi["wrong"], 0)
        self.assertEqual(spoken_vi["unknown"], 2)

    def test_lexical_proxy_is_explicitly_marked_non_semantic(self):
        lex = ablate_profiles.check_lexical_proxy("number 132", ["32"])
        self.assertFalse(lex["is_semantic_gate"])
        self.assertEqual(lex["metric_role"], "lexical_presence_proxy_non_semantic")

    def test_real_upstream_zip_fragmentation_is_wrong_while_legacy_is_correct(self):
        leg = entity_scorer.score_text_entities(
            written="734 Main Street, Harper Valley Oregon, 67297.",
            asr_text="734 Main Street, Harper Valley Oregon,67297.",
            expected_entities=["734", "67297"],
            lang="en",
        )
        self.assertEqual(leg["correct"], 2)
        self.assertEqual(leg["wrong"], 0)

        up = entity_scorer.score_text_entities(
            written="734 Main Street, Harper Valley Oregon, 67297.",
            asr_text="734 Main Street. | Harper Valley Oregon. | Six,7,29,7.",
            expected_entities=["734", "67297"],
            lang="en",
        )
        self.assertEqual(up["correct"], 1)
        self.assertEqual(up["wrong"], 1)
        self.assertEqual(up["unknown"], 0)


if __name__ == "__main__":
    unittest.main()
