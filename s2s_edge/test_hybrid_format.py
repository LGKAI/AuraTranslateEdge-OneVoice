"""Semantic examples and generated boundary checks, independent of audio models."""
import json
import random
import sys
import unittest
import unicodedata
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'s2s_edge'))
from s2s_edge.hybrid_textnorm import HybridNormalizer
from s2s_edge.tts_format import chunks


EXAMPLES = [
    ('vi', '5:30 sáng', 'năm giờ ba mươi phút sáng'),
    ('vi', '08:00', 'tám giờ'),
    ('vi', '12:00 PM', 'mười hai giờ trưa'),
    ('vi', 'Hẹn ngày 21/12/2027', 'Hẹn ngày hai mươi mốt tháng mười hai năm hai nghìn không trăm hai mươi bảy'),
    ('vi', '14h45', 'mười bốn giờ bốn mươi lăm phút'),
    ('vi', 'Số nhà 734', 'Số nhà bảy trăm ba mươi bốn'),
    ('vi', 'ZIP 67297', 'ZIP sáu bảy hai chín bảy'),
    ('vi', 'Mã ZIP 04782', 'Mã ZIP không bốn bảy tám hai'),
    ('vi', 'Điện thoại 0908 246 135', 'Điện thoại không chín không tám hai bốn sáu một ba năm'),
    ('vi', 'Chuyến bay VN247', 'Chuyến bay vê en hai bốn bảy'),
    ('vi', '330 đồng', 'ba trăm ba mươi đồng'),
    ('vi', '36,8°C', 'ba mươi sáu phẩy tám độ xê'),
    ('vi', '-12,50%', 'âm mười hai phẩy năm không phần trăm'),
    ('vi', '$18.25', 'mười tám đô la hai mươi lăm xen'),
    ('vi', '75.000 đồng', 'bảy mươi lăm nghìn đồng'),
    ('vi', '21/12/2027', 'ngày hai mươi mốt tháng mười hai năm hai nghìn không trăm hai mươi bảy'),
    ('vi', '21 tháng 12 năm 2027', 'ngày hai mươi mốt tháng mười hai năm hai nghìn không trăm hai mươi bảy'),
    ('en', '3:30 PM', 'three thirty pee em'),
    ('en', '8:05 AM', 'eight oh five ay em'),
    ('en', '08:00', "eight o'clock"),
    ('en', '$18.25', 'eighteen dollars twenty-five cents'),
    ('en', '$1.01', 'one dollar one cent'),
    ('en', '21/12/2027', 'December twenty first two thousand twenty-seven'),
    ('en', 'December 21, 2027', 'December twenty first two thousand twenty-seven'),
    ('en', 'ZIP 04782', 'ZIP zero four seven eight two'),
    ('en', 'Phone 0908246135', 'Phone zero nine zero eight two four six one three five'),
    ('en', 'Flight VN247', 'Flight vee en two four seven'),
    ('en', 'Code AB-012', 'Code ay bee dash zero one two'),
    ('en', '330 dollars', 'three hundred thirty dollars'),
    ('en', 'shopping 330', 'shopping three hundred thirty'),
    ('en', '36.8°C', 'thirty-six point eight degrees Celsius'),
    ('en', '12.5%', 'twelve point five percent'),
    ('en', '1kg', 'one kilogram'),
    ('en', 'Phone +84-908-246-135', 'Phone plus eight four dash nine zero eight dash two four six dash one three five'),
    ('en', '1,234.50', 'one thousand two hundred thirty-four point five zero'),
    ('ko', '3:30', '세 시 삼십 분'),
    ('ko', '08:00', '여덟 시'),
    ('ko', '18.25달러', '십팔 달러 이십오 센트'),
    ('ko', '36.8°C', '삼십육 점 팔 도'),
    ('ko', '우편번호는 04782', '우편번호는 영 사 칠 팔 이'),
    ('ko', '항공편은 VN247', '항공편은 브이 엔 이 사 칠'),
    ('ko', '2027년 12월 21일', '이천이십칠 년 십이 월 이십일 일'),
    ('zh', '3:30', '三点三十分'),
    ('zh', '08:00', '八点'),
    ('zh', '2027年12月21日3:30', '二零二七年十二月二十一日三点三十分'),
    ('zh', '2027年', '二零二七年'),
    ('zh', '18.25美元', '十八美元二十五美分'),
    ('zh', '36.8°C', '三十六点八摄氏度'),
    ('zh', '12.5%', '百分之十二点五'),
    ('zh', '邮政编码是04782', '邮政编码是零四七八二'),
    ('zh', '航班号是VN247', '航班号是维恩二四七'),
    ('zh', '1000000000000', '一万亿'),
    ('zh', '1000000001000', '一万亿零一千'),
    ('zh', '０８：００', '八点'),
    ('zh', '３６．８°C', '三十六点八摄氏度'),
]


class FormatTests(unittest.TestCase):
    def setUp(self): self.h = HybridNormalizer(learned_identifiers=False)

    def test_semantic_examples(self):
        for lang, text, expected in EXAMPLES:
            with self.subTest(lang=lang, text=text):
                output, _ = self.h(text, lang)
                self.assertEqual(output, expected)

    def test_dates_ambiguous_and_invalid(self):
        for text in ['03/04/2027', '31/02/2027', '2027-13-01', '29/02/2027', '03/04/27']:
            with self.subTest(text=text):
                _, r = self.h(text, 'en')
                self.assertTrue(r['warnings'])
                self.assertEqual(r['spans'][0]['kind'], 'unresolved')
        dmy = HybridNormalizer(date_order='DMY', learned_identifiers=False)
        mdy = HybridNormalizer(date_order='MDY', learned_identifiers=False)
        self.assertIn('April third', dmy('03/04/2027', 'en')[0])
        self.assertIn('March fourth', mdy('03/04/2027', 'en')[0])

    def test_invalid_times(self):
        for raw in ['24:00', '12:60', '15:30 PM', '08:15:61']:
            with self.subTest(raw=raw): self.assertTrue(self.h(raw, 'en')[1]['warnings'])

    def test_unsupported_notation_is_reported(self):
        for raw in ['1e-3', '3-5']:
            with self.subTest(raw=raw): self.assertTrue(self.h(raw, 'en')[1]['warnings'])

    def test_generated_dates(self):
        rng = random.Random(625)
        for _ in range(100):
            d = date(2000, 1, 1)+timedelta(days=rng.randrange(15000))
            for lang in ('vi', 'en', 'ko', 'zh'):
                _, r = self.h(d.isoformat(), lang)
                value = r['spans'][0]['value']
                self.assertEqual((value['year'], value['month'], value['day']), (d.year, d.month, d.day))

    def test_generated_times_and_decimals(self):
        rng = random.Random(626)
        for _ in range(100):
            h, m, n, frac = rng.randrange(24), rng.randrange(60), rng.randrange(100000), str(rng.randrange(100)).zfill(2)
            for lang in ('vi', 'en', 'ko', 'zh'):
                _, r = self.h(f'{h:02}:{m:02}', lang)
                self.assertEqual((r['spans'][0]['value']['hour'], r['spans'][0]['value']['minute']), (h, m))
                sep = ',' if lang == 'vi' else '.'
                _, r = self.h(f'{n}{sep}{frac}', lang)
                self.assertEqual(r['spans'][0]['value'], {'sign':'', 'whole':str(n), 'fraction':frac})

    def test_generated_identifiers(self):
        rng = random.Random(627)
        cues = dict(vi='Mã ', en='Code ', ko='항공편 ', zh='航班号')
        for _ in range(100):
            raw = 'AB0'+str(rng.randrange(100000)).zfill(5)
            for lang in cues:
                output, r = self.h(cues[lang]+raw, lang)
                self.assertEqual(r['spans'][0]['value']['characters'], raw)
                self.assertFalse(any(c.isdecimal() for c in output))

    def test_chunks_do_not_split_words(self):
        text = 'My phone number is zero nine zero eight two four six one three five. The price is eighteen dollars twenty-five cents.'
        parts = chunks(text, 'en', 45, 'char')
        self.assertEqual(' '.join(parts), text)
        self.assertTrue(all(len(p) <= 45 for p in parts))
        self.assertRaises(ValueError, chunks, 'x'*100, 'en', 45, 'char')

    def test_spans_and_untouched_text(self):
        for lang in ('vi', 'en', 'ko', 'zh'):
            text = 'prefix 3:30 suffix $18.25 end'
            _, record = self.h(text, lang)
            end = 0
            for span in record['spans']:
                self.assertGreaterEqual(span['start'], end)
                self.assertEqual(text[span['start']:span['end']], span['raw'])
                end = span['end']
            self.assertTrue(record['output'].startswith('prefix '))
            self.assertTrue(record['output'].endswith(' end'))

    def test_korean_post_preprocessing_budget(self):
        text = '회의는 오후 세 시 삼십 분에 있습니다. 전화번호는 영 구 영 팔 이 사 육 일 삼 오입니다.'
        counter = lambda s: len(unicodedata.normalize('NFKD', s))+10
        parts = chunks(text, 'ko', 96, 'char', counter=counter)
        self.assertTrue(all(counter(p) <= 96 for p in parts))
        self.assertEqual(' '.join(parts), text)


if __name__ == '__main__': unittest.main()
