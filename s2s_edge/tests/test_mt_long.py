# -*- coding: utf-8 -*-
"""Long input on the chip graphs (input S=64 -> 62 source tokens; decoder stops at Tm-1 / 2.2 x source + 14 tokens). Nllb._tr must:
  * fit: translate in one call;  * too long or decoder did not finish: split by sentence, then by clause ("," ":" + space), then in halves, recursively;
  * never lose a piece, keep order;  * never split inside "15.000", "36.5", "1,000", "7:30";  * count what it could only truncate.
The graph is faked: 1 token per word, the fake 'translation' is the upper-cased input, a call "finishes" only if the input fits FIT tokens."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from s2s_edge import mt as M

class Fake(M.Nllb):
    def __init__(self, S=12, FIT=99):
        self.S, self.FIT, self.calls, self.truncations, self.seen = S, FIT, 0, 0, []
    def encode(self, text): return text.split()
    def decode(self, ids): return " ".join(ids)
    def _run(self, toks, src, tgt):
        self.calls += 1; self.seen.append(" ".join(toks)); body = [w.upper() for w in toks]
        return (body + [2], True) if len(toks) <= self.FIT else (body[: self.FIT], False)      # decoder ran out of steps: no EOS

def words(t): return [w for w in t.replace(",", " ").replace(".", " ").split()]

def test_fits_one_call():
    f = Fake(); assert f("xin chào các bạn.", "vi", "en") == "XIN CHÀO CÁC BẠN." and f.calls == 1

def test_too_long_split_by_clause_nothing_lost():
    t = "một hai ba bốn năm sáu, bảy tám chín mười mười_một, mười_hai mười_ba mười_bốn mười_lăm."
    f = Fake(S=12); out = f(t, "vi", "en")
    assert words(out) == [w.upper() for w in words(t)], out                 # every word, in order
    assert all(len(s.split()) <= 10 for s in f.seen) and f.truncations == 0, f.seen

def test_no_clause_mark_halves():
    t = " ".join(f"w{i}" for i in range(30)); f = Fake(S=12); out = f(t, "en", "vi")
    assert out.split() == [w.upper() for w in t.split()] and f.truncations == 0, (out, f.seen)

def test_decoder_did_not_finish_resplits():
    t = "one two three four five six, seven eight nine ten."
    f = Fake(S=64, FIT=6); out = f(t, "en", "vi")                           # fits the input, but the decoder stops after 6 tokens
    assert words(out) == [w.upper() for w in words(t)], (out, f.seen)

def test_numbers_not_split():
    f = Fake(S=7)                                                            # forces splitting
    out = f("giá là 15.000 đồng, nhiệt độ 36.5 độ, lúc 7:30 sáng, tổng 1,000 won", "vi", "en")
    for tok in ("15.000", "36.5", "7:30", "1,000"): assert any(tok in s for s in f.seen), (tok, f.seen)
    assert "15.000" in out and "36.5" in out and "7:30" in out and "1,000" in out, out

def test_unsplittable_counts_truncation():
    f = Fake(S=4); out = f("supercalifragilistic", "en", "vi")              # one word cannot be split: accepted as is, counted
    assert out == "SUPERCALIFRAGILISTIC" and f.truncations == 0
    f = Fake(S=64, FIT=0); f("a", "en", "vi"); assert f.truncations == 1     # decoder never finishes even for one word: counted, not hidden

if __name__ == "__main__":
    for k, fn in list(globals().items()):
        if k.startswith("test_"): fn(); print("ok", k)
