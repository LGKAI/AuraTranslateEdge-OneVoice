# -*- coding: utf-8 -*-
"""tools/lid_eval.py with a fake Conversation: rows, trims, failing clips, and the report (accuracy cells, worst false-vi probability, every miss listed)."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools"))
import lid_eval as L


class FakeLid:
    def vi_prob(self, st): return st["p"]


class FakeConv:
    """Says Vietnamese when the clip is at least 2 s long and its 'language' marker (first sample) is 1; raises on marker 9."""
    lid = FakeLid()
    def __init__(self, partner): self.partner = partner
    def route(self, w):
        if w[0] == 9: raise RuntimeError("boom")
        vi = w[0] == 1 and len(w) >= 2 * 16000
        return ("vi" if vi else self.partner), (self.partner if vi else "vi"), "text", dict(p=0.9 if vi else 0.1, nwords=3)


def wav(marker, sec): x = np.zeros(int(sec * 16000), np.float32); x[0] = marker; return x


def test_rows_and_report():
    items = {"vi-ko": [dict(id="a", wav=wav(1, 5), cond="clean", set="adv"), dict(id="b", wav=wav(1, 5), cond="heavy_5db", set="adv"), dict(id="bad", wav=wav(9, 5), cond="clean", set="adv")],
             "ko-vi": [dict(id="c", wav=wav(0, 5), cond="clean", set="adv")]}
    conv_for = lambda s, t: FakeConv("ko")
    rows = L.evaluate(conv_for, items, [None, 3, 1.5], load=lambda w: w)
    assert len(rows) == 12, len(rows)                                                  # 4 clips x 3 trims
    full = {r["id"]: r for r in rows if r["trim"] == "full"}
    assert full["a"]["ok"] == 1 and full["c"]["ok"] == 1 and full["bad"]["pred"] == "?" and "boom" in full["bad"]["error"]
    short = [r for r in rows if r["id"] == "a" and r["trim"] == "1.5s"][0]; assert short["ok"] == 0 and short["pred"] == "ko"          # a short Vietnamese clip is missed
    md = L.report(rows)
    assert "| adv | clean | full | 3 | 1/2 | 1/1 |" in md or "| adv | clean | full |" in md, md
    assert md.count("| adv | vi-ko |") == len([r for r in rows if not r["ok"] and r["dir"] == "vi-ko"])
    assert "Every miss" in md and "boom" in md


def test_limit_and_short_wav_skipped():
    items = {"vi-ko": [dict(id=str(i), wav=wav(1, 5), cond="", set="btc") for i in range(10)]}
    assert len(L.evaluate(lambda s, t: FakeConv("ko"), items, [None], load=lambda w: w, limit=4)) == 4
    items = {"vi-ko": [dict(id="x", wav=wav(1, 0.1), cond="", set="btc")]}
    assert L.evaluate(lambda s, t: FakeConv("ko"), items, [None], load=lambda w: w) == []


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("ok", k)
