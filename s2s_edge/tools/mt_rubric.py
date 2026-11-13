# -*- coding: utf-8 -*-
"""MT-only rubric (eval/mt_rubric.json: 68 NEW sentences of the same error types the team reported on 12/11, not the same sentences).
ON THE BOX, through the real pipeline MT (Registry.mt(): the per-direction router, 3.3B where 600M is wrong):
  runuser -u system -- bash -c 'cd /tmp/onevoice/s2s_edge_pkg && . .venv/bin/activate && . tools/env.sh 2>/dev/null && python tools/mt_rubric.py 2>&1 | grep -a "^##" | tee /tmp/mt_rubric.txt'
Prints which model produced each sentence and pass/fail; exits 2 with a banner when NLLB-3.3B is not in use (the target), 3 if any routed sentence came from 600M.
`score(case, text)` is also imported by the laptop-side comparison (600M vs 3.3B vs routed)."""
import json, os, re, sys, time, unicodedata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POLITE = re.compile(r"(요|니다|니까|까|죠|세요|예요|이에요)$")


def norm(s): return unicodedata.normalize("NFC", s or "").lower().replace("’", "'")


def score(case, text):
    """-> (passed, reason)"""
    t = norm(text); flat = t.replace(" ", "")
    has = lambda w: norm(w) in t or norm(w).replace(" ", "") in flat
    if case.get("ok") and not any(has(w) for w in case["ok"]): return False, "missing one of " + "/".join(case["ok"])
    for w in case.get("bad", []):
        if has(w): return False, f"contains '{w}'"
    for w in case.get("must_cover", []):
        if not has(w): return False, f"lost information '{w}'"
    if case.get("polite"):
        last = re.sub(r"[\s\.\?\!。？！,~]+$", "", text.strip()).split(" ")[-1] if text.strip() else ""
        if not POLITE.search(last): return False, f"not polite: ends with '{last}'"
    if case.get("question") and not (("?" in text) or any(w in t for w in ("không", "chưa", "hả", "à", "chứ", "nhỉ", "sao"))): return False, "not a question"
    if case.get("min_len") and len(re.findall(r"[가-힣一-鿿]", text)) < case["min_len"]: return False, "too short (truncated)"
    return True, ""


def load(): return json.load(open(os.path.join(ROOT, "eval", "mt_rubric.json"), encoding="utf-8"))["cases"]


if __name__ == "__main__":
    sys.path.insert(0, ROOT); sys.stdout.reconfigure(encoding="utf-8")
    from s2s_edge.registry import Registry
    cfg = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "models_qcs8550", "models_nllb33.json")
    reg = Registry(cfg, "npu"); t0 = time.time(); mt = reg.mt(); st = mt.status() if hasattr(mt, "status") else {}
    print(f"## MT built in {time.time() - t0:.0f} s. route: {st.get('route')} | 3.3B in use: {st.get('big_in_use')} | degraded: {st.get('degraded')} {st.get('errors', '')}", flush=True)
    if st.get("degraded") or not st.get("big_in_use"):
        print("## " + "!" * 100 + "\n## FAIL: NLLB-3.3B IS NOT IN USE. This run would test 600M, NOT the target. Fix first (see the error above).\n## " + "!" * 100); sys.exit(2)
    res = []; per = {}; wrong_model = 0
    for c in load():
        s, t = c["dir"].split("-"); a = time.perf_counter(); out = mt(c["src"], s, t); dt = time.perf_counter() - a
        if t == "ko":
            from s2s_edge.ko_polite import polite; out = polite(out)                      # same post-edit as the pipeline
        ok, why = score(c, out); used = mt.last; want = mt.wanted(s, t); wrong_model += used != want
        per.setdefault(c["dir"], []).append(ok); res.append(dict(c, out=out, passed=ok, why=why, model=used, wanted=want, t=round(dt, 2)))
        print(f"## {'PASS' if ok else 'FAIL'} {c['id']} {c['dir']} [{used}] {dt:.1f}s | {c['src'][:50]} -> {out[:60]}" + ("" if ok else f"   <-- {why}"), flush=True)
    for d, v in per.items(): print(f"## {d}: {sum(v)}/{len(v)} passed", flush=True)
    print(f"## TOTAL {sum(r['passed'] for r in res)}/{len(res)} passed; sentences produced by the wrong model: {wrong_model}", flush=True)
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True); json.dump(res, open(os.path.join(ROOT, "logs", "mt_rubric.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    sys.exit(3 if wrong_model else 0)
