# -*- coding: utf-8 -*-
"""ON THE BOX: measure the language identification of the conversation mode (s2s_edge.conversation.LanguageID) on REAL audio with known languages:
the BTC sample set (clean) and the ADV noise set (clean / factory 15 dB / construction 10 dB / heavy 5 dB), full clips and short clips (first 3 s, 1.5 s).

  bash start.sh check   (once)   then:
  cd <package> && . .venv/bin/activate && . tools/env.sh && python tools/lid_eval.py --data_root <folder with onevoice_btc_data and onevoice_btc_adv> [--auto] [--limit 30]

Truth = the source language of the direction (vi-en clips are Vietnamese, en-vi clips are English, ...). For each clip the first speech segment goes through
Conversation.route, exactly as in live use. Pair mode (default): the partner language is the clip's own language when it is not Vietnamese, so only the
vi / not-vi decision is measured. --auto also measures the en / zh / ko decision (SenseVoice `auto` + script of the text).
Writes logs/lid_eval/lid_eval.csv (one row per clip x trim) and lid_eval.md (confusion per set / noise / trim, the worst false 'vi' probability, every miss)."""
import argparse, csv, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DIRS = ["vi-en", "en-vi", "vi-zh", "zh-vi", "vi-ko", "ko-vi"]


def evaluate(conv_for, items_by_dir, trims, load, limit=0, log=print):
    """conv_for(pair_langs) -> object with .route(wav) -> (src, tgt, text, stats) and .lid.vi_prob(stats); items_by_dir {dir: [item with wav/id/cond/set]}.
    Returns rows [dict]. Separate from the CLI so it can be tested with fakes."""
    rows = []
    for d, items in items_by_dir.items():
        src_true, tgt_true = d.split("-")
        for k, m in enumerate(items[:limit] if limit else items):
            wav = load(m["wav"])
            for tr in trims:
                w = wav if tr is None else wav[: int(tr * 16000)]
                if len(w) < 4800: continue
                try:
                    conv = conv_for(src_true, tgt_true); src, tgt, text, st = conv.route(w); p = float(conv.lid.vi_prob(st)) if st and st.get("nwords", 0) >= 1 else 0.0
                    err = ""
                except Exception as e:                                       # a failing clip must not stop the survey
                    src, p, err = "?", 0.0, repr(e)[:120]
                rows.append(dict(set=m.get("set", ""), dir=d, id=m["id"], cond=m.get("cond", ""), trim="full" if tr is None else f"{tr:g}s", true=src_true, pred=src, ok=int(src == src_true),
                                 vi_prob=round(p, 3), error=err))
            if (k + 1) % 20 == 0: log(f"  {d}: {k + 1}/{len(items)}")
    return rows


def report(rows):
    out = ["# LID on real audio", "", f"{len(rows)} decisions. 'ok' = predicted language equals the clip's language.", ""]
    by = {}
    for r in rows: by.setdefault((r["set"], r["cond"] or "-", r["trim"]), []).append(r)
    out += ["## Accuracy per set / noise / trim", "", "| set | noise | trim | n | Vietnamese found (vi clips) | foreign NOT called vi | en/zh/ko right (auto mode only) | worst false-vi prob |", "|---|---|---|---|---|---|---|---|"]
    for (s, c, t), rs in sorted(by.items()):
        vi = [r for r in rs if r["true"] == "vi"]; fo = [r for r in rs if r["true"] != "vi"]
        fa = [r for r in fo if r["pred"] in ("en", "zh", "ko")]
        out.append(f"| {s} | {c} | {t} | {len(rs)} | {sum(r['ok'] for r in vi)}/{len(vi)} | {sum(r['pred'] != 'vi' for r in fo)}/{len(fo)} | "
                   f"{(str(sum(r['ok'] for r in fa)) + '/' + str(len(fa))) if fa else '-'} | {max([r['vi_prob'] for r in fo], default=0):.2f} |")
    miss = [r for r in rows if not r["ok"]]
    out += ["", f"## Every miss ({len(miss)})", "", "| set | dir | id | noise | trim | true | predicted | P(vi) | error |", "|---|---|---|---|---|---|---|---|---|"]
    out += [f"| {r['set']} | {r['dir']} | {r['id']} | {r['cond'] or '-'} | {r['trim']} | {r['true']} | {r['pred']} | {r['vi_prob']} | {r['error']} |" for r in miss]
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data_root", default=os.environ.get("S2S_DATA", "/root/onevoice")); ap.add_argument("--models", default=None)
    ap.add_argument("--backend", default=os.environ.get("S2S_BACKEND", "npu")); ap.add_argument("--auto", action="store_true"); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--trims", default="full,3,1.5"); ap.add_argument("--sets", default="btc,adv"); ap.add_argument("--out", default="logs/lid_eval"); a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    from s2s_edge.registry import Registry
    from s2s_edge.conversation import Conversation
    from s2s_edge.audio import load_audio
    from s2s_edge import datasets as ds
    models = a.models or (open("logs/models_ok.txt").read().strip() if os.path.exists("logs/models_ok.txt") else "models_qcs8550/models.json")
    reg = Registry(models, a.backend); sets = ds.find_sets(a.data_root); trims = [None if t == "full" else float(t) for t in a.trims.split(",")]
    print("sets found:", {k: v for k, v in sets.items()}, "| models:", models)
    items = {}
    for s in a.sets.split(","):
        if s not in sets: print(f"!! set '{s}' not found under {a.data_root}"); continue
        for d in DIRS:
            for m in ds.items(s, sets[s], d): items.setdefault((s, d), []).append(dict(m, set=s))
    convs = {}
    def conv_for(src, tgt):
        key = "auto" if a.auto else (src if src != "vi" else tgt)                # pair mode: the partner is the clip's own foreign language (or the target of a Vietnamese clip)
        if key not in convs: convs[key] = Conversation(reg, auto=True) if a.auto else Conversation(reg, pair=("vi", key))
        return convs[key]
    rows = []; t0 = time.time()
    for (s, d), its in items.items():
        print(f"== {s} {d}: {len(its)} clips", flush=True)
        rows += evaluate(conv_for, {d: its}, trims, load_audio, a.limit)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "lid_eval.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["set", "dir", "id", "cond", "trim", "true", "pred", "ok", "vi_prob", "error"]); w.writeheader(); w.writerows(rows)
    md = report(rows); open(os.path.join(a.out, "lid_eval.md"), "w", encoding="utf-8").write(md); print(md); print(f"done in {time.time() - t0:.0f} s -> {a.out}/lid_eval.csv, lid_eval.md")


if __name__ == "__main__":
    main()
