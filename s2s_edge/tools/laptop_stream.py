# LAPTOP: python tools/laptop_stream.py <STAMP> [target folder]   (same STAMP as laptop_run_halves.py). Copies every finished clip from the box to the laptop as soon as it is done:
#   Downloads\outputOnevoice\<direction>\<BTC|ADV>\<clip>_in.wav (source), <clip>_out.wav (translated speech), <clip>.json (texts, metrics, latency)
#   Downloads\outputOnevoice\<direction>\<BTC|ADV>_results.csv  and  Downloads\outputOnevoice\all_results.csv (one line per clip, appended live)
import csv, json, os, sys, time, urllib.request as u, urllib.parse as up
B = "http://127.0.0.1:8770"; STAMP = sys.argv[1]; ROOT = sys.argv[2] if len(sys.argv) > 2 else r"C:\Users\Admin\Downloads\outputOnevoice"   # 2nd arg: another folder, e.g. outputOnevoice2
COLS = ["set", "dir", "id", "cond", "audio_s", "text", "asr", "asr_err", "asr_s", "ref", "mt", "chrf", "mt_s", "heard", "tts_err", "heard_chrf", "tts_s", "out_audio_s", "total_s", "graph_s", "heard_s", "issues", "error", "run"]
for d in ["vi-en", "en-vi", "vi-ko", "ko-vi", "vi-zh", "zh-vi"]:
    for s in ("BTC", "ADV"): os.makedirs(os.path.join(ROOT, d, s), exist_ok=True)
seen = {}; got = set()
for p in ("all_results.csv",):
    if os.path.exists(os.path.join(ROOT, p)):
        for r in csv.DictReader(open(os.path.join(ROOT, p), encoding="utf-8-sig")): got.add((r["run"], r["dir"], r["id"]))
def fetch(url, dst):
    with u.urlopen(url, timeout=120) as r, open(dst + ".part", "wb") as f: f.write(r.read())
    os.replace(dst + ".part", dst)
def append(path, row):
    new = not os.path.exists(path)
    with open(path, "a", encoding="utf-8-sig" if new else "utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        if new: w.writeheader()
        w.writerow(dict(row, issues=" ".join(row.get("issues", []))))
print("streaming clips of runs *" + STAMP + " into", ROOT, flush=True); idle = 0
while True:
    try:
        runs = [r for r in json.loads(u.urlopen(B + "/api/info", timeout=60).read())["runs"] if r.endswith(STAMP)]
        new_any = False
        for run in runs:
            rows = json.loads(u.urlopen(B + f"/api/rows?run={up.quote(run)}&since={seen.get(run, 0)}", timeout=60).read()); seen[run] = seen.get(run, 0) + len(rows["rows"])
            for r in rows["rows"]:
                key = (run, r["dir"], r["id"])
                if key in got: continue
                sd = "BTC" if r["set"] == "btc" else "ADV"; folder = os.path.join(ROOT, r["dir"], sd); base = os.path.join(folder, r["id"])
                fetch(B + f"/api/audio?set={r['set']}&dir={r['dir']}&id={up.quote(r['id'])}", base + "_in.wav")
                if r.get("file"): fetch(B + f"/api/out?run={up.quote(run)}&file={up.quote(r['file'])}", base + "_out.wav")
                json.dump({k: v for k, v in r.items()}, open(base + ".json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
                append(os.path.join(ROOT, r["dir"], f"{sd}_results.csv"), r); append(os.path.join(ROOT, "all_results.csv"), r)
                got.add(key); new_any = True
                print(f"  {sd} {r['dir']} {r['id']}: total {r['total_s']:.2f}s | {r['asr'][:30]} -> {r['mt'][:30]}", flush=True)
        idle = 0 if new_any else idle + 1
    except Exception as e:
        print("  (retry)", repr(e)[:120], flush=True)
    time.sleep(2)
