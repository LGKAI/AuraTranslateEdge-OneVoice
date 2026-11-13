# LAPTOP: run one data set on the box in two halves through the box web server (tools/box_serve.sh must be running, adb forward tcp:8770 tcp:8770),
# download each half's zip to Downloads before starting the next. Resumable: re-running with the same STAMP skips clips already done.
#   python tools/laptop_run_halves.py btc 20261113_1500        then   python tools/laptop_run_halves.py adv 20261113_1500
# ADV halves are split by sentence, so the 4 noise versions of a sentence stay together. Pair with tools/laptop_stream.py <STAMP> for live copies.
import json, os, re, time, urllib.request as u, urllib.parse as up
B = "http://127.0.0.1:8770"; OUT = r"C:\Users\Admin\Downloads"; DIRS = ["vi-en", "en-vi", "vi-ko", "ko-vi", "vi-zh", "zh-vi"]
import sys; SET = sys.argv[1] if len(sys.argv) > 1 else "btc"; STAMP = sys.argv[2] if len(sys.argv) > 2 else time.strftime("%Y%m%d_%H%M")
g = lambda p, t=120: json.loads(u.urlopen(B + p, timeout=t).read())
post = lambda p, d: json.loads(u.urlopen(u.Request(B + p, data=json.dumps(d).encode(), headers={"Content-Type": "application/json"}), timeout=900).read())
base = lambda cid: re.sub(r"_[a-z]+(_\d+db)?$", "", cid) if SET == "adv" else cid
items = {d: g(f"/api/items?set={SET}&dir={d}") for d in DIRS}
for half in (1, 2):
    run = f"{SET}_half{half}_{STAMP}"; plan = []
    for d in DIRS:
        sids = sorted({base(x["id"]) for x in items[d]}); keep = set(sids[: len(sids) // 2] if half == 1 else sids[len(sids) // 2:])
        plan += [(d, x["id"]) for x in items[d] if base(x["id"]) in keep]
    done = {(r["dir"], r["id"]) for r in g(f"/api/rows?run={run}&since=0")["rows"]}; t0 = time.time(); n = 0
    print(f"=== {SET} half {half}: {len(plan)} clips, run {run}", flush=True)
    for d, cid in plan:
        if (d, cid) in done: continue
        for attempt in range(3):
            try: post("/api/run_item", {"set": SET, "dir": d, "id": cid, "run": run}); break
            except Exception as e: print("  retry", d, cid, repr(e)[:100], flush=True); time.sleep(5)
        n += 1
        if n % 10 == 0 or n == len(plan): el = time.time() - t0; print(f"  {SET} half {half}: {n}/{len(plan)}, {el/60:.1f} min, ~{el/n*(len(plan)-n)/60:.0f} min left", flush=True)
    dst = os.path.join(OUT, f"onevoice_{run}.zip")
    with u.urlopen(B + "/api/export?run=" + up.quote(run), timeout=1800) as resp, open(dst + ".part", "wb") as f:
        while True:
            blk = resp.read(1 << 20)
            if not blk: break
            f.write(blk)
    os.replace(dst + ".part", dst); print(f"=== saved {dst} ({os.path.getsize(dst)/1e6:.0f} MB)", flush=True)
    s = g("/api/summary?run=" + up.quote(run))
    for x in s["summary"]: print(f"  {x['dir']} {x['cond']}: n {x['n']} | ASR {x['asr_metric']} {x['asr_err']}% | TTS err {x['tts_err']}% | total {x['total_s']}s", flush=True)
print(f"{SET} ALL DONE", flush=True)
