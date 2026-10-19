# -*- coding: utf-8 -*-
import sys, json, pickle, time, multiprocessing as mp
sys.path.insert(0, "MeloTTS"); sys.stdout.reconfigure(encoding="utf-8")
def work(s):
    from g2p_distill import per_char
    try: return per_char(s)
    except Exception: return None
if __name__ == "__main__":
    S = json.load(open("work/opus_zh.json", encoding="utf-8"))[:int(sys.argv[1])]; t0 = time.time(); out = []
    with mp.Pool(6) as p:
        for i, r in enumerate(p.imap(work, S, chunksize=200)):
            if r: out.append(r)
            if i % 10000 == 0: print(i, len(out), f"{time.time()-t0:.0f}s", flush=True)
    pickle.dump(out, open("work/g2p_big.pkl", "wb")); print("done", len(out), "sentences,", sum(len(r) for r in out), "chars", f"{time.time()-t0:.0f}s", flush=True)
