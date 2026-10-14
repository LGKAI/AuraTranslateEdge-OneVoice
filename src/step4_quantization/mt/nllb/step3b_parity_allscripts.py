import sys, json, torch, numpy as np
from transformers import AutoModelForSeq2SeqLM
sys.stdout.reconfigure(encoding="utf-8"); sys.path.insert(0, ".")
from nllb_static import NLLBStatic
import vocab_restrict as VR
tok, V, inv, keep = VR.build(all_scripts=True); print('kept', len(keep)); R = VR.restrict(tok, inv, keep)
hf = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M").eval(); S, T = 32, 24
m = NLLBStatic(hf, S, T, keep_ids=keep).eval()
test = json.load(open("work/fleurs_test.json", encoding="utf-8"))
PAIRS = [("vie_Latn","eng_Latn"),("eng_Latn","vie_Latn"),("kor_Hang","eng_Latn"),("zho_Hans","eng_Latn"),("eng_Latn","kor_Hang"),("eng_Latn","zho_Hans")]
mf = NLLBStatic(hf, S, T).eval()
def run(model, a, t):
    with torch.no_grad(): o = model(torch.from_numpy(a), torch.tensor([V[t]], dtype=torch.int32))[0].tolist()
    return o[:o.index(2) + 1] if 2 in o else o
res = {"pruned": [0, 0, 0], "unpruned": [0, 0, 0]}; per = {}
for s, t in PAIRS:
    sents = [x for x in test[s] if len(R.ids(x, s)) <= S][:8]; c = {"pruned": 0, "unpruned": 0}
    for x in sents:
        ids = R.ids(x, s); a = np.ones((1, S), np.int32); a[0, :len(ids)] = ids
        with torch.no_grad(): g = hf.generate(input_ids=torch.tensor([ids]), forced_bos_token_id=V[t], max_new_tokens=T - 1, num_beams=1, do_sample=False)[0].tolist()[1:]
        for name, mod in (("pruned", m), ("unpruned", mf)):
            cut = run(mod, a, t); n = min(len(cut), len(g)); same = cut[:n] == g[:n] and (n == len(g) or n == len(cut) == T)
            first = next((i for i in range(n) if cut[i] != g[i]), n); res[name][0] += same; res[name][1] += 1; res[name][2] += first / max(n, 1); c[name] += same
    per[f"{s[:3]}->{t[:3]}"] = f"pruned {c['pruned']}/{len(sents)} unpruned {c['unpruned']}/{len(sents)}"
for k, v in res.items(): print(f"{k:9s}: identical-to-HF {v[0]}/{v[1]} | mean fraction of tokens before first divergence {v[2]/v[1]:.3f}")
print(per)
