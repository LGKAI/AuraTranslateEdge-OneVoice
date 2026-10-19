# -*- coding: utf-8 -*-
"""Static NPU-style front vs the MeloTTS host front on held-out Chinese clauses: phone/tone/E/BERT-token exactness."""
import sys, re, json, unicodedata, numpy as np, torch
sys.path.insert(0, "MeloTTS"); sys.path.insert(0, "."); sys.stdout.reconfigure(encoding="utf-8")
from transformers import AutoTokenizer
from melo.api import TTS
from melo_front import MeloFront, load_tables, LB, N, L
from melo_full import host_front
from huggingface_hub import hf_hub_download
T = load_tables(); F = MeloFront(T).eval(); tts = TTS(language="ZH", device="cpu"); tok = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext-large")
sents = []
for sp in ("test",):
    p = hf_hub_download("google/fleurs", f"data/cmn_hans_cn/{sp}.tsv", repo_type="dataset")
    for l in open(p, encoding="utf-8"):
        f = l.rstrip("\n").split("\t")
        if len(f) > 3:
            for c in re.split(r"[。！？；]", f[2]):
                c = c.strip()
                if 4 <= len(c) <= 20 and re.fullmatch(r"[一-鿿，、]+", c): sents.append(c)
sents = list(dict.fromkeys(sents))[:300]; print("clauses", len(sents), flush=True)
ok_x = ok_t = ok_e = ok_b = 0; tot = 0; bad = []
for t in sents:
    try: bi, bm, E, x, tone, lang, xl = host_front(tts, tok, t, N, L)
    except Exception as e: continue
    b = np.zeros((1, LB), np.int32); by = unicodedata.normalize("NFC", t).encode("utf-8"); b[0, :len(by)] = list(by)
    with torch.no_grad(): gx, gt, gl, gxl, gE, gh, glw, gm = F(torch.from_numpy(b))
    tot += 1; n = int(xl[0]); sx = np.array_equal(gx.numpy(), x) and int(gxl[0]) == n; st = np.array_equal(gt.numpy(), tone); se = np.allclose(gE.numpy(), E)
    toks = (gh.numpy()[0] * 128 + glw.numpy()[0]).astype(int); ids = [tok.convert_tokens_to_ids(T["toks"][i]) if T["toks"][i] not in ("[PAD]",) else 0 for i in toks]
    sb = [i for i, m_ in zip(ids, gm.numpy()[0]) if m_ > 0.5] == [int(v) for v in bi[0][:int(bm[0].sum())]]
    ok_x += sx; ok_t += st; ok_e += se; ok_b += sb
    if not (sx and st): bad.append((t, int((gx.numpy() != x).sum()), int((gt.numpy() != tone).sum())))
print(f"phone ids exact {ok_x}/{tot} | tones exact {ok_t}/{tot} | E exact {ok_e}/{tot} | BERT token ids exact {ok_b}/{tot}")
print("examples of mismatch (text, #phone diffs, #tone diffs):", bad[:8])
