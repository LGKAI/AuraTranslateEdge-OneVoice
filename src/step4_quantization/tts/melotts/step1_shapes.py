import sys, json, torch; sys.path.insert(0, "MeloTTS"); sys.stdout.reconfigure(encoding="utf-8")
from melo.api import TTS
from melo import utils
t = TTS(language="ZH", device="cpu"); m = t.model; hps = t.hps
S = json.load(open("work/cpu_baseline.json", encoding="utf-8")); T = [r["text"] for r in S] + ["今天早上我去了超市，买了很多水果和蔬菜，然后回家做了一顿丰盛的午饭。", "你好。"]
for x in T:
    bert, ja, ph, tone, lang = utils.get_text_for_tts_infer(x, "ZH", hps, "cpu", t.symbol_to_id)
    with torch.no_grad():
        o = m.infer(ph[None], torch.LongTensor([len(ph)]), torch.LongTensor([1]), tone[None], lang[None], bert[None], ja[None], sdp_ratio=0.0, noise_scale=0.0, noise_scale_w=0.8, length_scale=1.0)
    print(f"{len(x):3d} chars -> N={len(ph):3d} phones(with blanks) | frames={o[2].sum().int().item():4d} ({o[2].sum().item()*512/44100:.2f}s) | bert {tuple(bert.shape)} ja {tuple(ja.shape)} lang ids {sorted(set(lang.tolist()))}")
