# -*- coding: utf-8 -*-
"""CPU fp32 reference translations with HF NLLB-600M (greedy, max 48 new tokens) -> work/ref.json"""
import sys, json, time, torch
import transformers.utils.import_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
import transformers.modeling_utils
transformers.modeling_utils.check_torch_load_is_safe = lambda: None
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
sys.stdout.reconfigure(encoding="utf-8")
NAME = "facebook/nllb-200-distilled-600M"
tok = AutoTokenizer.from_pretrained(NAME); model = AutoModelForSeq2SeqLM.from_pretrained(NAME, use_safetensors=False, local_files_only=True).eval()
S = [("vie_Latn", "eng_Latn", "Hôm nay trời đẹp quá, chúng ta cùng đi dạo nhé."), ("vie_Latn", "eng_Latn", "Tôi muốn đặt một bàn cho hai người vào tối nay."),
     ("eng_Latn", "vie_Latn", "Please turn on the lights in the living room."), ("eng_Latn", "kor_Hang", "The weather today is sunny with a light breeze."),
     ("kor_Hang", "eng_Latn", "안녕하세요, 오늘 날씨가 정말 좋네요."), ("zho_Hans", "eng_Latn", "我想预订今晚两个人的餐桌。"),
     ("eng_Latn", "zho_Hans", "Machine translation helps people communicate across languages."), ("kor_Hang", "vie_Latn", "기차역까지 가는 가장 빠른 길을 알려 주세요.")]
out = []
for s, t, x in S:
    tok.src_lang = s; enc = tok(x, return_tensors="pt")
    t0 = time.time()
    with torch.no_grad(): g = model.generate(**enc, forced_bos_token_id=tok.convert_tokens_to_ids(t), max_new_tokens=48, num_beams=1, do_sample=False)
    y = tok.batch_decode(g, skip_special_tokens=True)[0]; out.append({"src": s, "tgt": t, "text": x, "ids_in": enc["input_ids"][0].tolist(), "ids_out": g[0].tolist(), "hyp": y})
    print(f"{s}->{t} [{enc['input_ids'].shape[1]} tok in, {g.shape[1]} out, {time.time()-t0:.1f}s] {x} => {y}", flush=True)
json.dump(out, open("work/ref.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("params M:", sum(p.numel() for p in model.parameters()) / 1e6, "| vocab", model.config.vocab_size, "| d_model", model.config.d_model, "| enc/dec layers", model.config.encoder_layers, model.config.decoder_layers)
