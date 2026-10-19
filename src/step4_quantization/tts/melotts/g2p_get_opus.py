# -*- coding: utf-8 -*-
"""Chinese side of OPUS-100 en-zh (train) -> work/opus_zh.json: mostly-hanzi sentences, 6..60 chars (input corpus for the g2p table distillation)."""
import sys, re, json, pandas as pd
from huggingface_hub import list_repo_files, hf_hub_download
sys.stdout.reconfigure(encoding="utf-8")
fs = [f for f in list_repo_files("Helsinki-NLP/opus-100", repo_type="dataset") if "en-zh" in f and "train" in f]
df = pd.read_parquet(hf_hub_download("Helsinki-NLP/opus-100", fs[0], repo_type="dataset"))
zh = [r["zh"] for r in df["translation"]]
ok = [s for s in zh if 6 <= len(s) <= 60 and len(re.findall(r"[一-鿿]", s)) >= 0.8 * len(s)]
json.dump(ok[:500000], open("work/opus_zh.json", "w", encoding="utf-8"), ensure_ascii=False); print("sentences", len(ok[:500000]))
