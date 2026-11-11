# -*- coding: utf-8 -*-
"""Parity of s2s_edge/punct_vi.py (numpy + onnxruntime, no torch) with the reference vibert-capu wrapper (torch + transformers), on the laptop.
  python tools/check_punct_vi.py --ref_dir C:/Users/Admin/Downloads/vibert_capu --onnx logs/vibert/vibert_capu.onnx --vocab assets/vi_capu/vocab.txt
1) tokenisation: ids and word offsets vs transformers' fast tokenizer as configured by the reference wrapper;
2) end-to-end text vs the reference on the sample-set sentences (punctuation removed) + ASR transcripts in logs/btc_cpu30*.json.
Inputs of < 3 words are excluded: there the reference returns the text unchanged and we deliberately use vi_punct instead."""
import argparse, dataclasses, glob, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import openpyxl
from s2s_edge.punct_vi import ViCapu

ap = argparse.ArgumentParser(); ap.add_argument("--ref_dir", required=True); ap.add_argument("--onnx", required=True); ap.add_argument("--vocab", required=True)
ap.add_argument("--data", default="C:/Users/Admin/Downloads/onevoice_btc_data"); a = ap.parse_args()
sys.path.append(a.ref_dir)
import modeling_seq2labels
modeling_seq2labels.Seq2LabelsOutput = dataclasses.dataclass(modeling_seq2labels.Seq2LabelsOutput)      # transformers >= 4.3x needs @dataclass
from gec_model import GecBERTModel
ref = GecBERTModel(vocab_path=os.path.join(a.ref_dir, "vocabulary"), model_paths=a.ref_dir, split_chunk=True, device="cpu")
port = ViCapu(a.onnx, a.vocab)

strip = lambda s: re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", str(s).lower())).strip()
def rows(p): r = list(openpyxl.load_workbook(p, read_only=True).active.values); return [x for x in r[1:] if x and x[0]]
texts = set()
for p in glob.glob(os.path.join(a.data, "*", "*.xlsx")) + glob.glob(os.path.join(a.data, "*", "*", "*.xlsx")):
    for r in rows(p):
        for c in (1, 2):
            s = str(r[c])
            if re.search(r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]", s.lower()): texts.add(strip(s))
for p in glob.glob("logs/btc_cpu30*.json"):
    for d, v in json.load(open(p, encoding="utf-8")).items():
        if d.startswith("vi-"): texts.update(x["asr"] for x in v["rows"] if x["asr"])
texts = sorted(t for t in texts if len(t.split()) >= 3)

# 1) tokenisation
tk = ref.indexers[0]; bad_tok = 0
for t in texts:
    w = ["$START"] + t.split(); enc = tk(w, is_split_into_words=True, add_special_tokens=False)
    wid = enc.word_ids(); offs = [0] + [i for i in range(1, len(wid)) if wid[i] != wid[i - 1]]
    ids, o2 = port.wp.encode_words(t.split())
    if enc["input_ids"] != ids or offs != o2: bad_tok += 1; print("TOKEN MISMATCH:", t) if bad_tok <= 5 else None
print(f"tokenisation: {len(texts) - bad_tok}/{len(texts)} identical (ids + word offsets); $START id {tk.convert_tokens_to_ids('$START')} vs {port.wp.start_id}")

# 2) end-to-end text
same = 0; diffs = []
for t in texts:
    r = ref(t); r = r[0] if isinstance(r, list) else r; p = port(t)
    if r == p: same += 1
    else: diffs.append((t, r, p))
print(f"text: {same}/{len(texts)} identical to the reference ({100 * same / len(texts):.1f} %)")
for t, r, p in diffs[:8]: print(f"  in : {t}\n  ref: {r}\n  new: {p}")
sys.exit(0 if bad_tok == 0 and same == len(texts) else 1)
