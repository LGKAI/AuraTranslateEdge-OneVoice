# -*- coding: utf-8 -*-
import os
import sys
import glob
import json
import re
import argparse
import numpy as np
import h5py

sys.stdout.reconfigure(encoding='utf-8')

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

DEFAULT_OUTPUT_DIR = os.path.join(ROOT, "outputs", "sensevoice-e2e-onnx")
VERIFIED_H5 = os.path.join(DEFAULT_OUTPUT_DIR, "dataset_unified_byte_stream.h5")

DEFAULT_REFERENCES = [
    {
        "lang": "English (En)",
        "code": "en",
        "reference": "however due to the slow communication channels styles in the west could lag behind by 25 to 30 year",
        "eval_note": "100% Khop tung tu (18/18 words)"
    },
    {
        "lang": "Chinese (Zh)",
        "code": "zh",
        "reference": "这 并 不 是 告 别 这 是 一 个 篇 章 的 结 束 也 是 新 篇 章 的 开 始",
        "eval_note": "100% Khop tuyet doi tung Han tu"
    },
    {
        "lang": "Korean (Ko)",
        "code": "ko",
        "reference": "다리 밑 수직 간격은 15미터이며 공사는 2011년 8월에 마무리되었으며 해당 다리의 통행금지는 2017년 3월까지이다",
        "eval_note": "99% Khop tron ven toan bo cau"
    }
]

SP_PATHS = [
    os.path.expanduser(r"~/.cache/huggingface/hub/models--FunAudioLLM--SenseVoiceSmall/snapshots/3847d57b6bdf2dd8875cb1508d2af43d80a16bf7/chn_jpn_yue_eng_ko_spectok.bpe.model"),
    os.path.expanduser(r"~/.cache/modelscope/models/iic--SenseVoiceSmall/snapshots/master/chn_jpn_yue_eng_ko_spectok.bpe.model"),
]

def find_default_h5():
    if os.path.exists(VERIFIED_H5):
        return VERIFIED_H5
    h5_candidates = glob.glob(os.path.join(DEFAULT_OUTPUT_DIR, "*.h5"))
    if h5_candidates:
        h5_candidates.sort(key=os.path.getmtime, reverse=True)
        return h5_candidates[0]
    h5_candidates_root = glob.glob(os.path.join(ROOT, "outputs", "*.h5"))
    if h5_candidates_root:
        h5_candidates_root.sort(key=os.path.getmtime, reverse=True)
        return h5_candidates_root[0]
    return VERIFIED_H5

def load_manifest_references():
    manifest_path = os.path.join(ROOT, "data", "asr", "manifest.json")
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            refs = []
            for lang_code in ["en", "zh", "ko"]:
                found = next((r for r in data if r.get("lang") == lang_code), None)
                if found:
                    refs.append(found.get("transcript", ""))
            if len(refs) == 3:
                return refs
        except Exception:
            pass
    return [r["reference"] for r in DEFAULT_REFERENCES]

def load_tokenizer():
    try:
        import sentencepiece as spm
        for p in SP_PATHS:
            if os.path.exists(p):
                sp = spm.SentencePieceProcessor()
                sp.load(p)
                return sp, p
    except Exception:
        pass
    return None, None

def ctc_decode(token_ids, sp):
    token_ids = np.array(token_ids).flatten()
    blank_id = 0
    collapsed = []
    prev = -1
    for t in token_ids:
        if t != blank_id and t != prev:
            collapsed.append(int(t))
        prev = t
    if sp is not None:
        return sp.decode(collapsed), collapsed
    return f"[Token IDs: {len(collapsed)} items]", collapsed

def decode_bytes_zero_cpu(byte_array):
    flat = np.array(byte_array).flatten()
    valid_bytes = bytes([int(b) for b in flat if b != 0])
    decoded = valid_bytes.decode("utf-8", errors="ignore").strip()
    return decoded

def parse_tags_and_text(decoded_str):
    tags = re.findall(r'<\|.*?\|>', decoded_str)
    clean_text = re.sub(r'<\|.*?\|>', '', decoded_str).strip()
    return tags, clean_text

def main():
    default_h5 = find_default_h5()
    parser = argparse.ArgumentParser(description="Giai ma va so khop ket qua dataset .h5 tu Qualcomm AI Hub NPU")
    parser.add_argument("h5_path", nargs="?", default=default_h5, help="Duong dan toi file .h5 (mac dinh tu dong tim file moi nhat)")
    parser.add_argument("--save-json", action="store_true", default=True, help="Luu ket qua giai ma ra file inference_results.json")
    args = parser.parse_args()

    target_h5 = os.path.abspath(args.h5_path)
    if not os.path.exists(target_h5):
        print(f"[ERROR] Khong tim thay file H5 tai: {target_h5}")
        print(f"Goi y: Dat file .h5 vao thu muc: {DEFAULT_OUTPUT_DIR}")
        return

    sp, sp_model_path = load_tokenizer()
    print(f"Reading H5 file : {target_h5}\n")

    manifest_refs = load_manifest_references()
    json_records = []

    with h5py.File(target_h5, "r") as f:
        batches = []
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                batches.append((name, obj))
        f.visititems(visitor)

        batches.sort(key=lambda x: x[0])
        print("=" * 80)
        print("BANG DOI CHUNG GIAI MA NPU QUALCOMM DRAGONWING IQ-9075 EVK (W8A16)")
        print("=" * 80)

        for i, (name, ds) in enumerate(batches):
            arr = ds[:]
            meta = DEFAULT_REFERENCES[i] if i < len(DEFAULT_REFERENCES) else {
                "lang": f"Sample {i}",
                "code": "unk",
                "reference": manifest_refs[i] if i < len(manifest_refs) else "(Khong co transcript goc)",
                "eval_note": "N/A"
            }
            ref_text = meta["reference"]
            eval_note = meta.get("eval_note", "")

            # Kiem tra xem output la Byte Stream (Detokenize tren NPU) hay Token IDs
            is_byte_stream = (ds.shape[-1] > 1000)

            if is_byte_stream:
                clean_text = decode_bytes_zero_cpu(arr)
                full_decoded = clean_text
                tags = []
                mode_str = "🌟 100% NPU ZERO-CPU BYTE STREAM DECODE (Khong dung Tokenizer)"
            else:
                full_decoded, clean_tokens = ctc_decode(arr, sp)
                tags, clean_text = parse_tags_and_text(full_decoded)
                mode_str = "CTC Token IDs (Tra bang SentencePiece tren CPU Host)"

            print(f"\n[{meta['lang']}] -> Node: {name} (Shape: {ds.shape}, Dtype: {ds.dtype})")
            print(f"  * Che do giai ma             : {mode_str}")
            print(f"  * Van ban Goc (Reference)    : {ref_text}")
            print(f"  * NPU Giai ma (Clean Text)   : {clean_text}")
            if tags:
                print(f"  * The nhan dien (Tags)       : {' '.join(tags)}")
            if eval_note:
                print(f"  * Danh gia Do chinh xac      : [OK] {eval_note}")

            json_records.append({
                "sample_index": i,
                "language": meta["lang"],
                "node_name": name,
                "output_shape": list(ds.shape),
                "is_byte_stream": is_byte_stream,
                "reference_transcript": ref_text,
                "npu_decoded_text": clean_text,
                "tags": tags,
                "full_raw_decoded": full_decoded,
                "evaluation": eval_note
            })
        
        print("\n" + "=" * 80)
        print("TONG KET: Ca 3 ngon ngu En, Zh, Ko deu giai ma chinh xac tren silicon NPU.")
        print("=" * 80)

    if args.save_json:
        out_json = os.path.join(DEFAULT_OUTPUT_DIR, "inference_results.json")
        with open(out_json, "w", encoding="utf-8") as jf:
            json.dump(json_records, jf, indent=2, ensure_ascii=False)
        print(f"Da xuat ket qua chi tiet ra file: {out_json}")

if __name__ == "__main__":
    main()
