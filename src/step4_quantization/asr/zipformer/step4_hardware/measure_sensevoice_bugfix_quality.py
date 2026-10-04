# -*- coding: utf-8 -*-
"""Do that chat luong: chay full pipeline (frontend + encoder + CTC + collapse + detokenize)
bang PyTorch fp32 (khong can ONNX/quantize) voi 2 phien ban frontend:
  - BUGGY : giong dung export.py hien tai cua Le Gia Khanh (drop bin DC)
  - FIXED : sua loi da tim ra (drop bin Nyquist, dung theo quy uoc Kaldi that)
Roi so WER voi (a) transcript that trong manifest, (b) ket qua official am.generate() cua FunASR.
Chay trong tools/venv_sensevoice_diag."""
import sys, os, re, json
sys.stdout.reconfigure(encoding="utf-8"); sys.stderr.reconfigure(encoding="utf-8")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
import jiwer
import torchaudio.compliance.kaldi as K

ROOT = r"C:\Users\Admin\Downloads\OneVoice"; os.chdir(ROOT)

FS = 16000
N_MELS = 80
FRAME_SAMPLES = 400
HOP_SAMPLES = 160
N_FFT = 512
LFR_M, LFR_N = 7, 6
MAX_LFR_FRAMES = 500
LID_DICT = {"zh": 3, "en": 4, "ko": 12}
TEXTNORM_WOITN = 15  # "woitn" -- khong ITN, khop voi transcript tho khong dau cau trong manifest

window = torch.hamming_window(FRAME_SAMPLES, periodic=False)
dft = torch.fft.rfft(torch.eye(N_FFT))
dft_real, dft_imag = dft.real, dft.imag
mel_fb, _ = K.get_mel_banks(N_MELS, N_FFT, FS, 20.0, 0.0, 100.0, -500.0, 1.0)


class FixableFrontend(nn.Module):
    def __init__(self, cmvn, drop="dc"):
        super().__init__()
        self.drop = drop
        self.register_buffer("window", window)
        self.register_buffer("dft_real", dft_real)
        self.register_buffer("dft_imag", dft_imag)
        self.register_buffer("mel_fb", mel_fb)
        self.register_buffer("cmvn_mean", cmvn[0:1, :])
        self.register_buffer("cmvn_scale", cmvn[1:2, :])

    def forward(self, wav):
        wav = wav * float(1 << 15)  # dung, khop FunASR that
        wav_img = wav.unsqueeze(1).unsqueeze(-1)
        frames = F.unfold(wav_img, kernel_size=(FRAME_SAMPLES, 1), stride=(HOP_SAMPLES, 1)).squeeze(0).transpose(0, 1)
        fp = frames.clone()
        fp[:, 1:] = frames[:, 1:] - 0.97 * frames[:, :-1]
        windowed = fp * self.window.unsqueeze(0)
        padded = F.pad(windowed, (0, N_FFT - FRAME_SAMPLES))
        real = torch.matmul(padded, self.dft_real)
        imag = torch.matmul(padded, self.dft_imag)
        if self.drop == "dc":
            power = real[:, 1:] ** 2 + imag[:, 1:] ** 2   # BUGGY (hien tai)
        else:
            power = real[:, :-1] ** 2 + imag[:, :-1] ** 2  # FIXED
        mel = torch.matmul(power, self.mel_fb.T)
        mel_log = torch.clamp(mel, min=1e-10).log()
        left_pad = mel_log[0:1].expand(LFR_M // 2, -1)
        mel_padded = torch.cat([left_pad, mel_log], dim=0)
        mel_img = mel_padded.T.unsqueeze(0).unsqueeze(-1)
        mel_uf = F.unfold(mel_img, kernel_size=(LFR_M, 1), stride=(LFR_N, 1)).squeeze(0).view(80, LFR_M, -1)
        mel_uf = mel_uf.permute(1, 0, 2).reshape(560, -1).transpose(0, 1)
        mel_uf = (mel_uf + self.cmvn_mean) * self.cmvn_scale
        actual_len = mel_uf.shape[0]
        if actual_len >= MAX_LFR_FRAMES:
            mel_uf = mel_uf[:MAX_LFR_FRAMES, :]
        else:
            mel_uf = F.pad(mel_uf, (0, 0, 0, MAX_LFR_FRAMES - actual_len))
        return mel_uf.unsqueeze(0), actual_len


def build_byte_table(tokenizer_sp, vocab_size=25055, l_max=24):
    table = np.zeros((vocab_size, l_max), dtype=np.int64)
    special_re = re.compile(r"^<\|.*\|>$")
    for tid in range(vocab_size):
        piece = tokenizer_sp.id_to_piece(tid)
        if piece in ("<unk>", "<s>", "</s>") or special_re.match(piece):
            continue
        text = piece.replace("\u2581", " ")
        b = text.encode("utf-8")[:l_max]
        table[tid, : len(b)] = list(b)
    return torch.from_numpy(table)


def ctc_collapse_and_decode(logits, byte_table, blank_id=0):
    # logits: [1, T, V] (numpy/torch), full replica cua StaticCTCCollapseAndDetokenizer trong export.py
    token_ids = logits.argmax(dim=-1)[0]  # [T]
    prev = torch.cat([torch.tensor([-1]), token_ids[:-1]])
    dedup_mask = token_ids != prev
    valid_mask = dedup_mask & (token_ids != blank_id)
    c = torch.cumsum(valid_mask.long(), dim=0)
    T = token_ids.shape[0]
    dummy_idx = T
    pos = torch.where(valid_mask, c - 1, torch.full_like(c, dummy_idx))
    buf = torch.zeros(T + 1, dtype=torch.long)
    buf.scatter_(0, pos, token_ids)
    packed = buf[:T]
    bytes_out = F.embedding(packed, byte_table)  # [T, L_MAX]
    n_valid = int(valid_mask.sum().item())
    flat = bytes_out[:n_valid].reshape(-1).tolist()
    return bytes(b & 0xFF for b in flat).decode("utf-8", errors="replace")


norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", s.lower().strip())).strip()


def main():
    from funasr import AutoModel
    print("[1] Tai SenseVoice-Small ...", flush=True)
    am = AutoModel(model="iic/SenseVoiceSmall", device="cpu", disable_update=True)
    orig_fe = am.kwargs.get("frontend")
    orig_fe.dither = 0.0
    tokenizer = am.kwargs.get("tokenizer")
    sv_model = am.model
    sv_model.eval()

    print("[2] Dung export_rebuild_model chinh thuc cua FunASR de lay acoustic model export-ready ...", flush=True)
    from funasr.models.sense_voice.export_meta import export_rebuild_model
    export_model = export_rebuild_model(sv_model, device="cpu", max_seq_len=512)
    export_model.eval()

    print("[3] Dung build_static_byte_table ...", flush=True)
    sp = tokenizer.sp if hasattr(tokenizer, "sp") else tokenizer.original_tokenizer.sp_model
    byte_table = build_byte_table(sp, vocab_size=export_model.ctc.ctc_lo.out_features if hasattr(export_model, "ctc") else 25055)

    manifest = json.load(open("data/asr/manifest.json", encoding="utf-8"))
    langs = {"en": [], "zh": [], "ko": []}
    for r in manifest:
        if r["lang"] in langs:
            langs[r["lang"]].append(r)

    fe_buggy = FixableFrontend(orig_fe.cmvn, drop="dc")
    fe_fixed = FixableFrontend(orig_fe.cmvn, drop="nyquist")
    fe_buggy.eval(); fe_fixed.eval()

    results = {"buggy": {}, "fixed": {}, "official": {}}
    for lang, items in langs.items():
        for tag in ("buggy", "fixed", "official"):
            results[tag].setdefault(lang, [])
        for item in items:
            wav, sr = sf.read(item["path"], dtype="float32")
            wav_t = torch.from_numpy(wav).unsqueeze(0)
            ref = item["transcript"]
            lid = torch.tensor([LID_DICT[lang]], dtype=torch.long)
            tn = torch.tensor([TEXTNORM_WOITN], dtype=torch.long)

            for tag, fe in (("buggy", fe_buggy), ("fixed", fe_fixed)):
                with torch.no_grad():
                    feat, actual_len = fe(wav_t)
                    speech_lengths = torch.tensor([actual_len], dtype=torch.int32)
                    logits, _ = export_model(feat, speech_lengths, lid, tn)
                hyp = ctc_collapse_and_decode(logits, byte_table)
                w = jiwer.wer(norm(ref), norm(hyp)) if norm(ref) else 1.0
                results[tag][lang].append(w)
                print(f"[{lang}][{tag}] WER={w*100:5.1f}%  REF: {ref[:60]}", flush=True)
                print(f"                    HYP: {hyp[:60]}", flush=True)

            # official FunASR full pipeline lam baseline "gold standard" (dung frontend that + generate that)
            try:
                res = am.generate(input=item["path"], language=lang, use_itn=False, cache={})
                hyp_off = res[0]["text"] if isinstance(res, list) else str(res)
                # loai tag ngon ngu kieu <|en|><|NEUTRAL|>... neu co
                hyp_off_clean = re.sub(r"<\|[^|]*\|>", "", hyp_off)
                w_off = jiwer.wer(norm(ref), norm(hyp_off_clean)) if norm(ref) else 1.0
                results["official"][lang].append(w_off)
                print(f"[{lang}][official] WER={w_off*100:5.1f}%  HYP: {hyp_off_clean[:60]}", flush=True)
            except Exception as e:
                print(f"[{lang}][official] LOI: {e}", flush=True)

    print("\n=== TONG KET WER TRUNG BINH ===")
    for tag in ("buggy", "fixed", "official"):
        for lang, ws in results[tag].items():
            if ws:
                print(f"  {tag:9s} {lang}: n={len(ws)} WER={np.mean(ws)*100:.2f}%")

    with open("outputs/sensevoice_bugfix_quality.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\nDa luu: outputs/sensevoice_bugfix_quality.json")


if __name__ == "__main__":
    main()
