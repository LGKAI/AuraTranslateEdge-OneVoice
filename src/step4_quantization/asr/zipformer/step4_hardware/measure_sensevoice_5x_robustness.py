# -*- coding: utf-8 -*-
"""Test bo dataset gap 5x (25 mau/ngon ngu: 5 cau goc x 5 muc SNR 0/5/10/15/20 tu data/asr_mixed)
de xem pipeline co on dinh voi nhieu am thanh that khong, va in full transcript 1-2 mau/ngon ngu
de kiem tra bang mat (thay the cho "nghe" vi khong co cong cu phat am thanh).
So sanh 2 phien ban:
  - CURRENT_BUG : drop DC bin, khong sua preemph frame0 (nhung VAN co fix length-masking, vi
                  neu khong co fix nay thi moi thu deu la rac, khong do duoc gi ca)
  - FULLY_FIXED : drop Nyquist bin (dung) + preemph frame0 dung quy uoc Kaldi + length-masking
Chay trong tools/venv_sensevoice_diag."""
import sys, os, re, json, time
sys.stdout.reconfigure(encoding="utf-8"); sys.stderr.reconfigure(encoding="utf-8")
import numpy as np
import torch
torch.set_num_threads(4)
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
TEXTNORM_WOITN = 15

window = torch.hamming_window(FRAME_SAMPLES, periodic=False)
dft = torch.fft.rfft(torch.eye(N_FFT))
dft_real, dft_imag = dft.real, dft.imag
mel_fb, _ = K.get_mel_banks(N_MELS, N_FFT, FS, 20.0, 0.0, 100.0, -500.0, 1.0)


class FE(nn.Module):
    def __init__(self, cmvn, mode):
        super().__init__()
        self.mode = mode  # "current_bug" | "fully_fixed"
        self.register_buffer("window", window)
        self.register_buffer("dft_real", dft_real)
        self.register_buffer("dft_imag", dft_imag)
        self.register_buffer("mel_fb", mel_fb)
        self.register_buffer("cmvn_mean", cmvn[0:1, :])
        self.register_buffer("cmvn_scale", cmvn[1:2, :])

    def forward(self, wav):
        wav = wav * float(1 << 15)
        wav_img = wav.unsqueeze(1).unsqueeze(-1)
        frames = F.unfold(wav_img, kernel_size=(FRAME_SAMPLES, 1), stride=(HOP_SAMPLES, 1)).squeeze(0).transpose(0, 1)
        fp = frames.clone()
        fp[:, 1:] = frames[:, 1:] - 0.97 * frames[:, :-1]
        if self.mode == "fully_fixed":
            fp[:, 0] = frames[:, 0] * (1 - 0.97)
        windowed = fp * self.window.unsqueeze(0)
        padded = F.pad(windowed, (0, N_FFT - FRAME_SAMPLES))
        real = torch.matmul(padded, self.dft_real)
        imag = torch.matmul(padded, self.dft_imag)
        if self.mode == "fully_fixed":
            power = real[:, :-1] ** 2 + imag[:, :-1] ** 2  # drop Nyquist
        else:
            power = real[:, 1:] ** 2 + imag[:, 1:] ** 2     # drop DC (bug)
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


def build_byte_table(sp, vocab_size=25055, l_max=24):
    table = np.zeros((vocab_size, l_max), dtype=np.int64)
    special_re = re.compile(r"^<\|.*\|>$")
    for tid in range(vocab_size):
        piece = sp.id_to_piece(tid)
        if piece in ("<unk>", "<s>", "</s>") or special_re.match(piece):
            continue
        text = piece.replace("\u2581", " ")
        b = text.encode("utf-8")[:l_max]
        table[tid, : len(b)] = list(b)
    return torch.from_numpy(table)


def decode(logits, byte_table, out_len, blank_id=0):
    logits = logits[:, :out_len, :]
    token_ids = logits.argmax(dim=-1)[0]
    prev = torch.cat([torch.tensor([-1]), token_ids[:-1]])
    dedup = token_ids != prev
    valid = dedup & (token_ids != blank_id)
    c = torch.cumsum(valid.long(), 0)
    T = token_ids.shape[0]
    dummy = T
    pos = torch.where(valid, c - 1, torch.full_like(c, dummy))
    buf = torch.zeros(T + 1, dtype=torch.long)
    buf.scatter_(0, pos, token_ids)
    packed = buf[:T]
    bytes_out = F.embedding(packed, byte_table)
    n_valid = int(valid.sum().item())
    flat = bytes_out[:n_valid].reshape(-1).tolist()
    raw = bytes(b & 0xFF for b in flat)
    return raw.replace(b"\x00", b"").decode("utf-8", errors="replace")


def norm_wer(s):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", s.lower().strip())).strip()


def cer(ref, hyp):
    ref_c = re.sub(r"\s+", "", ref)
    hyp_c = re.sub(r"\s+", "", hyp)
    return jiwer.cer(ref_c, hyp_c) if ref_c else 1.0


def main():
    from funasr import AutoModel
    from funasr.models.sense_voice.export_meta import export_rebuild_model

    print("[1] Tai SenseVoice-Small ...", flush=True)
    am = AutoModel(model="iic/SenseVoiceSmall", device="cpu", disable_update=True)
    orig_fe = am.kwargs.get("frontend")
    orig_fe.dither = 0.0
    sp = am.kwargs.get("tokenizer").sp
    sv_model = am.model
    sv_model.eval()
    export_model = export_rebuild_model(sv_model, device="cpu", max_seq_len=512)
    export_model.eval()
    byte_table = build_byte_table(sp)

    fe_bug = FE(orig_fe.cmvn, "current_bug"); fe_bug.eval()
    fe_fix = FE(orig_fe.cmvn, "fully_fixed"); fe_fix.eval()

    manifest = json.load(open("data/asr/manifest.json", encoding="utf-8"))
    ref_by_lang_idx = {}
    for r in manifest:
        if r["lang"] in LID_DICT:
            idx = int(os.path.basename(r["path"]).split("_")[1].split(".")[0])
            ref_by_lang_idx[(r["lang"], idx)] = r["transcript"]

    snrs = [0, 5, 10, 15, 20]
    results = {lang: {"current_bug": {s: [] for s in snrs}, "fully_fixed": {s: [] for s in snrs}} for lang in LID_DICT}
    printed = {lang: 0 for lang in LID_DICT}

    t_start = time.time()
    n_done = 0
    for lang in ("en", "zh", "ko"):
        lid = torch.tensor([LID_DICT[lang]], dtype=torch.long)
        tn = torch.tensor([TEXTNORM_WOITN], dtype=torch.long)
        for idx in range(5):
            ref = ref_by_lang_idx[(lang, idx)]
            for snr in snrs:
                path = f"data/asr_mixed/{lang}/{lang}_{idx}_snr{snr}.wav"
                if not os.path.exists(path):
                    continue
                wav, sr = sf.read(path, dtype="float32")
                wav_t = torch.from_numpy(wav).unsqueeze(0)
                for tag, fe in (("current_bug", fe_bug), ("fully_fixed", fe_fix)):
                    t0 = time.time()
                    with torch.no_grad():
                        feat, actual_len = fe(wav_t)
                        sl = torch.tensor([actual_len], dtype=torch.int32)
                        logits, out_lens = export_model(feat, sl, lid, tn)
                    hyp = decode(logits, byte_table, int(out_lens[0].item()))
                    metric = cer(ref, hyp) if lang == "zh" else jiwer.wer(norm_wer(ref), norm_wer(hyp))
                    results[lang][tag][snr].append(metric)
                    n_done += 1
                    print(f"[{n_done}/150] {lang} idx={idx} snr={snr} {tag} metric={metric*100:.1f}% dt={time.time()-t0:.2f}s total_elapsed={time.time()-t_start:.1f}s", flush=True)
                    if printed[lang] < 2 and snr in (20, 0) and tag == "fully_fixed":
                        print(f"--- FULL TRANSCRIPT {lang} idx={idx} snr={snr} ---", flush=True)
                        print(f"REF: {ref}", flush=True)
                        print(f"HYP: {hyp}", flush=True)
                        if snr == 0:
                            printed[lang] += 1

    print("\n\n=== KET QUA ROBUSTNESS TREN 5x DATASET (25 mau/ngon ngu, 5 muc SNR) ===")
    for lang in ("en", "zh", "ko"):
        metric_name = "CER" if lang == "zh" else "WER"
        for tag in ("current_bug", "fully_fixed"):
            for snr in snrs:
                vals = results[lang][tag][snr]
                if vals:
                    print(f"  {lang} {tag:12s} snr={snr:2d} n={len(vals)} {metric_name}={np.mean(vals)*100:.2f}%")
    print("\n=== TRUNG BINH TOAN BO (tat ca SNR gop lai) ===")
    for lang in ("en", "zh", "ko"):
        metric_name = "CER" if lang == "zh" else "WER"
        for tag in ("current_bug", "fully_fixed"):
            allvals = [v for s in snrs for v in results[lang][tag][s]]
            print(f"  {lang} {tag:12s} n={len(allvals)} {metric_name}={np.mean(allvals)*100:.2f}%")

    with open("outputs/sensevoice_5x_robustness.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\nDa luu: outputs/sensevoice_5x_robustness.json")


if __name__ == "__main__":
    main()
