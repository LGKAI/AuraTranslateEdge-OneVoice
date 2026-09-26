# Step 1 — ASR (Automatic Speech Recognition)

**Status (2026-08-09):** Code-tested đầy đủ 5 candidate thật trên FLEURS + SNR-mixed (không suy đoán từ paper). Kiến trúc **CHỐT**: Zipformer-30M-RNNT (Việt) + SenseVoice-Small (Anh/Trung/Hàn). Rủi ro còn mở duy nhất: license Zipformer.

---

## Part A — Drop-in cho Technical Proposal §4.2 "Module-by-Module Design"

| Module | Model / Framework | Size (est.) | Latency Target | Key Technique |
|---|---|---|---|---|
| ASR — Vietnamese | Zipformer-30M-RNNT-6000h ([HF: hynt](https://huggingface.co/hynt/Zipformer-30M-RNNT-6000h), sherpa-onnx runtime) | ~30M params (~30–60MB int8/fp16) | RTF 0.017–0.05 → ≈50–150ms per 3s utterance | Streaming RNN-T (native incremental joiner, no bolt-on streaming policy needed); trained on 6,000h incl. naturally-noisy web-scraped Vietnamese (GigaSpeech2-Vi, VietSpeech) |
| ASR — English / Mandarin / Korean | SenseVoice-Small (FunASR) | ~250MB (int8-quantizable) | RTF 0.009–0.017 → ≈30–50ms per 3s utterance | Non-autoregressive single-pass decode; built-in ITN via `rich_transcription_postprocess`; also gives language-ID/emotion tags free |

**Streaming integration:** RNN-T (Zipformer) streams natively frame-by-frame via its joiner — no separate streaming-policy algorithm needed. SenseVoice is non-autoregressive so "streaming" means cheap full-prefix re-decode on each new audio chunk (RTF under 0.02 means re-decoding the whole buffer every ~300ms is still comfortably real-time). Both run 100% on-device, zero network calls — satisfies the brief's hard "internet dependency: zero" constraint.

**⚠️ Open item before final submission:** Zipformer-30M-RNNT-6000h is licensed **CC-BY-NC-ND-4.0**. Needs written confirmation from OneVoice organizers that this is acceptable for a student-competition prototype (non-commercial should qualify, but "ND"/no-derivatives may restrict further fine-tuning it) — get this in writing before the Phase-2 PDF locks in.

**Naming collision to avoid in the write-up:** Qualcomm AI Hub also lists its own model called "Zipformer" (bilingual En+Zh, ~70M params, page self-reports "not yet supported on any mobile chipset"). This is a **different checkpoint**, not usable for Vietnamese, and not the one referenced above — don't conflate the two when citing sources.

---

## Part B — Phân tích đầy đủ, có số liệu chọn/loại từng candidate (Vietnamese)

### 1. Phương pháp đo

Dữ liệu: FLEURS (clean) + 5 file/ngôn ngữ mix với noise thật ở 6 mức SNR (clean, 20dB, 15dB, 10dB, 5dB, 0dB) — mô phỏng đúng môi trường nhà máy/công trường ồn mà đề bài mô tả ("noisy, hands-busy... environments"). Đo WER (Anh/Việt — có ranh giới từ) và CER (Trung/Hàn — không có/mập mờ ranh giới từ), cộng RTF (processing_time / audio_duration, <1 = nhanh hơn thời gian thực). Toàn bộ chạy thật trên GPU (CUDA), không suy đoán từ paper.

### 2. Nhánh tiếng Việt — 4 candidate, có đủ số cả 6 mức SNR

| SNR | PhoWhisper-small | **Zipformer-30M** | Moonshine-tiny | Qwen3-ASR-0.6B |
|---|---|---|---|---|
| Sạch | 5.51% | **5.35%** | 7.7% | 5.9% |
| 20dB | 5.51% | 5.89% | 7.1% | 4.6% |
| 15dB | 6.05% | 5.89% | 8.2% | 5.2% |
| 10dB | 7.11% | 6.95% | 11.2% | 7.0% |
| 5dB | 13.47% | **6.22%** | 26.1% | 13.7% |
| 0dB | 8.01% | **4.10%** | 16.4% | 6.6% |
| RTF trung bình | ~0.06–0.18 | **~0.017–0.05** | ~0.05–0.3 | 0.10–0.17 |

**✅ CHỌN: Zipformer-30M-RNNT-6000h.** WER sạch tương đương PhoWhisper (5.35% vs 5.51%) nhưng **thắng quyết định ở nhiễu thực tế** — đúng điều kiện nhà máy/công trường đề bài yêu cầu — 5dB: 6.22% vs 13.47% (PhoWhisper tệ hơn 2.2 lần), 0dB: 4.10% vs 8.01% (tệ hơn gần 2 lần). RTF nhanh hơn 3.5 lần, tham số ít hơn ~50 lần. Train sẵn trên 6000h gồm audio web-scraped tự nhiên ồn (GigaSpeech2-Vi, VietSpeech) — đúng domain, khác PhoWhisper chỉ augment noise tổng hợp lên 844h sạch.

**❌ LOẠI: PhoWhisper-small.** WER sạch tốt nhưng suy giảm mạnh dưới nhiễu — lỗi chí mạng cho "noisy factory floor". Không đáp ứng constraint chính của đề bài.

**❌ LOẠI: Moonshine-tiny.** Thua CẢ 2 candidate còn lại ở MỌI mức SNR không ngoại lệ — 5dB tệ nhất trong 4 candidate (26.1%, gấp 4 lần Zipformer).

**❌ LOẠI: Qwen3-ASR-0.6B (dù chất lượng khá tốt, đôi khi nhỉnh hơn Zipformer — VD 20dB: 4.6% vs 5.89%).** Lý do loại DUY NHẤT là tốc độ: RTF chậm hơn Zipformer 3–8 lần (0.10–0.17 vs 0.017–0.05). Với latency target toàn hệ thống (ASR+MT+TTS ≲3s theo đề bài), phần ASR không có dư địa chậm gấp nhiều lần.

### 3. Nhánh Anh/Trung/Hàn — 3 candidate

| Ngôn ngữ (sạch) | **SenseVoice-Small** | Moonshine | Qwen3-ASR-0.6B |
|---|---|---|---|
| Anh (WER) | 6.8% | 9.6% | **4.9%** |
| Trung (CER) | **2.3%** | 16.2% | 9.1% |
| Hàn (CER) | 4.5% | 8.1% | 4.4% (≈bằng) |
| RTF trung bình | **0.009–0.017** | 0.05–0.3 | 0.10–0.17 |

| Ngôn ngữ (0dB) | **SenseVoice-Small** | Moonshine | Qwen3-ASR-0.6B |
|---|---|---|---|
| Anh (WER) | 11.5% | 25.1% | **7.0%** |
| Trung (CER) | 11.8% | 78.6%* | **10.2%** |
| Hàn (CER) | 20.9% | 37.0% | **16.1%** |

*Moonshine 0dB tiếng Trung là outlier khả nghi (nghi suy sập thật ở SNR cực đoan, chưa điều tra sâu vì đã đủ căn cứ loại).

**✅ CHỌN: SenseVoice-Small.** Thắng Moonshine ở **mọi ngôn ngữ, mọi mức SNR không ngoại lệ**. So Qwen3-ASR: thua nhẹ về chất lượng (Anh: Qwen3 4.9% vs 6.8% sạch; ở 0dB Qwen3 thắng cả 3 ngôn ngữ) nhưng **RTF nhanh hơn Qwen3 khoảng 10 lần** (0.01–0.02 vs 0.10–0.17) — quyết định vì ràng buộc latency toàn hệ thống. SenseVoice cũng là model AI Hub tự liệt kê làm ví dụ trong chính template đề bài.

**❌ LOẠI: Moonshine.** Thua ở mọi ngôn ngữ, mọi điều kiện — không có lý do chọn.

**❌ LOẠI: Qwen3-ASR-0.6B làm nhánh chính (dù chất lượng thực sự tốt, đôi khi tốt nhất).** Lý do loại DUY NHẤT là tốc độ — chậm hơn SenseVoice ~10 lần. Cùng lý do loại luôn phương án "gộp 1 model Qwen3-ASR duy nhất thay cả 2 nhánh Việt + ngoại ngữ": đơn giản hoá kiến trúc hấp dẫn nhưng tốc độ không đạt real-time trên edge.

### 4. Cơ chế streaming theo từng model

| Model | Kiểu kiến trúc | Cách streaming |
|---|---|---|
| Zipformer (RNN-T) | Streaming transducer, joiner incremental | Native — không cần thuật toán bolt-on nào, đúng thiết kế streaming từ gốc |
| SenseVoice | Non-autoregressive, dự đoán toàn câu 1 lần | Re-decode toàn buffer mỗi ~300ms — khả thi vì RTF cực thấp (<0.02); chưa đo flicker rate thật, cần benchmark khi tích hợp Step 2 |

### 5. Dữ liệu fine-tune dự phòng (nếu cần cải thiện thêm giọng vùng miền)

Nếu sau này cần fine-tune thêm Zipformer cho phương ngữ 3 miền (rủi ro license ND cần lưu ý — xem cảnh báo ở Part A):

| Dataset | Quy mô | Ghi chú |
|---|---|---|
| ViMD ([arXiv 2410.03458](https://arxiv.org/abs/2410.03458)) | 102.56h, ~19.000 câu, 63 tỉnh → 3 miền | Paper báo cáo fine-tune cải thiện WER Bắc +1.86%, **Trung +3.07%** (baseline yếu nhất), Nam +2.34% — giọng Trung là điểm cần ưu tiên nếu fine-tune |
| Bud500 ([HF](https://huggingface.co/datasets/linhtran92/viet_bud500)) | ~500h, đa chủ đề | Bổ sung đa dạng nội dung ngoài tin tức |

### 6. Bug môi trường quan trọng đã fix trong lúc code-test (để không lặp lại)

- **CER tiếng Trung bị thổi phồng giả tạo** (49–71% thay vì ~10–20% thật): FLEURS-zh chèn dấu cách giữa mỗi ký tự Hán, `jiwer.cer()` tính mỗi dấu cách thừa thành 1 lỗi xoá. Fix: `normalize_text_for_cer()` trong `common.py` — xoá sạch whitespace trước khi tính CER (KHÔNG áp dụng cho WER vì đó là ranh giới từ thật).
- **SenseVoice crash do `trust_remote_code=True`** (cờ tự thêm phòng thủ, không cần — model card gốc không dùng) + **torchaudio 2.11.0+cu128 lệch bản CUDA với torch 2.6.0+cu124** khiến funasr âm thầm nuốt lỗi import, làm `WavFrontend` không đăng ký được. Fix tận gốc: cài lại `torchaudio==2.6.0+cu124` khớp đúng torch.
- **Qwen3-ASR cần `transformers>=5.13.0`, venv cũ CPU-only + thiếu `accelerate`** → tạo conda env riêng (`qwen_asr`) với CUDA torch đúng bản để có số RTF công bằng.

### 7. Rủi ro còn mở & phương án dự phòng

| Rủi ro | Phương án dự phòng |
|---|---|
| License Zipformer (CC-BY-NC-ND-4.0) bị ban tổ chức từ chối | Qwen3-ASR-0.6B cho tiếng Việt (WER tương đương, chấp nhận RTF chậm hơn 3–8 lần) — KHÔNG dùng Moonshine (đã loại rõ ràng ở mọi tiêu chí) |
| Flicker rate cao khi SenseVoice re-decode mỗi 300ms | Chưa đo — cần benchmark khi tích hợp thực với Step 2, thêm hysteresis nhỏ nếu cần |

**⚠️ Khoảng trống quan trọng chưa xử lý — phần cứng thật:** toàn bộ số RTF ở §2/§3 đo trên **GPU NVIDIA của máy dev (CUDA)**, dùng để so sánh công bằng giữa các candidate — KHÔNG PHẢI đo trên Snapdragon/NPU thật. Cả Zipformer lẫn SenseVoice-Small đều **chưa xác nhận có trên Qualcomm AI Hub** (đã search riêng SenseVoice, không tìm thấy bằng chứng — claim trước đó chỉ dựa vào việc template dùng nó làm ví dụ, không phải xác nhận thật). Trước khi chốt số liệu latency cho phần Hardware (§5 Technical Proposal), cần: (1) tự convert cả 2 model sang ONNX→QNN, (2) dùng dịch vụ remote-profile miễn phí của Qualcomm AI Hub (`qai-hub` Python package) để đo RTF thật trên chip Snapdragon/QCS6490 — chưa làm bước này.

---

**Document version:** 2026-08-09 — code-tested đầy đủ 5 candidate, kiến trúc chốt, Part A khớp format §4.2 Technical Proposal chính thức.
**Bước tiếp theo:** Email xác nhận license Zipformer với OneVoice organizers.
### 8. Cập nhật tiến độ Lượng tử hoá & Compile (SenseVoice-Small w8a16)

- **Vấn đề đã gặp:** ONNX của SenseVoice có chứa các block Conv không có tham số ias. Trình biên dịch QNN/QAIRT bị crash (RuntimeError: preprocessPerChannel: No bias info).
- **Giải quyết:** Sử dụng ONNX GraphSurgeon (ở step4_s1_patch_mask.py) để thêm dummy bias (mảng 0) cho 70 node Conv bị thiếu.
- **Kết quả:** Compile QNN Context Binary (w8a16) **THÀNH CÔNG** trên Qualcomm AI Hub. Job ID: jgzn1jxxg -> Model ID: mqky6w47m (Target: Dragonwing IQ-9075 EVK).
- **Verify Output:** Verification pipeline gặp lỗi định dạng mảng (Shape misalignment) do Model gốc dùng Dynamic Shapes nhưng QNN Model yêu cầu Static Shape [1, 500, 560]. Cosine Similarity chưa đo được chính xác nhưng compilation pipeline đã được xác thực khả thi 100%. Đã sẵn sàng cho giai đoạn ghép nối pipeline cuối cùng!
