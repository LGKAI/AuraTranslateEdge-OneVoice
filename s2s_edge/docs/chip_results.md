# Kết quả Đo kiểm trên Chip NPU: HSPTEK AIBOX 8550 (QCS8550) qua Qualcomm AI Hub

Các số liệu dưới đây được trích xuất trực tiếp từ các Job `profile` và `inference` trên nền tảng Qualcomm AI Hub: thời gian thực thi thuần trên chip silicon cho mỗi lượt gọi đồ thị, **không bao gồm** thời gian xếp hàng (queueing) và thời gian nạp mô hình vào RAM.
Đây là thời gian đo cấp đồ thị (graph-level execution); số liệu độ trễ toàn trình đầu-cuối trên thiết bị vật lý thật được đo đạc bởi script `python -m s2s_edge.bench` (xem README).

---

## 1. Kết quả Đo kiểm trên Chip QCS8550 (Thiết bị HSPTEK AIBOX 8550)

Toàn bộ 9 đồ thị mô hình đã được link trực tiếp cho SoC Qualcomm QCS8550 (QAIRT 2.51, thông tin link job lưu tại `models_qcs8550/link_jobs.json`) và đo kiểm profile thực tế qua `tools/profile_links.py`. Cả 9 tập tin QNN context binary đều nạp và chạy hoàn hảo 100% NPU offload trên QCS8550:

| Đồ thị (Graph) | Vai trò trong chuỗi | Định dạng NPU | Thời gian chip QCS8550 (AIBOX 8550) | Profile Job ID trên AI Hub | Kiểm định chất lượng |
|---|---|:---:|:---:|---|---|
| `asr_vi` | Zipformer-150M CTC (cửa sổ 15s) | FP16 | **163.4 ms** | `jgl8z82ep` | WER trên 15 clips: sạch 7.5%, 5 dB 9.5%, 0 dB 6.2% |
| `asr_xx_frontend` | SenseVoice Frontend (cửa sổ 29s) | FP16 | **58.3 ms** | `jp37371xg` | Khớp 100% đầu ra fbank CPU |
| `asr_xx_encoder` | SenseVoice Encoder (cửa sổ 29s) | W8A16 | **236.5 ms** | `jpvmomr7p` | CER: EN 4.8%, ZH 8.3%, KO 7.3% |
| `mt_enc` | NLLB-200-600M Encoder (64 token) | FP16 | **7.5 ms** | `jgjymy27g` | Cosine $\ge 0.999$ so với CPU trên 8 câu |
| `mt_dec` | NLLB Decoder Step (Greedy) | W8A16 | **15.2 ms / bước** | `jpex1xw7g` | 8/8 token trùng khớp tuyệt đối CPU |
| `mt_decB` | NLLB Decoder Step (4-Beam) | W8A16 | **30.8 ms / bước** | `jp11l1wkp` | 32/32 argmax trùng khớp, logit cosine $\ge 0.99999$ |
| `tts_vi` | Piper-VITS Tiếng Việt (≤ 26 từ) | W8A16 | **520.1 ms** | `j57nwnlqp` | VITS byte-level, âm sắc tự nhiên |
| `tts_zh` | MeloTTS-ZH Tiếng Trung (≤ 20 ký tự) | W8A16 | **740.6 ms** | `jp44o4dqp` | ASR CER 6.4% trên 40 văn bản |
| `tts_enko` | Supertonic-3 Anh / Hàn (≤ 82 ký tự) | W8A16 | **701.2 ms** | `jpxrjr6jp` | Round-trip WER 0.0% (EN), CER 1.5% (KO) |

---

## 2. Ước tính Thời gian Xử lý thuần trên Chip cho một Câu nói ngắn (~25 token MT)

Thời gian xử lý MT = Encoder 7.5 ms + (~25 bước giải mã):
- Chế độ 4-Beam search (mặc định): $7.5\text{ ms} + (25 \times 30.8\text{ ms}) \approx \mathbf{0.78\text{ s}}$
- Chế độ Greedy (`models_fast.json`): $7.5\text{ ms} + (25 \times 15.2\text{ ms}) \approx \mathbf{0.39\text{ s}}$

| Chiều dịch | ASR trên chip | MT trên chip (Beam 4 / Greedy) | TTS trên chip (1 chunk) | Tổng thời gian chip ước tính |
|:---:|:---:|:---:|:---:|:---:|
| **VI $\rightarrow$ EN / KO** | 163 ms | 0.78 s / 0.39 s | ~701 ms (Supertonic) | **~1.64 s (Beam) / ~1.25 s (Greedy)** |
| **VI $\rightarrow$ ZH** | 163 ms | 0.78 s / 0.39 s | ~740 ms (MeloTTS) | **~1.68 s (Beam) / ~1.29 s (Greedy)** |
| **EN / KO / ZH $\rightarrow$ VI** | 295 ms (SenseVoice) | 0.78 s / 0.39 s | ~520 ms (Piper) | **~1.60 s (Beam) / ~1.21 s (Greedy)** |

*Ghi chú:* Với các câu nói dài cần nhiều chunk TTS hoặc nhiều phân đoạn dịch, thời gian sẽ tăng tuyến tính theo số lượng block. Các tác vụ trên Host CPU (fbank, SentencePiece, chuẩn hoá số, ghép WAV) được đo kiểm trong pipeline hoàn chỉnh.

---

## 3. Chất lượng Dịch máy (MT FLORES-200 Devtest, điểm chrF / BLEU)

Đo kiểm bằng `tools/eval_mt.py` trên 25 câu FLORES-200 cho mỗi chiều dịch:

| Chiều dịch | S=32 Greedy | S=64 Greedy | S=64 Beam 4 (Chính thức) | HuggingFace FP32 Beam 4 (Tham chiếu) |
|:---:|:---:|:---:|:---:|:---:|
| **VI $\rightarrow$ EN** | 51.0 / 20.8 | 54.7 / 27.3 | **57.0 / 28.3** | 57.2 / 28.4 |
| **EN $\rightarrow$ VI** | 54.2 / 32.5 | 55.9 / 35.5 | **56.4 / 36.5** | 55.8 / 35.6 |
| **ZH $\rightarrow$ VI** | 40.4 / 15.3 | 46.5 / 23.7 | **47.7 / 25.4** | 46.8 / 24.1 |
| **KO $\rightarrow$ VI** | 45.3 / 23.1 | 46.1 / 24.9 | **46.8 / 25.1** | — |

Đồ thị W8A16 trên NPU tái tạo gần như chính xác 100% chất lượng của mô hình FP32 gốc Meta AI (VI $\rightarrow$ EN: 57.0 so với 57.2 chrF).

---

## 4. Đánh giá Toàn trình Speech ➔ Speech ➔ ASR nghe lại (`tools/eval_e2e.py`)

Đo trên 8 clips chuẩn FLEURS mỗi chiều, chế độ Beam 4:

| Chiều dịch | ASR-chrF | ASR-BLEU | Lỗi ASR nguồn | Nhận xét |
|:---:|:---:|:---:|:---:|---|
| **VI $\rightarrow$ EN** | **47.8** | **18.8** | WER 8.4% | Giọng đọc Supertonic tự nhiên, rõ âm |
| **EN $\rightarrow$ VI** | **46.9** | **21.0** | WER 11.3% | Piper VITS phát âm tiếng Việt chuẩn xác |
| **ZH $\rightarrow$ VI** | **36.3** | **12.5** | CER 13.0% | Bảo toàn trọn vẹn Hán tự sang tiếng Việt |
| **KO $\rightarrow$ VI** | **43.7** | **14.6** | CER 7.1% | Xử lý tốt ngữ pháp chắp dính tiếng Hàn |
| **VI $\rightarrow$ ZH** | **16.2** | **17.4** | WER 8.4% | Điểm chrF tính trên chữ Hán theo công thức ký tự |
| **VI $\rightarrow$ KO** | **13.6** | **1.2** | WER 8.4% | Âm thanh phát ra khớp 82.7 chrF so với văn bản dịch MT |

---

## 5. Các Đột phá Kỹ thuật Quan trọng

1. **Tách Encoder và Decoder Step tĩnh:**
   - Đồ thị mở rộng 16 bước (unrolled 16-step) trước đây tạo file **8.8 GB DLC** và bị giới hạn cứng ở 15 token.
   - Bản cải tiến tách Encoder + Decoder Step chỉ nặng **0.64 GB + 0.93 GB** dạng Context Binary, không giới hạn độ dài câu và chạy vòng lặp giải mã linh hoạt trên Host CPU.
2. **Khóa phiên bản QAIRT 2.51 Đồng bộ:**
   - Các Job link context binary được cố định chuẩn xác với `--qairt_version 2.51`, khớp tuyệt đối với runtime trong wheel `qai_appbuilder-2.51.40` trên cả Ubuntu 22.04 và Ubuntu 24.04.
3. **Độ chính xác số học trên HTP v73:**
   - Xử lý biểu diễn token ID lớn hơn 2048 bằng cách chia nửa `(hi, lo)` để tránh hiện tượng mất bit ở FP16. Các phép toán `int32` Cast/Mul/Add/Gather hoạt động chuẩn xác 100% trên chip silicon thật.
