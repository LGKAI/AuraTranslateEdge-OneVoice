# Step 1 — Nhận dạng Giọng nói Tự động (Automatic Speech Recognition - ASR)

Mô-đun Step 1 chịu trách nhiệm chuyển đổi âm thanh đầu vào thành văn bản (Speech-to-Text). Đặc thù của dự án yêu cầu mô hình phải chạy **hoàn toàn offline (zero cloud dependency)**, có khả năng **chịu nhiễu công nghiệp tốt (SNR 0dB – 20dB)** cho môi trường nhà máy, công trường ồn ào, và tốc độ xử lý phải cực nhanh (RTF < 0.05) để đáp ứng chuẩn thời gian thực (real-time).

---

## 1. Các mô hình đã chọn

Sau khi đánh giá và benchmark thực nghiệm trên dữ liệu chuẩn FLEURS và môi trường pha trộn nhiễu thực tế, dự án chốt sử dụng kiến trúc định tuyến 2 mô hình:

*   **Zipformer-30M-RNNT (Tiếng Việt):**
    *   **Kiến trúc:** Streaming RNN-T (Recurrent Neural Network Transducer), runtime qua `sherpa-onnx`.
    *   **Đặc điểm:** Huấn luyện trên 6.000 giờ dữ liệu tiếng Việt thực tế (bao gồm GigaSpeech2-Vi và VietSpeech có nhiều tạp âm tự nhiên). Mô hình siêu nhẹ (~29.3 MB INT8) và có khả năng streaming tự nhiên theo từng khung âm thanh.
    *   **Hiệu năng thực đo:** WER sạch **5.35%**, đặc biệt xuất sắc ở môi trường cực ồn **SNR 0dB đạt WER 4.10%** (vượt trội hoàn toàn so với PhoWhisper-small 8.01% hay Moonshine-tiny 16.4%). RTF trung bình **0.017 – 0.05**.
*   **SenseVoice-Small (Tiếng Anh / Tiếng Trung / Tiếng Hàn):**
    *   **Kiến trúc:** Non-autoregressive (50 layers Transformer + CTC head), giải mã toàn bộ câu trong một lượt (single-pass forward).
    *   **Đặc điểm:** Tốc độ xử lý cực nhanh (RTF **0.009 – 0.017** trên GPU/CPU), tích hợp sẵn bộ chuẩn hoá văn bản ITN (Inverse Text Normalization) và tự động nhận diện ngôn ngữ (Language Identification - LID).
    *   **Hiệu năng thực đo:** CER tiếng Trung **2.3%**, CER tiếng Hàn **4.5%**, WER tiếng Anh **6.8%**. Cung cấp sự cân bằng hoàn hảo giữa độ chính xác và tốc độ xử lý.

> [!TIP]
> **Triển khai NPU & Lượng tử hóa (Step 4):** Toàn bộ mã nguồn xuất mô hình, lượng tử hóa W8A16, compile NPU, benchmark 90 câu và báo cáo kỹ thuật chi tiết của SenseVoice-Small được tổng hợp thống nhất tại:
> 👉 **[`src/step4_quantization/step1_asr/sensevoice/README.md`](../step4_quantization/step1_asr/sensevoice/README.md)**.

---

## 2. Kết quả Benchmark đối đầu ở Step 1

### Nhánh Tiếng Việt (Đo trên FLEURS + Nhiễu công nghiệp 6 mức SNR)

| Mức nhiễu (SNR) | PhoWhisper-small | **Zipformer-30M** *(CHỌN)* | Moonshine-tiny | Qwen3-ASR-0.6B |
|:---:|:---:|:---:|:---:|:---:|
| **Sạch** | 5.51% | **5.35%** | 7.7% | 5.9% |
| **20 dB** | 5.51% | 5.89% | 7.1% | 4.6% |
| **15 dB** | 6.05% | 5.89% | 8.2% | 5.2% |
| **10 dB** | 7.11% | 6.95% | 11.2% | 7.0% |
| **5 dB** | 13.47% | **6.22%** | 26.1% | 13.7% |
| **0 dB (Cực ồn)** | 8.01% | **4.10%** | 16.4% | 6.6% |
| **RTF trung bình** | ~0.06 – 0.18 | **~0.017 – 0.05** | ~0.05 – 0.3 | 0.10 – 0.17 |

*Lý do chọn Zipformer-30M:* Duy trì độ chính xác ổn định vượt trội khi nhiễu tăng cao, RTF nhanh hơn 3.5× so với PhoWhisper và nhanh hơn 3–8× so với Qwen3-ASR.

### Nhánh Ngoại ngữ: Anh / Trung / Hàn

| Ngôn ngữ | **SenseVoice-Small** *(CHỌN)* | Moonshine-tiny | Qwen3-ASR-0.6B |
|---|:---:|:---:|:---:|
| **Tiếng Anh (WER)** | 6.8% *(0dB: 11.5%)* | 9.6% *(0dB: 25.1%)* | **4.9%** *(0dB: 7.0%)* |
| **Tiếng Trung (CER)** | **2.3%** *(0dB: 11.8%)* | 16.2% *(0dB: 78.6%)* | 9.1% *(0dB: 10.2%)* |
| **Tiếng Hàn (CER)** | 4.5% *(0dB: 15.3%)* | 8.1% *(0dB: 28.5%)* | **4.4%** *(0dB: 14.1%)* |
| **RTF trung bình** | **0.009 – 0.017** | 0.05 – 0.30 | 0.10 – 0.17 |

*Lý do chọn SenseVoice-Small:* Tốc độ xử lý nhanh nhất trong mọi mô hình thử nghiệm, giải mã 1 câu chỉ mất ~30–50ms, hỗ trợ đồng thời cả 3 ngoại ngữ và tích hợp sẵn LID định tuyến.

---

## 3. Cấu trúc thư mục & Các tệp mã nguồn

Thư mục `src/step1_asr/` được tổ chức thành các nhóm chức năng rõ ràng phục vụ Step 1:

### 📁 Dữ liệu & Benchmark đánh giá
*   `fetch_asr_data.py`: Tải tập dữ liệu âm thanh kiểm thử từ chuẩn FLEURS về `data/asr/`.
*   `mix_asr_noise.py`: Trộn các mức nhiễu công nghiệp (SNR từ 0dB đến 20dB) vào file âm thanh sạch (`data/asr_mixed/`).
*   `run_all_asr.py`: Script chạy tự động toàn bộ benchmark cho tất cả các mô hình.
*   `test_asr_zipformer.py`: Kịch bản đánh giá chuyên biệt cho **Zipformer-30M (Tiếng Việt)**.
*   `test_asr_multi.py`: Kịch bản đánh giá chuyên biệt cho **SenseVoice-Small (Anh / Trung / Hàn)**.
*   `test_asr_vi.py`, `test_asr_moonshine.py`, `test_asr_qwen.py`: Mã kiểm thử cho các mô hình đã bị loại (PhoWhisper, Moonshine, Qwen3) để đối chiếu số liệu.

### 📁 Tài liệu tham chiếu kiến trúc NPU
*   Tài liệu kỹ thuật của Trần Quốc Khanh về cơ chế End-to-End ASR trên NPU Qualcomm Hexagon: [`src/step4_quantization/step1_asr/zipformer/README.pdf`](../step4_quantization/step1_asr/zipformer/README.pdf).
*   Thư mục mã nguồn và báo cáo triển khai NPU SenseVoice-Small: [`src/step4_quantization/step1_asr/sensevoice/`](../step4_quantization/step1_asr/sensevoice/).

---

## 4. Hướng dẫn cài đặt và chạy kiểm thử

### Cài đặt môi trường
Từ thư mục gốc của dự án, cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

> **Lưu ý phụ:** Script kiểm thử `Qwen3-ASR` (`test_asr_qwen.py`) yêu cầu `transformers>=5.13.0`. Nếu cần chạy lại mô hình đối chứng này, nên tạo môi trường ảo riêng để tránh xung đột phiên bản với `FunASR`.

### Các bước thực thi

**Bước 1: Chuẩn bị dữ liệu và tạo tập kiểm thử nhiễu**
```bash
cd src/step1_asr
python fetch_asr_data.py     # Tải dữ liệu FLEURS chuẩn
python mix_asr_noise.py      # Trộn nhiễu mô phỏng công xưởng
```

**Bước 2: Chạy benchmark từng mô hình đã chọn**
```bash
# Đánh giá Zipformer-30M trên tiếng Việt
python test_asr_zipformer.py

# Đánh giá SenseVoice-Small trên Anh / Trung / Hàn
python test_asr_multi.py
```
*Kết quả chi tiết được tự động xuất ra file CSV tại thư mục `outputs/` (`asr_multi_results.csv`, `asr_zipformer_results.csv`).*

> Đối với các bước xuất mô hình NPU, lượng tử hóa W8A16 và kiểm thử phần cứng, vui lòng chuyển sang thư mục **[`src/step4_quantization/sensevoice/`](../step4_quantization/sensevoice/)**.