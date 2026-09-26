# Step 5 — Pipeline Dịch Giọng nói sang Giọng nói Toàn trình (Speech-to-Speech - S2S)

Tài liệu chi tiết các bước thành phần liên quan:
*   Tiền xử lý âm thanh: [`../../docs/step0.md`](file:///d:/ChuyenNganhAI/AuraTranslateEdge-OneVoice/docs/step0.md)
*   Nhận dạng giọng nói (ASR): [`../../docs/step1.md`](file:///d:/ChuyenNganhAI/AuraTranslateEdge-OneVoice/docs/step1.md)
*   Dịch máy đa ngữ (MT): [`../../docs/step2.md`](file:///d:/ChuyenNganhAI/AuraTranslateEdge-OneVoice/docs/step2.md)
*   Tổng hợp giọng nói (TTS): [`../../docs/step3.md`](file:///d:/ChuyenNganhAI/AuraTranslateEdge-OneVoice/docs/step3.md)
*   Lượng tử hoá phần cứng & Thiết bị biên: [`../../docs/step4.md`](file:///d:/ChuyenNganhAI/AuraTranslateEdge-OneVoice/docs/step4.md)

Mô-đun Step 5 chịu trách nhiệm ghép nối tuần hoàn tất cả các mô hình đã được kiểm chứng và lượng tử hoá từ Step 1 đến Step 4 thành một chuỗi **Speech-to-Speech (S2S)** hoàn chỉnh, đo đạc độ trễ từng chặng, độ trễ toàn trình (End-to-End Latency) và chất lượng chuyển đổi đa ngữ hai chiều giữa **Tiếng Việt và 3 ngoại ngữ: Anh, Trung, Hàn**.

---

## 1. Kiến trúc Pipeline Toàn trình

Chuỗi xử lý S2S hoạt động hoàn toàn **Offline (Zero Cloud Dependency)** trên thiết bị biên:

```
Tín hiệu Âm thanh Đầu vào (WAV 16 kHz Mono)
       ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 1 — ASR (Nhận dạng Giọng nói Đa ngữ)                                   │
│   • Tiếng Việt : Zipformer-30M-RNNT INT8 qua sherpa-onnx (CPU)             │
│   • Anh / Trung / Hàn : SenseVoice-Small (funasr / ONNX INT8)               │
└─────────────────────────────────────────────────────────────────────────────┘
       ↓ Văn bản nhận dạng (Tự động lower-case nếu là tiếng Việt)
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 2 — MT (Dịch máy Đa ngữ 6 chiều)                                       │
│   • NLLB-200-distilled-600M INT8 nén qua CTranslate2                        │
│   • Hỗ trợ: Vi ↔ En, Vi ↔ Zh, Vi ↔ Ko (< 0.5s - 0.9s/câu)                   │
└─────────────────────────────────────────────────────────────────────────────┘
       ↓ Văn bản dịch ở ngôn ngữ đích
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 3 — TTS (Tổng hợp Giọng nói Đa ngữ)                                    │
│   • Tiếng Việt : Piper ONNX (vi_VN-vais1000-medium, RTF 0.144)             │
│   • Tiếng Anh & Hàn : Supertonic 3 (Bản nén lai Step 4, vocoder FP32)        │
│   • Tiếng Trung : MeloTTS-ZH (Tối ưu disable_bert=True)                     │
└─────────────────────────────────────────────────────────────────────────────┘
       ↓
Tín hiệu Âm thanh Đầu ra (WAV giọng nói dịch tự nhiên, chuẩn xác)
```

### Các chế độ phần cứng hỗ trợ:

Script [`pipeline_s2s.py`](file:///d:/ChuyenNganhAI/AuraTranslateEdge-OneVoice/src/step5_pipeline/pipeline_s2s.py) hỗ trợ 2 chế độ thiết bị đối sánh:

*   **Chế độ `--device cpu`:** Toàn bộ pipeline chạy trên CPU. SenseVoice chạy PyTorch FP32 (đo thực tế nhanh hơn 10–20× so với bản ONNX CPU thuần; bản ONNX INT8 được dành riêng cho đường biên dịch QNN/NPU).
*   **Chế độ `--device cuda`:** Tận dụng GPU NVIDIA để tăng tốc SenseVoice và MeloTTS (PyTorch CUDA), đồng thời chạy NLLB INT8 qua backend CTranslate2 CUDA. Các mô-đun Zipformer, Piper và Supertonic duy trì chạy CPU-only (do giới hạn các gói pre-built wheel của sherpa-onnx và ONNX Runtime trên môi trường Windows).

### Bảng tổng hợp mô hình & Dung lượng triển khai thực tế (Deploy Footprint):

| Thành phần | Mô hình triển khai | Định dạng & Kỹ thuật | Kích thước On-disk |
|---|---|---|:---:|
| **ASR (Vi)** | Zipformer-30M-RNNT | ONNX INT8 (`sherpa-onnx`) | **29.3 MB** |
| **ASR (En/Zh/Ko)** | SenseVoice-Small | ONNX INT8 (QNN Deploy) / PyTorch (Dev) | **233 MB** *(Deploy)* |
| **MT (6 chiều)** | NLLB-200-distilled-600M | CTranslate2 INT8 | **594 MB** |
| **TTS (Vi)** | Piper (`vais1000-medium`) | ONNX FP32 | **61 MB** |
| **TTS (En/Ko)** | Supertonic 3 | Mixed INT8 (3 submodel INT8 + Vocoder FP32) | **178 MB** |
| **TTS (Zh)** | MeloTTS-ZH | PyTorch / Qualcomm AI Hub NPU | **199 MB** |
| **TỔNG CỘNG** | **Toàn bộ Pipeline S2S** | **4 Ngôn ngữ, Chạy Offline 100%** | **≈ 1.38 GB** |

> [!TIP]
> Toàn bộ pipeline chỉ chiếm **~1.38 GB bộ nhớ lưu trữ**, nằm thoải mái trong dung lượng RAM 8GB của bo mạch biên **Rubik Pi 3 (Qualcomm QCS6490)** hoặc kit **Qualcomm Dragonwing IQ-9075 EVK**, đảm bảo tính khả thi cao nhất khi triển khai thực địa.

---

## 2. Các phát hiện kỹ thuật & Giải pháp khi ghép nối Pipeline

Trong quá trình ghép nối thực tế giữa các tầng mô hình, nhiều vấn đề tương thích phát sinh đã được điều tra và xử lý triệt để:

### 1. Xử lý lỗi văn bản viết hoa toàn bộ (ALL-CAPS) của Zipformer
*   **Hiện tượng:** Mô hình Zipformer-30M xuất văn bản tiếng Việt viết HOA toàn bộ (ví dụ: `KHI MỘT CUỘC ĐẤU TRANH...`).
*   **Vấn đề:** Khi nạp nguyên văn chuỗi viết hoa này vào NLLB-200, cơ chế tokenization BPE bị lệch phân phối, khiến mô hình dịch sai lệch hoàn toàn (ví dụ: bị dịch thành *"When a Struggle Curses the World..."* thay vì nghĩa đúng).
*   **Giải pháp:** Tự động hạ về chữ thường (`hyp.lower()`) trước khi nạp vào MT đối với tiếng Việt. Mô hình NLLB sẽ tự động khôi phục viết hoa/thường chuẩn ngữ pháp ở ngôn ngữ đích.

### 2. Khắc phục tính bất định của Supertonic tiếng Hàn (Quality-gated Retry)
*   **Hiện tượng:** Bộ lấy mẫu flow-matching của Supertonic có tính ngẫu nhiên cao, xuất hiện hiện tượng lặp âm tiết (repetition artifact) ở khoảng 50% số lần chạy đơn lẻ (kể cả khi tăng tham số `total_steps` từ 5 lên 8, xem nhật ký kiểm chứng [`outputs/ko_reliability.log`](file:///d:/ChuyenNganhAI/AuraTranslateEdge-OneVoice/outputs/ko_reliability.log)).
*   **Giải pháp:** Xây dựng cơ chế **Thử lại có kiểm soát chất lượng (Quality-gated retry)**: Sau khi TTS sinh âm thanh, hệ thống gọi nhanh mô-đun SenseVoice ASR đã nạp sẵn trong RAM để nhận dạng ngược lại. Nếu độ lệch ký tự $\text{CER} > 5\%$, hệ thống tự động sinh lại (tối đa 5 lần) và chọn bản phát âm chuẩn xác nhất.

### 3. Tối ưu Prosody BERT của MeloTTS
*   **Hiện tượng:** Mặc định MeloTTS sẽ ngầm tải một mô hình BERT dự đoán ngữ điệu nặng ~1.35 GB qua mạng, gây chậm trễ và nguy cơ lỗi kết nối khi thiết bị mất mạng.
*   **Giải pháp:** Thiết lập `disable_bert=True` trong pipeline. Giọng nói tiếng Trung vẫn giữ được độ tự nhiên chuẩn xác mà loại bỏ hoàn toàn việc tải ngầm mô hình nặng.

### 4. Đánh giá chất lượng 2 đầu (Round-trip Quality Scoring)
*   Không chỉ đo thời gian xử lý từng chặng, pipeline tự động tính điểm chất lượng:
    1.  **Input ASR Score:** Đo WER/CER của văn bản ASR so với câu mẫu chuẩn (`data/asr/manifest.json`).
    2.  **Round-trip Score:** Dùng chính ASR nhận dạng lại file âm thanh WAV do TTS phát ra, đối chiếu với câu văn bản dịch MT để kiểm chứng độ rõ ràng (intelligibility) của giọng đọc.

---

## 3. Cấu trúc thư mục & Tệp mã nguồn

Thư mục `src/step5_pipeline/` bao gồm:

*   [`pipeline_s2s.py`](file:///d:/ChuyenNganhAI/AuraTranslateEdge-OneVoice/src/step5_pipeline/pipeline_s2s.py): Toàn bộ mã nguồn tích hợp chuỗi Speech-to-Speech, bao gồm các lớp nạp mô hình ASR, MT, TTS, cơ chế lọc văn bản, vòng lặp retry và xuất báo cáo hiệu năng ra file CSV.

---

## 4. Yêu cầu chuẩn bị (Prerequisites)

Do thư viện `tokenizers` của MeloTTS cần môi trường Python $\le 3.11$ trên Windows (tránh lỗi yêu cầu trình biên dịch Rust khi cài đặt trên Python 3.13), khuyến nghị tạo môi trường ảo chuyên biệt cho pipeline:

```bash
# Tạo và kích hoạt môi trường Python 3.11:
python3.11 -m venv venv_pipeline
venv_pipeline\Scripts\activate
pip install -r requirements.txt
```

### Các tài nguyên mô hình bắt buộc phải có trước khi chạy:

1.  **NLLB-600M INT8:** Chạy kiểm chứng Step 2 để tạo thư mục mô hình nén:
    ```bash
    python src/step2_mt/verify_nllb_int8.py
    # -> Tạo thư mục outputs/nllb-ct2-int8/
    ```
2.  **Supertonic Deploy Model:** Chạy lượng tử hoá lai Step 4:
    ```bash
    python src/step3_tts/quantize_supertonic.py
    # -> Tạo thư mục outputs/supertonic-deploy/
    ```
3.  **Tệp từ điển Zipformer:** Tự động sinh khi chạy kiểm thử Zipformer lần đầu:
    ```bash
    python src/step1_asr/test_asr_zipformer.py
    # -> Tạo tệp third_party/zipformer/.../tokens.generated.txt
    ```
4.  **Giọng đọc Piper tiếng Việt:** Tải file mô hình giọng đọc:
    ```bash
    cd src/step3_tts
    python -m piper.download_voices vi_VN-vais1000-medium
    ```

---

## 5. Hướng dẫn chạy Pipeline & Báo cáo kết quả

### Thực thi trên CPU:
```bash
python src/step5_pipeline/pipeline_s2s.py --device cpu
```

### Thực thi trên GPU NVIDIA (CUDA):
```bash
python src/step5_pipeline/pipeline_s2s.py --device cuda
```

### Kết quả đầu ra:
Sau khi hoàn tất, script sẽ tự động tạo:
*   **Bảng báo cáo chi tiết:** [`outputs/pipeline_results_<device>.csv`](file:///d:/ChuyenNganhAI/AuraTranslateEdge-OneVoice/outputs/) ghi nhận chi tiết: thời gian chạy ASR (`asr_sec`), MT (`mt_sec`), TTS (`tts_sec`), tổng thời gian toàn trình (`e2e_sec`), điểm số chất lượng đầu vào (`asr_score`) và điểm độ rõ phát âm (`rt_score`).
*   **Thư mục lưu trữ âm thanh:** `outputs/pipeline/<device>/*.wav` chứa toàn bộ các file âm thanh đã được dịch và tổng hợp hoàn chỉnh.
*   **Bảng tổng kết hiệu năng in trực tiếp ra màn hình terminal:**
    ```text
    [pipeline] === SUMMARY (cuda, n=8) ===
      mean ASR stage : 0.045s
      mean MT  stage : 0.210s
      mean TTS stage : 0.480s
      mean end-to-end: 0.735s
      mean input-ASR score : 0.0512
      mean round-trip score: 0.0824
    ```
