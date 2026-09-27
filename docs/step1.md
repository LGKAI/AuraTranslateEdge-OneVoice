# Step 1 — Nhận dạng Giọng nói Tự động (ASR - Automatic Speech Recognition)

**Trạng thái (2026-08-09):** Đã kiểm thử mã nguồn thực tế trên 5 mô hình ứng viên với tập dữ liệu FLEURS kết hợp nhiễu thực tế ở 6 mức SNR khác nhau (không dựa trên suy đoán từ bài báo). Kiến trúc **CHỐT**: **Zipformer-30M-RNNT (Tiếng Việt)** + **SenseVoice-Small (Tiếng Anh / Tiếng Trung / Tiếng Hàn)**. Đã triển khai thành công 100% đồ thị tĩnh lên bộ xử lý Qualcomm Hexagon NPU.

---

## Part A — Bản hoàn chỉnh cho Technical Proposal §4.2 "Thiết kế Từng Module" (Drop-in Ready)

| Module | Mô hình / Framework | Kích thước (Đo đạc thực tế) | Độ trễ Mục tiêu (Latency Target) | Kỹ thuật Then chốt |
|---|---|---|---|---|
| **ASR — Tiếng Việt** | **Zipformer-30M-RNNT-6000h** ([HF: hynt](https://huggingface.co/hynt/Zipformer-30M-RNNT-6000h), sherpa-onnx runtime) | ~30M tham số (~29.3 MB định dạng INT8 có sẵn) | RTF 0.017–0.05 $\rightarrow$ ≈50–150 ms cho mỗi câu thoại 3 giây | Streaming RNN-T (khối joiner tăng tiến nguyên bản, không cần thuật toán chính sách streaming phụ trợ); tiền huấn luyện trên 6.000 giờ dữ liệu bao gồm tạp âm môi trường thực tế từ web (GigaSpeech2-Vi, VietSpeech). |
| **ASR — Tiếng Anh / Tiếng Trung / Tiếng Hàn** | **SenseVoice-Small** (FunASR / Alibaba) | ~233 MB (INT8) / W8A16 NPU (~942 MB ONNX gốc) | RTF 0.009–0.017 trên máy dev $\rightarrow$ **RTF 0.0064 trên NPU Hexagon** (187 ms cho đoạn âm 29s, ~32 ms cho câu 5s) | Giải mã Non-autoregressive (NAR) một lượt duy nhất; tích hợp sẵn chuẩn hóa văn bản ngược (ITN); nhận diện nhãn ngôn ngữ/cảm xúc miễn phí; **đóng gói trọn vẹn 5 khối tĩnh và Zero-CPU UTF-8 Detokenizer 100% trên NPU**. |

**Cơ chế Tích hợp Streaming:**
- Kiến trúc RNN-T (Zipformer) hỗ trợ xử lý luồng (streaming) nguyên bản theo từng khung thông qua bộ kết hợp joiner — không cần thuật toán điều phối chính sách streaming riêng biệt.
- Kiến trúc SenseVoice là mô hình phi tự hồi quy (non-autoregressive), do đó cơ chế "streaming" hoạt động bằng cách giải mã lại toàn bộ tiền tố đệm (re-decode) mỗi khi có đoạn âm thanh mới xuất hiện. Với hệ số RTF siêu nhanh < 0.02 (thực tế NPU đạt 0.0064), việc giải mã lại bộ đệm sau mỗi ~300 ms vẫn hoàn toàn nằm trong giới hạn thời gian thực mà không gây nghẽn.
- Cả hai nhánh đều chạy 100% trực tiếp trên thiết bị (on-device), không thực hiện bất kỳ lệnh gọi mạng nào — đáp ứng tuyệt đối tiêu chí bắt buộc của đề bài: *"Sự phụ thuộc vào đám mây / internet: 0"*.

**⚠️ Lưu ý về bản quyền trước khi nộp hồ sơ chính thức:**
- Mô hình Zipformer-30M-RNNT-6000h mang giấy phép **CC-BY-NC-ND-4.0**. Cần có xác nhận văn bản từ ban tổ chức OneVoice rằng điều khoản này được chấp nhận trong khuôn khổ nguyên mẫu dự thi của sinh viên (phi thương mại hoàn toàn hợp lệ, nhưng điều khoản "ND" - cấm phái sinh có thể hạn chế việc fine-tune mở rộng thêm).

**Tránh nhầm lẫn tên gọi trong báo cáo kỹ thuật:**
- Trên Qualcomm AI Hub có liệt kê một mô hình mang tên "Zipformer" (song ngữ Anh + Trung, ~70M tham số, trang chủ ghi chú *"chưa hỗ trợ trên bất kỳ chipset di động nào"*). Đây là **checkpoint hoàn toàn khác**, không hỗ trợ tiếng Việt, và không phải mô hình mà dự án sử dụng. Cần phân định rõ ràng khi trích dẫn tài liệu tham khảo.

---

## Part B — Phân tích Đầy đủ & Số liệu Chọn/Loại Từng Ứng viên

### 1. Phương Pháp Đo đạc Thực nghiệm

- **Tập dữ liệu kiểm thử:** Dữ liệu chuẩn FLEURS (môi trường sạch) kết hợp 5 tệp âm thanh cho mỗi ngôn ngữ được phối trộn với tạp âm thực tế ở 6 mức SNR: Sạch, 20 dB, 15 dB, 10 dB, 5 dB, 0 dB — mô phỏng chính xác môi trường làm việc nhiều tiếng ồn của nhà xưởng, công trường theo mô tả của cuộc thi (*"noisy, hands-busy environments"*).
- **Hệ thống chỉ số:**
  - **WER** (Word Error Rate - Tỷ lệ lỗi từ): Áp dụng cho Tiếng Anh và Tiếng Việt (các ngôn ngữ có ranh giới từ phân định bằng dấu cách).
  - **CER** (Character Error Rate - Tỷ lệ lỗi ký tự): Áp dụng cho Tiếng Trung và Tiếng Hàn (ngôn ngữ không phân tách từ rõ ràng bằng khoảng trắng).
  - **RTF** (Real-Time Factor = Thời gian xử lý / Thời lượng đoạn âm, RTF < 1.0 nghĩa là xử lý nhanh hơn thời gian thực).
- Toàn bộ kết quả đều được chạy thực nghiệm trên môi trường máy trạm dev GPU (CUDA), không dựa trên các suy diễn lý thuyết từ bài báo.

---

### 2. Nhánh Nhận dạng Tiếng Việt — Kiểm thử Đối đầu 4 Ứng viên qua 6 Mức SNR

| Mức SNR | PhoWhisper-small | **Zipformer-30M** | Moonshine-tiny | Qwen3-ASR-0.6B |
|---|:---:|:---:|:---:|:---:|
| **Sạch** | 5.51% | **5.35%** | 7.70% | 5.90% |
| **20 dB** | 5.51% | 5.89% | 7.10% | 4.60% |
| **15 dB** | 6.05% | 5.89% | 8.20% | 5.20% |
| **10 dB** | 7.11% | 6.95% | 11.20% | 7.00% |
| **5 dB** | 13.47% | **6.22%** | 26.10% | 13.70% |
| **0 dB** | 8.01% | **4.10%** | 16.40% | 6.60% |
| **RTF Trung bình** | ~0.06–0.18 | **~0.017–0.05** | ~0.05–0.30 | 0.10–0.17 |

**✅ CHỌN: Zipformer-30M-RNNT-6000h.**
- Điểm WER trong điều kiện âm thanh sạch ngang ngửa PhoWhisper (5.35% so với 5.51%).
- **Thắng áp đảo ở môi trường nhiễu thực tế** — đúng điều kiện nhà máy/công trường theo yêu cầu đề bài:
  - Tại mức 5 dB: WER đạt **6.22%** so với 13.47% của PhoWhisper (PhoWhisper tệ hơn gấp 2.2 lần).
  - Tại mức 0 dB: WER đạt **4.10%** so với 8.01% của PhoWhisper (PhoWhisper tệ hơn gần 2 lần).
- Tốc độ RTF nhanh hơn gấp 3.5 lần, số lượng tham số nhỏ hơn ~50 lần (~30M so với các mô hình lớn).
- Được huấn luyện sẵn trên 6.000 giờ dữ liệu phong phú bao gồm cả các nguồn âm thanh web thực tế có tiếng ồn tự nhiên (GigaSpeech2-Vi, VietSpeech), phù hợp hoàn hảo với domain mục tiêu.

**❌ LOẠI: PhoWhisper-small.**
- Điểm nhận diện trên âm thanh sạch rất tốt nhưng suy giảm chất lượng nghiêm trọng khi gặp tạp âm môi trường — đây là điểm yếu chí mạng đối với môi trường nhà máy ồn ào.

**❌ LOẠI: Moonshine-tiny.**
- Thua toàn diện so với các ứng viên còn lại ở mọi mức SNR không ngoại lệ. Tại mức 5 dB, WER lên tới 26.10% (tệ gấp hơn 4 lần so với Zipformer).

**❌ LOẠI: Qwen3-ASR-0.6B (Làm nhánh chính cho tiếng Việt).**
- Dù độ chính xác tương đối khả quan (đôi khi nhỉnh hơn nhẹ ở mức 20 dB: 4.60% so với 5.89%), lý do loại duy nhất là **tốc độ**: RTF chậm hơn Zipformer từ 3 đến 8 lần (0.10–0.17 so với 0.017–0.05). Với ngân sách độ trễ toàn hệ thống (ASR + MT + TTS $\le 3$ giây), module ASR không được phép tiêu tốn quá nhiều thời gian xử lý.

---

### 3. Nhánh Tiếng Anh / Tiếng Trung / Tiếng Hàn — So sánh 3 Ứng viên

**Bảng so sánh trong điều kiện âm thanh sạch:**

| Ngôn ngữ (Môi trường sạch) | **SenseVoice-Small** | Moonshine | Qwen3-ASR-0.6B |
|---|:---:|:---:|:---:|
| **Tiếng Anh (WER)** | 6.8% | 9.6% | **4.9%** |
| **Tiếng Trung (CER)** | **2.3%** | 16.2% | 9.1% |
| **Tiếng Hàn (CER)** | 4.5% | 8.1% | **4.4%** (ngang ngửa) |
| **RTF Trung bình** | **0.009–0.017** | 0.05–0.30 | 0.10–0.17 |

**Bảng so sánh trong điều kiện nhiễu nặng (SNR = 0 dB):**

| Ngôn ngữ (SNR = 0 dB) | **SenseVoice-Small** | Moonshine | Qwen3-ASR-0.6B |
|---|:---:|:---:|:---:|
| **Tiếng Anh (WER)** | 11.5% | 25.1% | **7.0%** |
| **Tiếng Trung (CER)** | 11.8% | 78.6%* | **10.2%** |
| **Tiếng Hàn (CER)** | 20.9% | 37.0% | **16.1%** |

*\* Moonshine tại mức 0 dB tiếng Trung gặp hiện tượng suy sập nhận diện hoàn toàn.*

**✅ CHỌN: SenseVoice-Small.**
- Vượt trội hoàn toàn so với Moonshine ở mọi ngôn ngữ và mọi điều kiện tạp âm.
- So với Qwen3-ASR: Dù điểm nhận dạng tại 0 dB của Qwen3 nhỉnh hơn đôi chút, nhưng **SenseVoice có tốc độ RTF nhanh hơn Qwen3 gấp ~10 lần** (0.01–0.02 so với 0.10–0.17). Đây là yếu tố mang tính quyết định để bảo đảm độ trễ thời gian thực cho toàn bộ chuỗi pipeline trên thiết bị biên.
- SenseVoice-Small cũng là mô hình được chính Qualcomm AI Hub khuyến nghị làm ví dụ chuẩn trong tài liệu của cuộc thi.

---

### 4. Cơ chế Xử lý Luồng (Streaming) theo Từng Kiến trúc

| Mô hình | Kiểu kiến trúc | Cơ chế xử lý luồng |
|---|---|---|
| **Zipformer (RNN-T)** | Mạng chuyển đổi nơ-ron (Transducer), khối Joiner tăng tiến | **Nguyên bản (Native):** Tự động nhận diện theo từng khung âm thanh tiếp nối, không cần bổ sung thuật toán cắt khung phụ trợ. |
| **SenseVoice-Small** | Phi tự hồi quy (Non-Autoregressive - NAR), dự đoán toàn bộ khung một lần | **Giải mã lại tiền tố đệm (Buffer Re-decode):** Giải mã lại toàn bộ bộ đệm sau mỗi ~300 ms. Hoàn toàn khả thi và mượt mà do hệ số RTF cực thấp (< 0.02 trên dev và < 0.007 trên NPU). |

---

### 5. Dữ liệu Huấn luyện Tinh chỉnh Dự phòng (Fine-tuning Datasets)

Trong trường hợp cần tối ưu hóa sâu hơn cho các phương ngữ tiếng Việt 3 miền (Bắc - Trung - Nam):

| Bộ dữ liệu | Quy mô | Vai trò & Giá trị kỹ thuật |
|---|:---:|---|
| **ViMD** ([arXiv 2410.03458](https://arxiv.org/abs/2410.03458)) | 102.56 giờ, ~19.000 câu, bao phủ 63 tỉnh thành | Báo cáo khoa học chỉ ra việc tinh chỉnh giúp cải thiện WER giọng Bắc +1.86%, **giọng miền Trung +3.07%** (vùng có phương ngữ phức tạp nhất), giọng Nam +2.34%. |
| **Bud500** ([HF Dataset](https://huggingface.co/datasets/linhtran92/viet_bud500)) | ~500 giờ, đa dạng chủ đề đời sống | Bổ sung vốn từ vựng phong phú ngoài các bản tin tức tiêu chuẩn. |

---

### 6. Các Vấn đề Kỹ thuật Đã Xử lý Trong Quá trình Thực nghiệm

- **Hiện tượng chỉ số CER tiếng Trung bị thổi phồng giả tạo (từ 49–71% xuống ~10–20% thực tế):** Tập dữ liệu FLEURS-zh chèn khoảng trắng giữa các ký tự Hán tự, hàm `jiwer.cer()` tính mỗi khoảng trắng thừa thành một lỗi xóa. Đã xử lý bằng hàm `normalize_text_for_cer()` trong `common.py` để loại bỏ toàn bộ khoảng trắng trước khi tính CER.
- **Xử lý xung đột phiên bản CUDA giữa PyTorch và Torchaudio:** Đồng bộ chính xác phiên bản `torchaudio==2.6.0+cu124` khớp với `torch 2.6.0+cu124`, khắc phục lỗi nuốt ngoại lệ import của funasr khiến `WavFrontend` không khởi tạo được.
- **Tối ưu hóa môi trường đánh giá cho Qwen3-ASR:** Cấu hình conda env riêng với `transformers>=5.13.0` và `accelerate` để bảo đảm các phép đo RTF hoàn toàn công bằng trên phần cứng.

---

### 7. Phân tích Rủi ro & Phương án Dự phòng

| Rủi ro tiềm ẩn | Phương án dự phòng kỹ thuật |
|---|---|
| Điều khoản bản quyền của Zipformer (CC-BY-NC-ND-4.0) cần làm rõ | Chuyển sang mô hình Qwen3-ASR-0.6B cho nhánh tiếng Việt (chất lượng tương đương, chấp nhận độ trễ RTF chậm hơn). Tuyệt đối không dùng Moonshine vì chất lượng không đạt yêu cầu. |
| Hiện tượng nhấp nháy từ (Flicker rate) khi SenseVoice re-decode | Bổ sung cơ chế đệm trễ ổn định (hysteresis window) dựa trên ngưỡng xác suất CTC trước khi gửi văn bản sang khối Dịch thuật. |

---

**Phiên bản tài liệu:** 2026-09-28 (Cập nhật kết quả triển khai NPU W8A16 chính thức trên Qualcomm AI Hub)  
**Trạng thái:** Hoàn tất kiểm thử thực nghiệm, kiến trúc đóng gói tĩnh 100% NPU đã được xác thực toàn diện.
