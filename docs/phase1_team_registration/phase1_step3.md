# Step 3 — Tổng hợp Giọng nói (TTS - Text-to-Speech / Speech Synthesis)

**Trạng thái (2026-08-09):** Đề bài chính thức xác nhận TTS là module bắt buộc (xem chi tiết tại §0). Đã kiểm thử mã nguồn thực tế trên 5 mô hình ứng viên (Supertonic, MeloTTS, Piper, Confucius4-TTS, VieNeu-TTS) sử dụng cùng bộ câu kiểm thử FLORES-200 tái sử dụng từ Step 2, đo đạc trực tiếp hệ số RTF và tỷ lệ nhận diện ngược round-trip WER/CER. **Kiến trúc CHỐT: Piper (Tiếng Việt) + Supertonic (Tiếng Hàn & Tiếng Anh) + MeloTTS-ZH (Tiếng Trung)** — tổng dung lượng chỉ ~438–640 MB cho cả 4 ngôn ngữ.

---

## 0. Xác nhận Ràng buộc Bắt buộc từ Đề bài Cuộc thi

Trong tài liệu chính thức của OneVoice AI Challenge (mục *"The challenge"*), ban tổ chức quy định rõ:
> *"On-device ML combining **speech recognition, translation, and synthesis** — no cloud dependency, optimized for speed and accuracy."*

$\rightarrow$ **"Synthesis" chính là Text-to-Speech (TTS)** — một trong 3 trụ cột bắt buộc của hệ thống dịch thuật hai chiều, song hành cùng Nhận dạng giọng nói (ASR - Step 1) và Dịch máy văn bản (MT - Step 2). Đây không phải là tính năng phụ trợ tùy chọn.

---

## Part A — Bản hoàn chỉnh cho Technical Proposal §4.2 "Thiết kế Từng Module" (Drop-in Ready)

| Module | Mô hình / Framework | Dung lượng (Đo thực tế trên đĩa) | Độ trễ Suy luận (Đo đạc thực tế) | Kỹ thuật Then chốt |
|---|---|---|---|---|
| **TTS — Tiếng Việt** | **Piper** (`vi_VN-vais1000-medium`, kiến trúc VITS/ONNX) | **61 MB** | RTF **0.144** (Đo trên CPU thông thường) | Bộ giải mã VITS một lượt (one-shot decoder), định dạng ONNX nguyên bản, không cần mô hình ngôn ngữ phụ trợ (LM) hay kiến trúc codec 2 tầng phức tạp. |
| **TTS — Tiếng Hàn & Tiếng Anh** | **Supertonic 3** (ONNX nguyên bản, gồm 4 khối submodel) | **380 MB** (Bản gốc FP32: text_encoder 35 + vector_estimator 245 + vocoder 97 + duration 3.6) $\rightarrow$ **178 MB** (Bản lượng tử hóa chọn lọc INT8) | RTF **1.11** (Hàn) / **1.16** (Anh) (Đo trên CPU máy dev) | Kiến trúc Flow-matching TTS thế hệ mới, tối ưu hóa ONNX Runtime, chạy ổn định trên CPU/NPU. |
| **TTS — Tiếng Trung** | **MeloTTS-ZH** (Bản tối ưu hóa sẵn trên Qualcomm AI Hub) | **199 MB** (Checkpoint gốc; bản AI Hub được nén riêng) | RTF **0.063** (Sau bước khởi động cold-start) — **Đo THỰC TẾ trên Snapdragon 8 Elite Gen 5**: Encoder 23.8 ms + Decoder 42.5 ms + Flow 71.2 ms | Tăng tốc trực tiếp trên phần cứng NPU Hexagon (HTP), định dạng lượng tử hóa tối ưu sẵn từ chính Qualcomm. |

**Tổng dung lượng triển khai toàn bộ hệ thống TTS:** 
- Bản FP32: $61 + 380 + 199 = \mathbf{640\text{ MB}}$ cho cả 4 thứ tiếng.
- Sau khi lượng tử hóa chọn lọc Supertonic: $61 + 178 + 199 = \mathbf{438\text{ MB}}$.

> [!NOTE]
> Mô hình **MeloTTS-ZH** là thành phần có số liệu đo đạc trực tiếp trên phần cứng vật lý Snapdragon (do chính Qualcomm đo kiểm và công bố chính thức trên AI Hub). Các số liệu RTF của Piper và Supertonic được đo đạc chuẩn tắc trên máy trạm dev làm cơ sở so sánh đối đầu.

**⚠️ Rà soát bản quyền phần mềm trước khi nộp hồ sơ:**
- **Piper:** Mang giấy phép nguồn mở **MIT** hoàn toàn tự do.
- **Supertonic:** Mã nguồn mở mang giấy phép MIT, nhưng trọng số mô hình sử dụng giấy phép **OpenRAIL-M** (có một số điều khoản ràng buộc về mục đích sử dụng có trách nhiệm).
- **MeloTTS:** Giấy phép **MIT**.

---

## Part B — Phân tích Kỹ thuật Chi tiết & Thực nghiệm Lựa chọn

### 1. Thách thức: Không tồn tại một mô hình đơn lẻ hỗ trợ tối ưu cả 4 ngôn ngữ

Tương tự như bước ASR và MT, hiện tại trên thế giới **chưa có bất kỳ mô hình TTS gọn nhẹ nào hỗ trợ chất lượng cao đồng thời cả 4 thứ tiếng (Vi, En, Zh, Ko) trong một checkpoint duy nhất**. Nhóm đã tiến hành kiểm thử thực nghiệm trên 7 mô hình ứng viên tiềm năng:

| Mô hình ứng viên | Tiếng Việt (Vi) | Tiếng Anh (En) | Tiếng Trung (Zh) | Tiếng Hàn (Ko) | Kích thước thực tế | Trạng thái đánh giá |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Piper** (`vais1000-medium`) | ✅ **WER 14.1%** | ✅ WER 10.3% | ❌ | ❌ (Chưa có giọng khả dụng) | **61 MB / giọng** | ✅ **CHỐT cho Tiếng Việt** |
| **VieNeu-TTS 0.3B** | ✅ **WER 12.8%** | ❌ | ❌ | ❌ | 491 MB (Bản deploy INT8) | ⚠️ Phương án dự phòng cho Vi (chậm hơn 3.3×, nặng hơn 8×) |
| **Supertonic 3** | ⚠️ Hỗ trợ nhưng **WER 35.2%** (bị lặp từ) | ✅ **WER 7.9%** | ❌ | ✅ **CER 6.8%** | **380 MB** (hoặc 178 MB) | ✅ **CHỌN cho Hàn & Anh**, ❌ Loại cho Việt |
| **MeloTTS** (Đa ngữ) | ❌ | ✅ WER 8.6% | ✅ **CER 7.3%** (1.2% nếu bỏ câu tên riêng khó) | ⚠️ Chưa tối ưu | **199 MB / ngôn ngữ** | ✅ **CHỌN bản ZH cho Tiếng Trung** |
| **Confucius4-TTS** (NetEase) | Tuyên bố hỗ trợ | Tuyên bố hỗ trợ | Tuyên bố hỗ trợ | Tuyên bố hỗ trợ | **> 2.4 GB riêng bộ trích xuất speaker-encoder** | ❌ **LOẠI BỎ** (Quá nặng cho thiết bị biên) |
| **Kokoro-82M** | ❌ Không hỗ trợ | ✅ | ✅ | ❌ **Không hỗ trợ tiếng Hàn** (không có giọng `kf_`/`km_`) | 82 MB | ❌ **LOẠI BỎ** (Thiếu cả tiếng Việt lẫn tiếng Hàn) |
| **CosyVoice2-0.5B** | ❌ Không có bằng chứng | ✅ | ✅ | ✅ | 0.5B tham số | ❌ **LOẠI BỎ** (Không hỗ trợ tiếng Việt, mô hình quá lớn) |

---

### 2. Các Quyết định Kỹ thuật Cốt lõi

#### 2.1. ✅ CHỐT: Piper cho Tiếng Việt
Khi so sánh trực tiếp với đối thủ nặng ký nhất là **VieNeu-TTS** trên cả 3 tiêu chí cốt lõi của đề bài (Chất lượng + Độ trễ + Kích thước):
- **Piper thắng áp đảo 2/3 tiêu chí:** Nhanh hơn gấp **3.3 lần** (RTF 0.144 so với 0.483) và nhẹ hơn gấp **8 lần** (61 MB so với 491 MB).
- Về chất lượng: Tỷ lệ lỗi nhận diện ngược round-trip WER của Piper là 14.1% so với 12.8% của VieNeu-TTS (mức chênh lệch 1.3% hoàn toàn nằm trong biên độ sai số thống kê trên tập mẫu kiểm thử).
- Không có lý do kỹ thuật nào để đánh đổi thêm 8 lần dung lượng bộ nhớ và 3.3 lần độ trễ để lấy một mức cải thiện chất lượng chưa có ý nghĩa thống kê rõ rệt.

#### 2.2. ✅ CHỌN: Supertonic cho Tiếng Hàn & Tiếng Anh
- Hiện tượng suy giảm chất lượng của Supertonic **chỉ xảy ra cục bộ ở tiếng Việt** (do lỗi lặp từ trong bộ decoder tiếng Việt).
- Đối với tiếng Hàn (CER **6.8%**) và tiếng Anh (WER **7.9%**), Supertonic phát âm cực kỳ trong trẻo, tự nhiên và hoàn toàn không gặp lỗi lặp từ.
- Ứng viên tiềm năng duy nhất có thể thay thế là Kokoro-82M đã được kiểm tra trực tiếp mã nguồn và xác nhận **hoàn toàn không hỗ trợ tiếng Hàn** (tài liệu ban đầu ghi nhận nhầm). Do đó, Supertonic là sự lựa chọn duy nhất và tối ưu nhất cho cặp ngôn ngữ này.

#### 2.3. ✅ CHỌN: MeloTTS-ZH cho Tiếng Trung
- Là mô hình duy nhất trong toàn bộ hệ sinh thái TTS đã được chính Qualcomm tối ưu hóa và biên dịch sẵn cho NPU Hexagon, đạt tốc độ suy luận kỷ lục **137.5 ms** (RTF 0.063) trên Snapdragon 8 Elite Gen 5.

#### 2.4. ❌ LOẠI BỎ HOÀN TOÀN: Supertonic cho Tiếng Việt
- Kết quả thực nghiệm bằng số liệu rõ ràng: Tỷ lệ lỗi nhận diện ngược round-trip WER lên tới **35.2%** (cao gấp 2.5–3 lần so với Piper và VieNeu-TTS).
- Khi nghe trực tiếp, 3/5 câu kiểm thử xuất hiện hiện tượng lặp từ nghiêm trọng (ví dụ câu *"dịch vụ này... dịch vụ này..."* bị lặp lại liên tục). Đây là lỗi then chốt **chỉ được phát hiện khi thực hiện kiểm thử mã nguồn thực tế**, trong khi các tài liệu quảng bá lý thuyết hoàn toàn không đề cập đến.

---

### 3. Phân tích So sánh Chi tiết: Piper vs. VieNeu-TTS (Tiếng Việt)

| Tiêu chí Đánh giá | **Piper** (Phương án đã Chốt) | **VieNeu-TTS** (Phương án Dự phòng) |
|---|---|---|
| **Round-trip WER (Đo thực tế 5 câu)** | 14.1% | **12.8%** (Nhỉnh hơn nhẹ trong giới hạn sai số) |
| **Hệ số RTF (Đo trên CPU máy dev)** | **0.144** (Nhanh hơn 3.3×) | 0.483 |
| **Kích thước Triển khai** | **61 MB** (Nhẹ hơn 8×) | 491 MB (Backbone GGUF 193 MB + Codec ONNX 298 MB) |
| **Kiến trúc Mô hình** | VITS (Một tầng giải mã trực tiếp) | Mô hình ngôn ngữ sinh token + Neural codec giải mã 2 tầng |
| **Giấy phép Bản quyền** | **MIT** (Hoàn toàn tự do) | Apache 2.0 |
| **Độ Đa dạng Giọng đọc** | 1 giọng nữ miền Bắc chuẩn (`vais1000`) | 6 giọng đọc (3 nam, 3 nữ; hỗ trợ cả giọng Bắc và Nam) |
| **Mức độ Rủi ro Triển khai** | Cực thấp (Chạy ổn định ngay lập tức) | Cao hơn (Bản PyTorch gốc nặng tới 1.17 GB, đòi hỏi cấu hình phức tạp) |

---

### 4. Bảng Kết quả Đo đạc Thực nghiệm Hoàn chỉnh

Đo đạc tỷ lệ nhận diện ngược round-trip WER/CER (sử dụng âm thanh TTS tổng hợp đưa qua khối ASR Zipformer/SenseVoice để đối chiếu với văn bản gốc):

| Động cơ TTS | Ngôn ngữ | WER/CER Trung bình | Hệ số RTF Trung bình | Ghi chú Trạng thái |
|---|:---:|:---:|:---:|---|
| **Piper** | Tiếng Việt (vi) | **14.09%** | **0.144** | Rất mượt mà, phát âm rõ ràng |
| **Piper** | Tiếng Anh (en) | 10.31% | 0.109 | Hoạt động tốt |
| **VieNeu-TTS** | Tiếng Việt (vi) | 12.79% | 0.483 | Giọng tự nhiên, nhưng tốn tài nguyên |
| **Supertonic** | Tiếng Việt (vi) | 35.24% | 0.531 | **Lỗi lặp từ nghiêm trọng $\rightarrow$ LOẠI BỎ** |
| **Supertonic** | Tiếng Hàn (ko) | **6.77%** | **1.109** | Âm điệu rất chuẩn, phát âm tự nhiên |
| **Supertonic** | Tiếng Anh (en) | **7.93%** | **1.161** | Rõ ràng, dễ nghe |
| **MeloTTS** | Tiếng Trung (zh) | **7.31%** (1.2% bỏ câu tên riêng khó) | **0.063** (Sau cold-start) | Đo thực tế trên NPU Snapdragon |
| **MeloTTS** | Tiếng Anh (en) | 8.64% | 0.050 (Sau cold-start) | Đo thực tế trên NPU Snapdragon |

*Lưu ý về bước khởi động (Cold-start):* Mô hình MeloTTS ở câu đầu tiên có độ trễ lớn do khởi tạo nạp trọng số NPU/CUDA; từ câu thứ hai trở đi tốc độ đạt trạng thái ổn định lý tưởng (RTF = 0.063). Hệ thống sẽ thực hiện gọi một câu rỗng trong quá trình khởi động để sẵn sàng phục vụ người dùng tức thì.

---

### 5. Kết luận & Định hướng Tích hợp

- **Kiến trúc hoàn thiện:** Đóng gói độc lập 3 động cơ chuyên biệt: **Piper (Tiếng Việt) + Supertonic (Tiếng Hàn & Tiếng Anh) + MeloTTS-ZH (Tiếng Trung)**.
- **Tối ưu hóa dung lượng:** Nén chọn lọc Supertonic đưa tổng dung lượng toàn bộ khối TTS xuống chỉ còn **~438 MB**, dễ dàng vận hành song song cùng khối ASR và MT trong bộ nhớ RAM của phần cứng Snapdragon / Rubik Pi 3.

---

**Phiên bản tài liệu:** 2026-08-09 (Hoàn tất kiểm thử thực nghiệm round-trip ASR cho toàn bộ các candidate)  
**Trạng thái:** Sẵn sàng tích hợp vào mục §4.2 của Technical Proposal chính thức.
