# Step 2 — Dịch máy Văn bản (MT - Machine Translation, Text → Text)

**Trạng thái (2026-08-09):** Đã kiểm thử mã nguồn thực nghiệm đối đầu giữa các mô hình (NLLB-600M so với Qwen3-0.6B và Qwen3-1.7B trên bộ dữ liệu chuẩn FLORES-200, đo đạc đủ cả 6 chiều dịch) kết hợp nghiên cứu toàn văn 5/7 bài báo khoa học về chính sách streaming, xử lý ngôn ngữ ít tài nguyên và trộn mã (code-switching). Kiến trúc **CHỐT**: **NLLB-200-distilled-600M + Thuật toán điều phối luồng AlignAtt**.

---

## Part A — Bản hoàn chỉnh cho Technical Proposal §4.2 "Thiết kế Từng Module" (Drop-in Ready)

| Module | Mô hình / Framework | Kích thước (Ước tính / Thực đo) | Độ trễ Mục tiêu (Latency Target) | Kỹ thuật Then chốt |
|---|---|---|---|---|
| **Dịch máy Thần kinh Cốt lõi (NMT Core)** | **NLLB-200-distilled-600M** | ~600M tham số (~1.2 GB FP16, lượng tử hóa INT8 nén xuống **594 MB**) | RTF < 0.1 chế độ không streaming (đo thực tế: 0.4–0.9s/câu trên FLORES-200) | Tinh chỉnh (fine-tune) theo từng cặp ngôn ngữ; tự chuyển đổi đồ thị tĩnh ONNX $\rightarrow$ QNN DLC biên dịch cho NPU Hexagon. |
| **Chính sách Streaming (Bản MVP)** | **AlignAtt** (Papi et al., 2023) | **0 tham số bổ sung** | AL ≈ 2.0s (số liệu tham chiếu từ bài báo gốc) | Thiết lập ngưỡng dựa trên ma trận cross-attention, không cần huấn luyện lại, áp dụng trực tiếp trên mô hình NLLB có sẵn. |
| **Chính sách Streaming (Lộ trình Nâng cấp)** | Mạng chính sách Đọc/Ghi kiểu **AliBaStr-MT** | Thêm ~1–3M tham số | Giảm **32–37% độ trễ** so với không streaming (tham chiếu hệ thống on-device tương đương của Meta AI arXiv 2508.13358) | Mạng nơ-ron chính sách học có giám sát dựa trên nhãn giả (pseudo-labels) trích xuất từ tầng chú ý (attention). |
| **Tối ưu Cặp Ít Tài nguyên (Vi↔Zh, Vi↔Ko)** | Tinh chỉnh mBART-50 + Dịch ngược (Back-translation) lọc theo domain | — | — | Chọn lọc văn bản đơn ngữ liên quan bằng TF-IDF + Dịch ngược có kiểm soát (không áp dụng dịch ngược ngây thơ). |
| **Xử lý Hiện tượng Trộn mã (Code-switch)** | Tăng cường dữ liệu Giữ nguyên (Copy-Through Augmentation) | — | — | Bổ sung các cặp đồng nhất (identity pairs: En $\rightarrow$ En, X $\rightarrow$ X) vào tập dữ liệu fine-tune; tối ưu chi phí hơn hẳn pipeline dựa trên LLM lớn. |
| **Mô hình Giáo viên Sinh dữ liệu (Chỉ chạy offline trên máy dev)** | **Hunyuan-MT-7B** (Tencent - Top 1 WMT2025 trong 30/31 cặp ngôn ngữ) | 7B tham số, **tuyệt đối không đưa lên thiết bị biên** | — | Sinh dữ liệu song ngữ tổng hợp Vi↔Zh/Ko chất lượng vượt trội hơn hẳn kỹ thuật dịch ngược thô. |

**⚠️ Lưu ý quan trọng cho phần Phần cứng (§5 Technical Proposal):**
- Mô hình NLLB-600M **không** nằm trong danh mục mô hình tối ưu sẵn của Qualcomm AI Hub (khác với Qwen3-0.6B/1.7B đã được lượng tử hóa sẵn cho Snapdragon). Tuy nhiên, qua kiểm thử thực nghiệm đối đầu trực tiếp (xem chi tiết tại Part B §1), NLLB đã thắng áp đảo cả về chất lượng dịch thuật lẫn tốc độ xử lý.
- Kế hoạch triển khai: Nhóm tự thực hiện quy trình đóng gói ONNX $\rightarrow$ QNN DLC cho NPU Hexagon và ghi nhận đây là tác vụ kỹ thuật bổ sung trong mốc thời gian của dự án (§7).
- Toàn bộ điểm số BLEU và tốc độ đo đạc trong tài liệu này được ghi nhận trên GPU máy trạm dev (CUDA) nhằm phục vụ so sánh công bằng giữa các thuật toán.

**Mục tiêu chất lượng (Quality Target):**
- Điểm BLEU ở chế độ dịch luồng (streaming) duy trì trong khoảng **~2–3 điểm** so với đường cơ sở không streaming (khoảng cách tham chiếu từ hệ số đo đạc thực tế của hệ thống AliBaStr-MT: 43.89 so với 45.56 điểm BLEU trên cặp ngôn ngữ tương đương).

---

## Part B — Phân tích Kỹ thuật Đầy đủ & Dữ liệu Thực nghiệm Lựa chọn

### 1. Thử nghiệm Đối đầu: NLLB-600M vs. Qwen3-0.6B vs. Qwen3-1.7B

**Bối cảnh thực nghiệm:**
- Đề bài của ban tổ chức khuyến khích tận dụng các mô hình có sẵn trên Qualcomm AI Hub — NLLB không có sẵn tại đó, trong khi Qwen3-0.6B và Qwen3-1.7B đều có bản lượng tử hóa sẵn cho chipset Snapdragon.
- Nhóm đã tiến hành kiểm thử thực tế trên tập dữ liệu chuẩn **FLORES-200** (30 câu ngẫu nhiên $\times$ 6 chiều dịch thuật) để xác định xem lợi thế "có sẵn trên AI Hub" có đủ bù đắp cho chất lượng và tốc độ hay không.

**Bảng kết quả so sánh Điểm BLEU và Tốc độ suy luận (Đo đạc thực tế):**

| Chiều dịch | NLLB-600M | Qwen3-0.6B | Qwen3-1.7B |
|---|:---:|:---:|:---:|
| **vi $\rightarrow$ en** | **33.81** | 18.44 | 26.85 |
| **en $\rightarrow$ vi** | **29.67** | 16.19 | 26.01 |
| **vi $\rightarrow$ zh** | 20.45 | 17.74 | **23.41** |
| **zh $\rightarrow$ vi** | **21.07** | 10.39 | 17.58 |
| **vi $\rightarrow$ ko** | **8.05** | 3.18 | 5.75 |
| **ko $\rightarrow$ vi** | **18.60** | 5.63 | 13.97 |
| **Tốc độ (giây/câu)** | **0.4–0.9 s** | 2.7–6.9 s | 1.6–10.1 s |

**✅ CHỌN: NLLB-200-distilled-600M.**
- Chiến thắng áp đảo ở **5/6 chiều dịch thuật** và thắng tuyệt đối về tốc độ ở **tất cả các chiều** (nhanh hơn Qwen3 từ 3 đến 15 lần tùy từng cặp ngôn ngữ).
- Dù đòi hỏi nhóm phải tự tay chuyển đổi ONNX $\rightarrow$ QNN, số liệu thực nghiệm chứng minh đây là quyết định hoàn toàn đúng đắn: Lợi thế "tối ưu sẵn" của Qwen3 không thể bù đắp khoảng cách chất lượng quá lớn và tốc độ suy luận quá chậm của mô hình sinh từ tự hồi quy kiểu LLM.

**❌ LOẠI: Qwen3-0.6B.**
- Dù có cùng cỡ tham số (~600M) và có sẵn trên AI Hub, mô hình **thua rất xa ở mọi chiều dịch** (ví dụ chiều vi $\rightarrow$ en chỉ đạt 18.44 so với 33.81 của NLLB — kém gần gấp đôi) và tốc độ chậm hơn từ 3 đến 8 lần.

**❌ LOẠI: Qwen3-1.7B (Làm mô hình dịch chính).**
- Chỉ dẫn trước duy nhất ở 1/6 chiều dịch (vi $\rightarrow$ zh: 23.41 so với 20.45) nhưng thua toàn diện ở 5 chiều còn lại và có tốc độ chậm nhất trong 3 ứng viên (lên tới 10.1 giây cho một câu đơn).
- **Giữ lại làm ứng viên phụ dự phòng** cho riêng chiều vi $\rightarrow$ zh nếu cần chế độ dịch chính xác cao (Accurate Mode) không yêu cầu streaming.

---

## 2. Lựa chọn Chính sách Streaming: AlignAtt (MVP) & Phương án Nâng cấp AliBaStr-MT

Phân tích toàn văn hai bài báo khoa học then chốt dễ bị nhầm lẫn về tên gọi:

| Tiêu chí | **AlignAtt** (SimulSeamless, IWSLT 2024) | **AliBaStr-MT** (Meta AI, On-device MT, 2025) |
|---|---|---|
| **Cơ chế hoạt động** | **Không cần huấn luyện lại:** Tận dụng ma trận cross-attention của mô hình offline có sẵn, tạm dừng phát từ nếu mô hình đang "nhìn" vào $f$ khung âm thanh gần nhất. | **Cần huấn luyện thêm:** Huấn luyện một mạng chính sách nhỏ (policy network), học từ các nhãn giả trích xuất từ cơ chế attention của mô hình lớn. |
| **Mô hình gốc trong bài báo** | SeamlessM4T-medium (~1.2B tham số — quá nặng cho thiết bị biên). | Kiến trúc Encoder/Decoder tách biệt (~103M tham số — chuẩn on-device). |
| **Siêu tham số điều khiển** | Tham số $f$ (số khung gần nhất) — tinh chỉnh riêng cho từng cặp ngôn ngữ. | Ngưỡng $\delta$ — điều chỉnh linh hoạt trong lúc suy luận, không cần train lại. |
| **Số liệu thực nghiệm từ bài báo** | en $\rightarrow$ zh: BLEU 20.56, AL 1.94s (tác giả thừa nhận gặp khó khăn ở cặp ngôn ngữ đảo trật tự từ này). | Cặp ES: BLEU 43.89/45.30 so với non-streaming 45.56/46.60 (chỉ giảm ~1.5 điểm); AL giảm mạnh từ 4.39s xuống 2.74s (**nhanh hơn 37%**). |

**✅ CHỌN cho bản MVP: Thuật toán AlignAtt áp dụng trên NLLB-600M.**
- Lý do: Không mất công huấn luyện lại, triển khai nhanh chóng, không đòi hỏi dữ liệu huấn luyện chính sách phức tạp — hoàn toàn khớp với mục tiêu phát triển giai đoạn đầu.
- **Lộ trình nâng cấp:** Mạng chính sách AliBaStr-MT rất tiềm năng cho thiết bị biên và sẽ được cân nhắc triển khai ở pha tiếp theo nếu còn thời gian.

---

## 3. Cải thiện Cặp Ngôn ngữ Ít Tài nguyên (Vi-Zh): Hiệu chỉnh Nhận định về Dịch ngược (Back-Translation)

Phân tích toàn văn bài báo arXiv 2501.19314 (Samsung SDS, VLSP 2022) cho thấy số liệu thực tế:

| Chiều dịch | Google Translate | UET Engine | mBART-50 + Dịch ngược (Bài báo) |
|---|:---:|:---:|:---:|
| **vi $\rightarrow$ zh** | **41.6** | 39.2 | 38.97 |
| **zh $\rightarrow$ vi** | 38.1 | **40.6** | 38.90 |

Tác giả bài báo tự đúc kết: *"Google Translate đạt chất lượng tốt nhất cho chiều dịch Việt $\rightarrow$ Trung, trong khi UET Engine dẫn đầu ở chiều Trung $\rightarrow$ Việt"*. Do đó, dịch ngược không giúp vượt trội Google Translate ở cả hai chiều mà chỉ nhỉnh hơn 0.8 điểm ở một chiều duy nhất.

**Giá trị thực tế của kỹ thuật Dịch ngược (Back-Translation):**
- Giúp cải thiện đáng kể so với mô hình cơ sở ban đầu (Zh $\rightarrow$ Vi tăng +3.19 BLEU từ 35.58 lên 38.90).
- Đây là giải pháp hiệu quả để nâng cấp một hệ thống yếu, **không phải là công thức thần kỳ để đánh bại hoàn toàn các dịch vụ đám mây thương mại**.

**✅ Định hướng áp dụng cho AuraTranslate-Edge:**
- Áp dụng kỹ thuật dịch ngược có chọn lọc theo domain (dùng TF-IDF chọn lọc các câu đơn ngữ liên quan đến môi trường nhà máy, an toàn lao động).
- Lợi thế cạnh tranh cốt lõi (Moat) của dự án là:
  1. **Hoạt động hoàn toàn Offline** (bảo đảm quyền riêng tư, không phụ thuộc kết nối internet).
  2. **Tối ưu hóa chuyên sâu cho thuật ngữ an toàn công nghiệp và sản xuất** (vốn là điểm yếu của các công cụ dịch tổng quát).
  3. Tiêu chí đánh giá thành công (DoD) được đặt là **tỷ lệ dịch chính xác thuật ngữ chuyên ngành đạt > 95%**.

---

## 4. Hiện tượng Trộn mã (Code-Switching): Áp dụng Kỹ thuật "Copy-Through"

Nghiên cứu VietMix (arXiv 2505.24472, Đại học Maryland) đo đạc trực tiếp: Kỹ thuật dịch ngược truyền thống trên văn bản trộn mã Việt-Anh chỉ giúp cải thiện **+0.55 điểm xCOMET so với zero-shot** (77.72 lên 78.27) — gần như không mang lại hiệu quả do mô hình có xu hướng tự động "chuẩn hóa" câu về dạng đơn ngữ sạch, làm mất hiện tượng trộn mã tự nhiên.

**❌ LOẠI: Pipeline VietMix hoàn chỉnh.**
- Đòi hỏi sử dụng LLM từ 7B–9B tham số kết hợp nhiều tầng lọc — quá nặng nề và bất khả thi cho thiết bị biên.

**✅ CHỌN: Kỹ thuật "Copy-Through" từ bài báo AliBaStr-MT.**
- Bổ sung các cặp dữ liệu giữ nguyên danh xưng/thuật ngữ (En $\rightarrow$ En, Term $\rightarrow$ Term) vào tập dữ liệu fine-tune.
- Dạy mô hình nhận biết và giữ nguyên các từ ngữ chuyên ngành tiếng Anh đã chuẩn xác mà không cố gắng dịch thô sang tiếng Việt.
- Sử dụng tập kiểm thử chuẩn 1.002 câu của VietMix (Vi $\rightarrow$ En) làm bộ benchmark nội bộ để đánh giá năng lực xử lý trộn mã.

---

## 5. Sử dụng Hunyuan-MT-7B làm Mô hình Giáo viên (Offline Teacher)

- Báo cáo arXiv 2509.05209 (Tencent): Hunyuan-MT-7B đạt Top 1 giải WMT2025 ở 30/31 cặp ngôn ngữ, có hỗ trợ tiếng Việt chính thức.
- Khoảng cách chất lượng so với các công cụ dịch thương mại là rất lớn (nhóm ZH $\Rightarrow$ XX đạt 0.876 so với 0.762 điểm XCOMET-XXL của Google).
- **Quy trình triển khai:** Mô hình 7B được vận hành hoàn toàn trên máy trạm offline để sinh dữ liệu huấn luyện song ngữ nhân tạo chất lượng cao cho các cặp Vi↔Zh và Vi↔Ko, **hoàn toàn không đưa lên thiết bị biên**, giúp bảo toàn 100% ngân sách tài nguyên của phần cứng mục tiêu.

---

## 6. Phân tích Hiện tượng Điểm BLEU Thấp Bất Thường ở Chiều vi $\rightarrow$ ko

Trong bảng đo đạc thực nghiệm, chiều vi $\rightarrow$ ko đạt điểm BLEU tương đối thấp (8.05 so với 18.60 của chiều ngược lại). Phân tích chi tiết từng mẫu dịch cho thấy:
- **Ngữ nghĩa của câu dịch cơ bản chuẩn xác** (chỉ có 1 lỗi từ vựng đơn lẻ khi từ "tuần lộc" bị dịch nhầm thành "오징어" - con mực).
- **Nguyên nhân chính:** Tiếng Hàn là **ngôn ngữ chắp dính (agglutinative language)** với hệ thống kính ngữ và trợ từ phức tạp (ví dụ quá khứ "않았습니다" so với hiện tại "않습니다" đều mang ngữ nghĩa chuẩn xác). Thang đo BLEU dựa trên n-gram bề mặt nên bị phạt điểm rất nặng dù bản dịch hoàn toàn tự nhiên.
- Chiều ngược lại (ko $\rightarrow$ vi) dịch sang tiếng Việt không phải ngôn ngữ chắp dính nên điểm BLEU cao hơn rõ rệt (18.60).
- $\rightarrow$ Cần bổ sung các chỉ số đo đạc ngữ nghĩa như **COMET/xCOMET** để đánh giá công bằng chất lượng của NLLB trên cặp ngôn ngữ này.

---

## 7. Các Lỗi Môi trường Kỹ thuật Đã Khắc phục

- **Nguồn dữ liệu FLORES-200:** Thư viện `datasets` không còn hỗ trợ các loading script cũ từ HuggingFace; nhóm đã trực tiếp tải và xử lý tệp nén tarball chính thức từ Meta (`dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz`).
- **Khắc phục lỗi treo tiến trình khi đánh giá Qwen3:** Do sử dụng tham số `torch_dtype` đã bị loại bỏ trên nhánh transformers mới (`5.15.0.dev0`); đã chuyển sang cú pháp chuẩn `dtype` và bổ sung thanh theo dõi tiến độ chi tiết theo từng câu.

---

## 8. Kiến trúc Tổng hợp & Lộ trình Hoàn thiện

1. **Chế độ Vận hành Trực tiếp (Live Mode):** NLLB-200-distilled-600M kết hợp cơ chế dịch luồng AlignAtt.
2. **Cặp Vi $\leftrightarrow$ En:** Tinh chỉnh trên bộ ngữ liệu chất lượng cao PhoMT.
3. **Cặp Vi $\leftrightarrow$ Zh và Vi $\leftrightarrow$ Ko:** Tinh chỉnh kết hợp dữ liệu dịch ngược có lọc TF-IDF và dữ liệu tổng hợp sinh từ mô hình giáo viên Hunyuan-MT-7B.
4. **Xử lý Trộn mã (Code-Switching):** Tích hợp kỹ thuật Copy-Through và đánh giá bằng tập test VietMix.
5. **Đóng gói NPU:** Lượng tử hóa INT8 (CTranslate2 / QNN W8A16) đưa dung lượng mô hình từ 2.3 GB xuống **594 MB**, bảo đảm vận hành mượt mà trong bộ nhớ RAM của phần cứng Snapdragon.

---

**Phiên bản tài liệu:** 2026-08-09 (Cập nhật dữ liệu đo đạc FLORES-200 và phân tích toàn văn các bài báo khoa học)  
**Trạng thái:** Hoàn tất kiểm thử thực nghiệm, kiến trúc dịch máy đã được chốt chính thức cho Technical Proposal.
