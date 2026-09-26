# Step 4 — Lượng tử hoá và Triển khai mô hình (Quantization & Deployment)

*Tài liệu tổng hợp cho buổi họp team. Bao quát: tiến độ tổng thể, cơ chế quantization/deployment, bug lớn nhất của dự án (đã tìm ra nguyên nhân + fix), và kế hoạch tiếp theo.*

---

## 1. Bức tranh tổng thể — OneVoice là gì, đang ở đâu

OneVoice là thiết bị dịch song ngữ edge AI offline cho môi trường nhà máy (Vi↔Anh/Trung/Hàn), dự thi **OneVoice AI Challenge** (Saigon AI Hub × Qualcomm). Kiến trúc pipeline:

```
Audio vào → VAD (Silero) → Denoise/Beamform (GTCRN + MVDR) → ASR → MT → TTS → Audio ra
```

| Bước | Module | Model đã chọn | Trạng thái |
|---|---|---|---|
| 0 | Audio front-end | Silero VAD + GTCRN + MVDR/GSC | ✅ Chọn xong, đo WER thật |
| 1 | ASR | Zipformer-30M (Vi) + SenseVoice-Small (En/Zh/Ko) | ✅ Chọn xong, đo WER thật |
| 2 | MT | NLLB-200-distilled-600M | ✅ Chọn xong, đo BLEU thật |
| 3 | TTS | Piper (Vi) + Supertonic (Ko/En) + MeloTTS-ZH (Zh) | ✅ Chọn xong, đo MOS thật |
| 4 | Hardware & Quantization | Qualcomm Snapdragon (QCS6490 → đang chuyển sang IQ-9075/8 Elite Gen5) | 🔶 Đang dở dang — đây là trọng tâm buổi họp |
| 5 | End-to-end pipeline | Ghép 1-4, CPU/GPU baseline | ✅ Chạy được, có số RTF thật |

Toàn bộ lựa chọn model đều **code-test bằng số đo thật** (không lấy từ tuyên bố của nhà sản xuất) — WER/CER/RTF/BLEU/MOS đo trực tiếp trên dữ liệu thật, so sánh nhiều ứng viên, ghi lại từng lý do chọn/loại trong `step0.md`–`step4.md`.

**Việc còn thiếu duy nhất trước khi có thể chốt Technical Proposal:** chứng minh toàn bộ pipeline chạy **đúng** (không chỉ chạy được) trên phần cứng Snapdragon thật, ở độ chính xác quantize sẽ dùng để deploy thật. Đây là phần vừa trải qua một hành trình điều tra dài và phức tạp — nội dung chính của tài liệu này.

---

## 2. Vì sao phải quantize — cơ chế và lý do

### 2.1 Bài toán

Model gốc (fp32 — số thực dấu phẩy động 32-bit) cho độ chính xác cao nhất nhưng:
- **To**: NLLB-600M ở fp32 nặng ~2.4GB, không thể chạy trên thiết bị RAM giới hạn (Rubik Pi 3 chỉ có 8GB, phải chia sẻ cho cả hệ điều hành + toàn bộ pipeline 5 model)
- **Chậm & tốn điện**: CPU/GPU chạy fp32 tốn nhiều chu kỳ tính toán và điện năng hơn hẳn — không phù hợp thiết bị pin, mục tiêu vận hành 8+ giờ liên tục

**NPU (Neural Processing Unit)** trên chip Snapdragon — cụ thể là khối **Hexagon Tensor Processor (HTP)** — được thiết kế phần cứng chuyên biệt để tính toán cực nhanh và cực tiết kiệm điện, nhưng **đánh đổi bằng việc chỉ làm việc với số nguyên (fixed-point), không phải số thực dấu phẩy động**. Đây không phải giới hạn phần mềm có thể vá — là kiến trúc silicon.

### 2.2 Quantization hoạt động thế nào

Quantize = ánh xạ một khoảng giá trị số thực liên tục (ví dụ trọng số mạng nơ-ron nằm trong [-2.5, 3.1]) sang một tập hữu hạn số nguyên rời rạc, bằng công thức:

```
giá_trị_lượng_tử = round((giá_trị_thực - zero_point) / scale)
```

- **int8**: chỉ có 256 mức giá trị phân biệt (-128 đến 127)
- **int16**: có 65,536 mức giá trị phân biệt (gấp 256 lần độ phân giải so với int8)

Mỗi lần lượng tử hoá là một lần **làm tròn** — mất thông tin không thể phục hồi. Với 1 phép tính đơn lẻ, sai số này nhỏ và không đáng kể. Nhưng mạng nơ-ron sâu (nhiều lớp xếp chồng) thực hiện **hàng nghìn** phép tính liên tiếp, và **sai số làm tròn CÓ THỂ dồn tích (compound) qua từng lớp** — đây chính là cơ chế gốc rễ của bug lớn nhất dự án gặp phải, giải thích chi tiết ở mục 4.

Có 2 thứ cần lượng tử hoá riêng biệt trong một model:
- **Weights (trọng số)**: các con số cố định học được sau khi train, không đổi khi chạy
- **Activations (kích hoạt)**: kết quả trung gian tính ra tại mỗi lớp trong lúc chạy — luôn thay đổi theo input

Đây là điểm mấu chốt: **có thể lượng tử hoá weights và activations ở độ chính xác (bit-width) khác nhau** — gọi là *mixed precision*. Ký hiệu quen thuộc: `w8a16` nghĩa là weights ở int8 (8-bit), activations ở int16 (16-bit).

---

## 3. Quy trình deploy lên Qualcomm AI Hub — cơ chế

### 3.1 Chuỗi công cụ

Model đi qua 3-4 bước biến đổi trước khi chạy được trên silicon thật:

```
ONNX (fp32, từ PyTorch/HuggingFace)
   │
   ├─► [1] submit_quantize_job  → gán scale/zero-point cho từng tensor,
   │       dùng "calibration data" (vài mẫu input thật) để ước lượng khoảng
   │       giá trị thực tế mỗi tensor sẽ gặp
   │
   ├─► [2] submit_compile_job   → biên dịch thành "QNN context binary"
   │       (nhị phân thực thi trực tiếp trên HTP, giống .exe cho NPU)
   │
   ├─► [3] submit_profile_job   → đo latency/power thật trên board
   │
   └─► [4] submit_inference_job → chạy input thật, lấy output thật để verify
```

Toàn bộ chạy qua **Qualcomm AI Hub** — dịch vụ cloud của Qualcomm cho phép submit job, họ tự động cấp phát board thật (QCS6490, IQ-9075, Snapdragon 8 Elite...) từ xa, chạy, trả kết quả về. Không cần sở hữu phần cứng vật lý để test.

### 3.2 Vì sao "compile SUCCESS" không có nghĩa là "chạy đúng"

Đây là **phát hiện quan trọng nhất về mặt phương pháp luận** của cả quá trình làm Step 4: `submit_compile_job` chỉ kiểm tra model có **biên dịch được** thành công (đúng cú pháp graph, đúng op được hỗ trợ) — nó **không hề kiểm tra kết quả tính toán ra có đúng hay không**. Suốt nhiều tuần trước đó, mọi số liệu latency đã đo được đều dựa trên các model "compile SUCCESS" — nhưng **chưa từng có ai so sánh output thật với kết quả fp32 tham chiếu**.

Khi làm việc này (so sánh bằng **cosine similarity** giữa output hardware và output fp32 local), phát hiện ra: hầu hết model cho ra **kết quả sai nghiêm trọng** dù compile "thành công" và có số latency "đẹp". Đây là lý do cả quá trình điều tra sau đó tồn tại.

---

## 4. Vấn đề gặp phải — thuật toán hiện tại (int8-only) và vì sao FAIL

### 4.1 Triệu chứng

Với **mọi model có output từ 2 chiều trở lên** (NLLB encoder, Zipformer encoder, Piper Stage2, cả 4 submodel Supertonic), khi chạy int8 trên phần cứng thật:

- **Cosine similarity chỉ ~0.18–0.29** so với fp32 reference (0 = không liên quan gì, 1 = giống hệt) — gần như phá huỷ hoàn toàn thông tin ngữ nghĩa
- **Duy nhất 2 model có output 1 chiều** (decoder, joiner của Zipformer — các lớp dense đơn giản) là ĐÚNG (cos_sim ~1.0)

### 4.2 Quá trình loại trừ — vì sao khó tìm ra

Đây là phần tốn nhiều thời gian nhất: có **ít nhất 6 giả thuyết hợp lý** đã bị loại bỏ bằng thực nghiệm thật trước khi tìm ra nguyên nhân đúng:

| # | Giả thuyết | Vì sao tưởng đúng | Kết quả thực nghiệm |
|---|---|---|---|
| 1 | Do precision (int8 vs fp16 vs int16) | Precision cao hơn phải chính xác hơn | Cả 3 đều cho cos_sim ~0.21 — **giống hệt nhau bất thường** |
| 2 | Do phần cứng thiết bị cụ thể lỗi | QCS6490 là board rẻ, có thể có bug | Test cả QCS6490, Snapdragon 8 Elite Gen5, IQ-9075 — cả 3 đều hỏng như nhau |
| 3 | Do calibration data quá ít (4-8 mẫu) | Ít mẫu → scale/zero-point ước lượng sai | Tăng lên 600 mẫu thật — KHÔNG giúp, còn TỆ HƠN |
| 4 | Do thuật toán calibration cơ bản | Min-max đơn giản có thể không tối ưu | Đổi sang AIMET AdaRound (thuật toán tối ưu hoá nâng cao) — KHÔNG giúp |
| 5 | Do đọc sai thứ tự trục tensor (layout bug) | Giải thích được nghịch lý #1 (fp16≈int8) | Brute-force test mọi cách diễn giải buffer — bác bỏ dứt khoát, không phải lỗi đọc dữ liệu |
| 6 | Do `--quantize_io` âm thầm ép I/O về int dù khai "fp16" | Cũng giải thích được nghịch lý #1 | Compile không có `--quantize_io` → Qualcomm AI Hub từ chối thẳng: **HTP về mặt kiến trúc không hỗ trợ I/O dạng float, không có cách nào bỏ qua** |

**Bằng chứng phụ quan trọng:** một issue công khai trên GitHub của chính Qualcomm (`qualcomm/aimet#3978`, chưa có câu trả lời chính thức) báo cáo đúng pattern: sai số giữa mô phỏng quantize và hardware thật **tăng theo cấp số nhân với độ sâu mạng** (MSE 5.64×10⁻⁵ ở 1 lớp → 1.04×10⁻¹ ở 29 lớp). Đây là dấu hiệu cho thấy đây không phải bug riêng của dự án — là giới hạn đã biết (nhưng chưa official fix) của toolchain AIMET/QNN khi lượng tử hoá mạng sâu.

### 4.3 Nguyên nhân gốc — tại sao thuật toán int8-only FAIL

Kết luận cuối cùng, xác nhận bằng 2 thực nghiệm 0-chi-phí (mô phỏng local, không cần chạy hardware):

**Thực nghiệm A — chỉ lượng tử hoá output (boundary):** cos_sim vẫn đạt 0.9999 → chứng minh **KHÔNG phải lỗi ở biên I/O**.

**Thực nghiệm B — mô phỏng int8 đầy đủ (cả weights lẫn activations nội bộ) bằng công cụ chuẩn onnxruntime:** cos_sim rơi xuống 0.32 — **khớp chính xác với vùng "hỏng" đo được trên hardware thật (0.18-0.29)**.

→ **Root cause: int8 ACTIVATIONS bị lượng tử hoá quá thô (chỉ 256 mức giá trị) dồn tích sai số qua hàng chục lớp transformer sâu.** Mỗi lớp làm tròn một chút, lớp sau nhận input đã lệch từ lớp trước, lệch tiếp tục cộng dồn — đến lớp cuối, tín hiệu gốc gần như bị nhấn chìm trong nhiễu lượng tử hoá.

**Yếu tố khuếch đại thêm:** một số model dùng hằng số "vô cực âm" cực đoan để triệt tiêu attention (ví dụ NLLB dùng -3.4×10³⁸ cho phần bị che/pad trong self-attention mask). Khi lượng tử hoá, thang đo (scale) của cả tensor phải trải rộng để chứa được giá trị cực đoan này — kéo theo **độ phân giải dành cho các giá trị "thật" (nằm trong khoảng nhỏ, có ý nghĩa) bị nén ép cực kỳ thô**. Đây là hiệu ứng "outlier nuốt hết độ phân giải" — giống như đo nhiệt độ phòng bằng thước đo từ -273°C đến +1000°C: dải đo hợp lý (18-30°C) chỉ còn vài "vạch chia" để phân biệt.

---

## 5. Thuật toán có thể thành công — cơ chế và bằng chứng thật

### 5.1 Ý tưởng: mixed precision (w8a16)

Thay vì lượng tử hoá đồng loạt mọi thứ về int8, **giữ weights ở int8 (nhỏ gọn, tiết kiệm bộ nhớ — vì weights cố định, không đổi theo input) nhưng nâng activations lên int16 (65,536 mức thay vì 256 mức — vì activations là nơi sai số dồn tích qua từng lớp)**.

Về mặt tín hiệu học: Signal-to-Quantization-Noise Ratio (SQNR) tăng gần tuyến tính theo số bit — mỗi bit thêm vào tăng SQNR khoảng 6dB. Từ 8-bit lên 16-bit là thêm 8 bit → tăng ~48dB tỷ lệ tín hiệu/nhiễu — đủ để sai số dồn tích qua nhiều lớp không còn áp đảo tín hiệu thật nữa.

**Đây không phải ý tưởng lý thuyết suông** — chính Qualcomm publicly dùng recipe `w8a16` cho model `whisper_small_quantized` chính thức của họ trên AI Hub. Nghĩa là chính đội ngũ Qualcomm cũng từng gặp và giải quyết đúng vấn đề này cho kiến trúc transformer.

**Phát hiện API quan trọng — giải thích nghịch lý #1 ở mục 4.2:** cờ `--quantize_full_type float16`/`int16` khi dùng qua `submit_compile_job` **chỉ đặt precision cho WEIGHT**, hoàn toàn KHÔNG đụng tới precision của ACTIVATION (mặc định luôn là int8 trừ khi dùng đúng API `submit_quantize_job` với tham số `activations_dtype` tường minh). Đây chính là lý do 3 job "int8"/"fp16"/"int16" trước đó đều hội tụ về cùng cos_sim ~0.21 — cả 3 đều **âm thầm giữ activations ở int8** dù tên gọi khác nhau.

### 5.2 Kèm theo: giảm outlier cực đoan

Song song với w8a16, giảm giá trị "vô cực âm" (ví dụ -30000 thay vì -3.4×10³⁸, hoặc -30 thay vì -30000 khi cần chính xác hơn) — miễn giá trị đủ âm để hàm softmax vẫn triệt tiêu hoàn toàn phần bị che (e⁻³⁰ ≈ 10⁻¹³ ≈ 0, đủ nhỏ để không ảnh hưởng kết quả), nhưng không "nuốt" hết độ phân giải lượng tử hoá của các giá trị thật xung quanh.

### 5.3 Bằng chứng thật trên hardware — đã verify, không phải lý thuyết

| Model | Kiến trúc | int8-only (baseline) | w8a16 (fix) | Ghi chú |
|---|---|---|---|---|
| **NLLB encoder** | Transformer seq2seq MT, 24 lớp | cos_sim = 0.18 | **cos_sim = 0.9998** ✅ | Sửa dứt điểm — kết hợp w8a16 + giảm mask outlier -30000→-30 |
| **Zipformer encoder** | Streaming conformer ASR | cos_sim = 0.21 | cos_sim = 0.80 | Cải thiện lớn (~4×) nhưng CHƯA đạt 0.95 — per-frame cho thấy còn 1 outlier cấu trúc riêng chưa tìm ra (nghi liên quan layer downsample/upsample) |
| Piper Stage2, Supertonic ×4 | Flow-based / diffusion TTS | chưa verify trước đây | đang verify (dở dang) | Xem mục 6 |

**Kết luận có thể trình bày tại họp:** đây KHÔNG còn là "chưa biết nguyên nhân" — nguyên nhân đã xác định rõ ràng, có cơ chế toán học giải thích được, có bằng chứng thật trên 2 kiến trúc khác nhau xác nhận hướng đi đúng. Việc còn lại là *kỹ thuật thực thi* (áp dụng lại cho các model còn lại + xử lý một vài outlier đặc thù từng model), không phải *nghiên cứu chưa có lời giải*.

---

## 6. Giới hạn phần cứng quan trọng — ảnh hưởng đến lựa chọn thiết bị

**w8a16 chỉ chạy được trên Hexagon phiên bản v73 trở lên** (NPU generation mới). Đã xác nhận bằng thực nghiệm thật (không phải suy đoán):

- **QCS6490 / Rubik Pi 3 (Hexagon v68)**: compile FAILED thẳng — thông báo lỗi từ chính Qualcomm AI Hub xác nhận: *"floating-point type not supported"* / *"exit code 14"* — đây là giới hạn kiến trúc silicon, không có cách vượt qua bằng phần mềm.
- **Dragonwing IQ-9075 EVK (Hexagon v73)**: compile + chạy THÀNH CÔNG, cho kết quả đúng như bảng trên.
- **Snapdragon 8 Elite Gen 5 (Hexagon v81)**: cũng thuộc thế hệ v73+, về lý thuyết chạy được nhưng chưa test riêng với w8a16.

**Hệ quả chiến lược quan trọng cho buổi họp:** kế hoạch ban đầu (Rubik Pi 3/QCS6490 làm thiết bị demo chính) **không còn khả thi về mặt kỹ thuật** nếu muốn deploy pipeline đúng — cần chuyển sang IQ-9075 EVK hoặc điện thoại Snapdragon 8 Elite Gen 5 làm thiết bị đích. Đây là quyết định cần chốt tại buổi họp (xem mục 7).

---

## 7. Trạng thái hiện tại (chi tiết) — 5 model đang verify dở dang

Sau khi xác nhận w8a16 hoạt động trên NLLB, đang chạy song song việc áp dụng cùng recipe cho 5 model còn lại (Piper Stage2 + 4 submodel Supertonic: duration_predictor, text_encoder, vector_estimator, vocoder). Trạng thái tại thời điểm dừng phiên làm việc:

| Model | Quantize (w8a16) | Compile (IQ-9075 EVK) | Inference/verify |
|---|---|---|---|
| Piper Stage2 | ✅ Xong | 🔶 Đang chạy | Chưa test |
| duration_predictor | ✅ Xong | ❌ FAILED (lỗi generic, cần xem log chi tiết trên UI) | — |
| text_encoder | ✅ Xong | ❌ FAILED (lỗi generic, cần xem log chi tiết trên UI) | — |
| vector_estimator | ✅ Xong | 🔶 Đang chạy | Chưa test |
| vocoder | ✅ Xong | ✅ **Compile SUCCESS** | Chưa chạy inference — việc tiếp theo cần làm ngay |

Toàn bộ chi tiết kỹ thuật (job ID cụ thể, shape đã pin, bài học môi trường/tooling) đã ghi đầy đủ trong `step4.md` §4n–§4r, để bất kỳ ai (kể cả agent AI khác) tiếp tục đúng từ điểm dừng mà không phải làm lại từ đầu.

---

## 8. Kế hoạch tiếp theo

**Ngắn hạn (kỹ thuật, có thể giao cho agent/dev tiếp tục ngay):**
1. Hoàn tất verify 5 model còn lại (vocoder cần chạy inference ngay vì đã compile xong; 2 model lỗi compile cần điều tra log chi tiết trên Qualcomm AI Hub Workbench UI vì thông báo lỗi qua API không đủ chi tiết)
2. Với Zipformer (cos_sim 0.80, chưa đạt 0.95): tìm outlier cấu trúc riêng — dữ liệu per-frame cho thấy lỗi tập trung ở vài vị trí cụ thể (nghi vấn: layer downsample/upsample), cần soi sâu hơn hoặc dùng AIMET QuantAnalyzer (công cụ debug chính thức của Qualcomm cho đúng loại vấn đề này, chạy local không cần cloud)
3. Sau khi có cos_sim > 0.95 cho toàn bộ 6 model (NLLB + Zipformer + Piper Stage2 + 4 Supertonic), coi như pipeline correctness đã xong

**Quyết định cần chốt tại buổi họp:**
- **Thiết bị demo cuối cùng:** IQ-9075 EVK (dev board, giữ được câu chuyện BOM/power-budget/form-factor tự thiết kế cho mục 25% điểm "Hardware & Device Concept" của Technical Proposal) hay Snapdragon 8 Elite Gen 5 (điện thoại — nhanh có sẵn nhưng yếu hơn về câu chuyện "thiết kế phần cứng riêng")
- Có cần liên hệ Qualcomm mentor xin quyền truy cập/mượn board IQ-9075 EVK vật lý không (khác QCS6490/Rubik Pi 3 vốn dễ mua ngoài thị trường ~$150)

**Trung hạn (sau khi correctness xong):**
- Đo lại RTF/latency thật trên thiết bị mới (IQ-9075/8 Elite) — số liệu QCS6490 cũ sẽ không còn dùng được cho bài nộp cuối
- Cập nhật Technical Proposal §5 (Hardware & Device Concept) theo thiết bị mới đã chốt
- Re-validate toàn bộ pipeline end-to-end (không chỉ từng model riêng lẻ) trên thiết bị mới

---

## 9. Tóm tắt 3 câu cho slide

1. **Vấn đề:** mọi model quantize int8 "compile thành công" nhưng cho kết quả SAI trên phần cứng thật (cos_sim ~0.2 so với fp32) — chỉ phát hiện được sau khi chủ động so sánh output, chứ không phải do compile báo lỗi.
2. **Nguyên nhân:** int8 activations (không phải weights, không phải I/O, không phải bug graph) mất quá nhiều độ phân giải, sai số dồn tích qua các lớp mạng sâu — đúng pattern đã được cộng đồng AIMET/Qualcomm ghi nhận (chưa có fix chính thức).
3. **Giải pháp đã xác nhận thật trên silicon:** nâng activations lên int16 (giữ weights int8) + giảm outlier cực đoan trong mask — NLLB đạt cos_sim 0.9998, Zipformer đạt 0.80 (đang tinh chỉnh thêm) — nhưng bắt buộc đổi thiết bị đích từ QCS6490 sang chip Hexagon v73+ (IQ-9075/8 Elite Gen5).
4. **Quyết định phân công nhiệm vụ Quantize và Deploy (step4):**

   > **Nền tảng thực thi:** bộ công cụ Dragonwing IQ-9075 EVK + chip Qualcomm Hexagon NPU v73

   | Thành viên | Nhiệm vụ phân công (Mô hình & Step) | Trạng thái |
   |---|---|:---:|
   | **Trần Quốc Khanh** | • Silero VAD + GTCRN + MVDR/GSC (Step 0)<br>• Zipformer-30M (Vi) (Step 1)<br>• NLLB-200-distilled-600M (Step 2) | *Đã xong*<br>—<br>*Đã xong* |
   | **Lê Gia Khánh** | • SenseVoice-Small (En/Zh/Ko) (Step 1) | — |
   | **Trần Trung Hiếu** | • MeloTTS-ZH (Zh) (Step 3) | — |
   | **Phạm Chấn Khoa** | • Supertonic (Ko/En) (Step 3) | — |
   | **Cái Hoàng Bảo Kin** | • Piper (Vi) (Step 3) | — |
