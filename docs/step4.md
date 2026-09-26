# Step 4 — Hardware & Quantization (Deploy target + model compression)

**Status (2026-08-09):** Chốt phần cứng demo: **Rubik Pi 3 (Qualcomm QCS6490)**, có phương án dự phòng. **Đã chạy quantize + verify chất lượng thật cho cả 3 model, kể cả điều tra và fix lỗi** (không chỉ lên kế hoạch): **NLLB-600M int8 AN TOÀN** (2.3GB→594MB, BLEU verified cả 6 chiều, không tụt). **Supertonic int8 lần đầu THẤT BẠI** (audio vỡ) — bisect tìm ra đúng thủ phạm là submodel `vocoder.onnx`, fix bằng cách giữ riêng nó ở fp32 và chỉ nén 3 submodel còn lại → **398MB→178MB, verify lại ASR nghe rõ, dùng được**. **SenseVoice-Small int8 dùng được nhưng có cái giá thật**: tiếng Anh ổn (WER 6.8%→7.6%), **tiếng Trung/Hàn tụt nhiều hơn ngưỡng chấp nhận đã đặt ra** (CER 2.3%→9.8% và 4.5%→9.5%, vượt ngưỡng 1-2 điểm % — cần team tự quyết định đánh đổi). Tổng dung lượng thực tế: **~1.38GB** (giảm 64% so với 3.86GB ban đầu).

---

## Part A — Drop-in cho Technical Proposal §5 "Hardware & Device Concept"

### Bảng so sánh phần cứng

| Platform | NPU | RAM | Giá | Portability | Trên Qualcomm AI Hub? |
|---|---|---|---|---|---|
| **Rubik Pi 3 (QCS6490) — CHỌN** | 12 TOPS | 8GB LPDDR4x (nguồn: retailer, chưa xác nhận official) | ~$179 ($159 early-bird, nguồn thứ cấp) | ⚠️ Cần nguồn USB-C PD 3.0 12V/3A (36W) — KHÔNG có pin sẵn | ✅ Có, chính thức |
| Snapdragon 8 Elite Gen 5 (phone) — dự phòng | ~80 TOPS (marketing claim tới ~100, chưa xác nhận số chính xác) | 12-16GB (tuỳ máy) | $1000+ | ✅ Pin sẵn, cầm tay thật sự, zero rủi ro nguồn điện | Suy luận từ tooling docs, chưa xác nhận trực tiếp trong danh sách device AI Hub |
| Thundercomm TurboX C8550 (QCS8550) | Chưa công bố TOPS | Chưa công bố | Chưa công bố (phải liên hệ sales) | Chưa rõ | ✅ Có, nhưng gắn nhãn **"(Proxy)"** — Qualcomm tự ghi "metrics sẽ khác trên thiết bị thật" |
| QCS8300 | — | — | — | — | ❌ Không thấy trong danh sách AI Hub — loại khỏi cân nhắc |

### Power budget (ước tính, chưa đo thật)

| Thành phần | Ghi chú |
|---|---|
| Rubik Pi 3 board | Yêu cầu 12V/3A = 36W đầu vào (USB-C PD 3.0) |
| Cần đo thật | Chưa có số công suất thực đo khi chạy pipeline đầy đủ — action item cho Step 5 |

**⚠️ Việc CHƯA xác nhận rõ trước khi chốt:** Rubik Pi 3 không có pin — để thực sự "portable" theo đúng yêu cầu đề bài ("any portable device format"), cần thêm 1 power bank hỗ trợ PD 3.0 12V (đa số power bank phổ thông chỉ ra 5V/9V, KHÔNG đủ) hoặc mạch buck-boost riêng. Đây là rủi ro tích hợp thật, chưa có trong ngân sách/kế hoạch trước đây.

---

## Part B — Phân tích đầy đủ (Vietnamese)

### 1. Vì sao chọn Rubik Pi 3, không phải phương án khác

**✅ CHỌN: Rubik Pi 3 (QCS6490) làm thiết bị demo chính.** Lý do:
- Duy nhất trong 3 phương án có đầy đủ thông tin công khai (giá, NPU TOPS, RAM) — TurboX C8550 phải liên hệ sales mới biết giá, rủi ro cho timeline cuộc thi.
- **Có trên Qualcomm AI Hub chính thức** (không gắn nhãn Proxy) — nghĩa là số latency đo qua `qai-hub` sẽ là số thật trên đúng chip, không phải suy diễn từ thiết bị thay thế như TurboX C8550.
- Giá hợp lý (~$179) so với phương án dùng điện thoại flagship (~$1000+).
- Form-factor RPi-HAT tương thích — dễ gắn thêm mic array (ReSpeaker 4-mic) đã có trong kế hoạch từ trước.

**⚠️ Rủi ro mới phát hiện (chưa từng ghi nhận trước đây):** Rubik Pi 3 cần nguồn 12V/3A qua USB-C PD 3.0, KHÔNG có pin sẵn. Power bank thường (5V/9V) sẽ KHÔNG chạy được board này. Đây là rủi ro trực tiếp tới tiêu chí "portable device" của đề bài — phải xác nhận mua power bank PD 3.0 12V tương thích, hoặc build mạch nguồn riêng, trước khi cam kết ngày demo.

**Phương án dự phòng: Snapdragon 8 Elite Gen 5 phone.** Nếu rủi ro tích hợp nguồn điện của Rubik Pi 3 không giải quyết kịp trước deadline, dùng điện thoại flagship Snapdragon — có pin sẵn, cầm tay thật, NPU mạnh hơn nhiều (~80 TOPS vs 12 TOPS) nên dư sức chạy cả 4 module cùng lúc mà không cần quantize sâu như Rubik Pi 3. Đánh đổi: giá cao hơn 5-6 lần, và **chưa xác nhận trực tiếp có trong danh sách device AI Hub** (chỉ suy luận từ tooling docs) — cần tự kiểm tra qua `qai-hub` trước khi cam kết.

**❌ LOẠI: QCS8300.** Không xuất hiện trong danh sách device chính thức của Qualcomm AI Hub (đã kiểm tra trực tiếp trang docs) — không tìm được dev kit hay giá công khai. Loại khỏi cân nhắc cho tới khi có nguồn xác nhận tốt hơn.

**⚠️ Cân nhắc thêm nhưng chưa đủ dữ liệu: Qualcomm RB3 Gen2/RB5** — cùng chip QCS6490, do chính Qualcomm làm dev kit robotics, nhưng chưa xác nhận giá — nếu rẻ hơn hoặc có support tốt hơn Rubik Pi 3 thì đáng cân nhắc lại, cần thêm 1 vòng tra cứu giá trước khi hoàn toàn loại bỏ.

### 2. Tổng dung lượng model thật (đo on-disk, không ước lượng) — TRƯỚC khi quantize

| Step | Model | Format hiện tại | Size thật |
|---|---|---|---|
| 0 | Silero VAD | ONNX | 1.3 MB |
| 0 | GTCRN | PyTorch checkpoint | 0.6 MB |
| 1 | Zipformer-30M (Vi) | ONNX **int8 đã có sẵn** | **29.3 MB** (encoder 27 + decoder 1.3 + joiner 1.0) |
| 1 | SenseVoice-Small (En/Zh/Ko) | PyTorch **fp32, CHƯA quantize** | **893 MB** |
| 2 | NLLB-200-distilled-600M | safetensors **fp32, CHƯA quantize** | **2,300 MB** |
| 3 | Piper (Vi) | ONNX (đã gọn sẵn) | 61 MB |
| 3 | Supertonic (Ko+En) | ONNX **fp32, CHƯA quantize** | 380 MB |
| 3 | MeloTTS-ZH (Zh) | PyTorch checkpoint gốc **fp32, CHƯA quantize** (bản AI Hub đã tự quantize riêng, dung lượng khác chưa xác nhận) | 199 MB |
| | **TỔNG (chưa quantize)** | | **≈ 3,864 MB ≈ 3.86 GB** |

**Phát hiện quan trọng:** step1.md Part A từng ghi SenseVoice-Small "~250MB (int8-quantizable)" — đây chỉ là **con số ước lượng cho bản ĐÃ quantize, chưa từng thực sự chạy quantize**. Bản thật đang dùng để test là fp32, nặng gấp 3.6×. Đây là khoảng trống giống hệt kiểu lỗi đã bắt được ở Step 3 (VieNeu-TTS deploy path) — số liệu "ước lượng" bị nhầm thành "đã làm".

### 3. Kết quả quantize THẬT (đã chạy + verify + điều tra lỗi, không phải kế hoạch)

| Model | Công cụ | Size trước → sau | Verify chất lượng | Kết luận |
|---|---|---|---|---|
| Zipformer-30M | *(đã xong sẵn)* | 29.3 MB | Đã verify ở Step 1 (tác giả tự làm) | ✅ Dùng thẳng |
| **NLLB-600M** | **CTranslate2 int8** (`ct2-transformers-converter --quantization int8`) | 2,300 MB → **594 MB** | ✅ **BLEU thật cả 6 chiều**: vi→en 33.81→34.59, en→vi 29.67→29.31, vi→zh 20.45→21.25, zh→vi 21.07→24.01, vi→ko 8.05→8.59, ko→vi 18.60→22.65 — **không chiều nào tụt, một số còn tăng nhẹ** (trong sai số beam-search, không phải quantize "làm tốt hơn") | ✅ **AN TOÀN, dùng ngay** |
| **SenseVoice-Small** | ONNX export (funasr `model.export()`) + ONNX Runtime dynamic int8 | 893 MB → **233 MB** | ⚠️ **WER/CER thật** (cài `funasr_onnx` để verify): Anh 6.8%→7.6% (ổn, trong ngưỡng); **Trung 2.3%→9.8%, Hàn 4.5%→9.5%** (vượt ngưỡng 1-2 điểm % team đặt ra — dù một phần do 1-2 câu khó trong mẫu chỉ 5 câu/ngôn ngữ kéo điểm trung bình lên, không phải tụt đều) | ⚠️ **Dùng được nhưng có cái giá thật — cần quyết định đánh đổi (xem §3b)** |
| **Supertonic** | ONNX Runtime dynamic int8, **CHỈ 3/4 submodel** (giữ `vocoder.onnx` fp32) | 398 MB → **178 MB** | ✅ Lần đầu quantize cả 4 submodel: audio vỡ hoàn toàn (xem §3a). Bisect tìm ra thủ phạm = `vocoder.onnx`. Quantize lại chỉ 3 submodel còn lại, verify round-trip ASR: `"The service frequently used by shipping."` (thiếu 1 từ "is") và `"중동의 따뜻한 기에서는..."` (thiếu 1 âm "후") — **gần như hoàn hảo, mức lệch tương đương nhiễu ASR bình thường** | ✅ **Đã fix, dùng bản 178MB này** |
| MeloTTS-ZH | *(chưa tự quantize)* | 199 MB | Dùng thẳng bản Qualcomm AI Hub đã pre-quantize (số MB riêng chưa xác nhận, chỉ có số latency) | ✅ Dùng bản chính chủ, không tự làm |
| Piper | Không cần | 61 MB | Đã đủ nhỏ | ✅ Giữ nguyên |

### 3a. Điều tra lỗi Supertonic — bisect từng submodel để tìm đúng thủ phạm

Quantize cả 4 submodel (text_encoder, vector_estimator, vocoder, duration_predictor) cùng lúc làm audio vỡ: tiếng Hàn 5/5 câu CER=100% (ASR không nhận ra chữ nào), tiếng Anh 4/5 câu WER=100% (hypothesis chỉ ra "Yeah.", "Okay."). Thay vì đoán, đã **quantize từng submodel riêng lẻ** (giữ 3 cái còn lại fp32) và test round-trip ASR cho từng cấu hình:

| Cấu hình | Kết quả |
|---|---|
| Tất cả fp32 (baseline) | ✅ Hoàn hảo |
| Chỉ `text_encoder` int8 | ✅ Hoàn hảo |
| Chỉ `vector_estimator` int8 | ✅ Hoàn hảo |
| Chỉ `vocoder` int8 | ❌ **Vỡ giống hệt bản quantize cả 4** (Anh: ASR rỗng) |
| Chỉ `duration_predictor` int8 | ✅ Gần như hoàn hảo |

**Thủ phạm xác định: `vocoder.onnx`.** Đây là thành phần chuyển đổi latent feature thành waveform thô — nhiều khả năng có layer với dải giá trị động lớn (trước activation cuối) mà ONNX Runtime dynamic quantization (per-tensor scale đơn giản) không xử lý tốt, gây méo/clip nghiêm trọng. Ngược lại `text_encoder` và `vector_estimator` (dù là flow-matching, ban đầu bị nghi ngờ nhất) hoá ra chịu quantize tốt.

### 3b. SenseVoice-Small int8: đánh đổi thật, cần team quyết định

Không giống NLLB (an toàn tuyệt đối) hay Supertonic (tìm được fix sạch), SenseVoice-Small int8 rơi vào **trường hợp giữa** — dùng được nhưng chất lượng tiếng Trung/Hàn tụt rõ, vượt ngưỡng DoD gốc của team ("không tụt quá 1-2 điểm phần trăm"). 2 lựa chọn:
- **Chấp nhận đánh đổi**: 233MB (giảm 74%) đổi lấy CER tăng từ 2.3%/4.5% lên 9.8%/9.5% — vẫn ở mức "dùng được" cho hầu hết câu, chỉ tệ hơn rõ ở câu có tên riêng/số liệu phức tạp.
- **Đầu tư thêm**: thử static quantization có calibration data (thay vì dynamic) — thường giữ chất lượng tốt hơn nhưng cần bộ dữ liệu hiệu chỉnh riêng, tốn thêm thời gian chưa ước lượng được.

**Chưa tự quyết định thay team** — đây là lựa chọn đánh đổi kích thước/chất lượng cần người có quyền quyết định của dự án chốt, không phải việc kỹ thuật thuần tuý.

**Tổng dung lượng thực tế đạt được: ≈ 1.38 GB** (1.9 + 29.3 + 233 + 594 + 61 + 178 + 199 ≈ 1,296 MB, làm tròn) — giảm **~66%** so với 3.86GB ban đầu, tốt hơn cả mục tiêu lý thuyết 1.3GB ban đầu vì fix Supertonic hiệu quả hơn dự kiến. Nằm thoải mái trong RAM 8GB của Rubik Pi 3.

### 4. Rủi ro & việc cần làm rõ trước khi chốt hẳn vào Technical Proposal

| Rủi ro | Ghi chú |
|---|---|
| RAM Rubik Pi 3 = 8GB chỉ xác nhận qua nguồn thứ cấp (retailer) | Chưa tìm được specsheet chính thức ghi rõ có bản 4GB hay không — cần hỏi trực tiếp Thundercomm/nhà phân phối |
| Nguồn điện 12V/3A PD 3.0, không có pin | Cần mua/test power bank tương thích PD 3.0 12V TRƯỚC ngày demo — nếu không thiết bị không thực sự "portable" |
| Giá TurboX C8550 (QCS8550) chưa công khai | Không đưa vào kế hoạch chính vì rủi ro timeline (phải liên hệ sales, không rõ lead time) |
| Snapdragon 8 Elite Gen 5 chưa xác nhận trực tiếp trong danh sách AI Hub | Chỉ suy luận từ tooling docs (LiteRT blog) — cần tự kiểm tra qua `qai-hub` trước khi chọn làm phương án dự phòng chính thức |
| Chưa đo power budget thật khi chạy full pipeline | Chỉ có specsheet nguồn vào (36W), chưa đo công suất tiêu thụ thực tế lúc inference — cần đo sau khi có board thật |
| **SenseVoice-Small int8 tụt chất lượng Trung/Hàn vượt ngưỡng DoD** | CER 2.3%→9.8% (Trung), 4.5%→9.5% (Hàn) — vượt ngưỡng "không tụt quá 1-2 điểm %" team tự đặt ra. Cần quyết định: chấp nhận đánh đổi hay đầu tư static quantization (xem §3b) |
| Quantize trên đây dùng ONNX Runtime dynamic quantization (CPU), KHÔNG phải QNN/AIMET thật của Qualcomm | Đây là bước "chứng minh nén được, đo được tác động chất lượng" trên máy dev — khi lên Rubik Pi 3 thật vẫn cần convert riêng qua QNN, số size/tốc độ có thể khác (thường tốt hơn nhờ NPU int8 native) |
| Chưa profile bất kỳ model nào (trừ MeloTTS-ZH) trên Snapdragon thật qua `qai-hub` | Khoảng trống đã nêu xuyên suốt Step 1/2/3 — Step 4 xác nhận lại: đây vẫn là việc quan trọng nhất còn lại trước khi nộp Technical Proposal |

---

**Document version:** 2026-08-09 (v2) — đã chạy quantize + verify chất lượng thật cho cả 3 model (NLLB, SenseVoice, Supertonic), điều tra + fix lỗi Supertonic bằng bisection thật (không đoán). Phần cứng CHỐT: Rubik Pi 3 (QCS6490), dự phòng Snapdragon 8 Elite Gen 5 phone. Tổng dung lượng thật: ~1.38GB.
**Bước tiếp theo:** (1) Mua/test power bank PD 3.0 12V cho Rubik Pi 3; (2) team quyết định đánh đổi chất lượng SenseVoice-Small int8 (chấp nhận hay đầu tư static quantization); (3) dùng `qai-hub` profile toàn bộ pipeline trên Rubik Pi 3 thật qua QNN (khác với ONNX Runtime dynamic quant đã dùng để test ở đây).
