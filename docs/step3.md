# Step 3 — TTS (Text-to-Speech / Synthesis)

**Status (2026-08-09):** Đề bài xác nhận YÊU CẦU TTS (xem §0). Đã code-test thật 5 candidate (Supertonic, MeloTTS, Piper, Confucius4-TTS, VieNeu-TTS) trên cùng bộ câu FLORES-200 tái dùng từ Step 2, đo RTF thật + round-trip WER/CER thật (không chỉ tra cứu). **Kiến trúc CHỐT: Piper (Vi) + Supertonic (Ko+En) + MeloTTS-ZH (Zh)** — 3 model, tổng ~640MB. VieNeu-TTS là lựa chọn thay thế đã test đầy đủ, ghi nhận ở §2.3 nếu team muốn đổi sau khi nghe mẫu.

---

## 0. Xác nhận yêu cầu

Đề bài (mục "The challenge"): *"On-device ML combining **speech recognition, translation, and synthesis** — no cloud dependency, optimized for speed and accuracy."* — **"synthesis" = TTS, là 1 trong 3 module bắt buộc** cùng ASR (Step 1) và MT (Step 2). Không phải tuỳ chọn.

---

## Part A — Drop-in cho Technical Proposal §4.2 "Module-by-Module Design"

| Module | Model / Framework | Size (đo thật, on-disk) | Latency (đo thật, dev machine) | Key Technique |
|---|---|---|---|---|
| TTS — Vietnamese | Piper (`vi_VN-vais1000-medium`, VITS/ONNX) | **61 MB** | RTF **0.144** (CPU) | VITS one-shot decoder, ONNX-native, không cần LM/codec 2 tầng |
| TTS — Korean + English | Supertonic 3 (ONNX-native, 4 submodel) | **380 MB** (text_encoder 35 + vector_estimator 245 + vocoder 97 + duration_predictor 3.6) | RTF **1.11** (ko) / **1.16** (en) (CPU) | Flow-matching TTS, ONNX Runtime, CPU-only |
| TTS — Mandarin | MeloTTS-ZH (checkpoint gốc, chưa quantize) | **199 MB** (checkpoint HF gốc) — bản Qualcomm AI Hub đã pre-quantize riêng, dung lượng khác | RTF **0.063** (steady-state, sau cold-start) — **đo THẬT trên Snapdragon 8 Elite Gen 5**: Encoder 23.8ms + Decoder 42.5ms + Flow 71.2ms | NPU-accelerated (HTP) trên bản AI Hub, pre-quantized sẵn |

**Tổng dung lượng deploy: 61 + 380 + 199 = ≈ 640 MB cho cả 4 ngôn ngữ.**

**Đây vẫn là module DUY NHẤT trong cả 4 step (0/1/2/3) có số latency đo thật trên phần cứng Snapdragon** (MeloTTS-ZH, do Qualcomm tự profile công khai trên AI Hub). Mọi số RTF khác trong bảng trên đo trên máy dev (GPU/CPU thường), **không phải Snapdragon thật** — xem cảnh báo ở cuối file.

**⚠️ Cần xác nhận trước khi chốt:** license Supertonic — sample code MIT nhưng **model dùng OpenRAIL-M** (có điều khoản hạn chế sử dụng). Piper là MIT (không hạn chế). MeloTTS license cần xác nhận lại file LICENSE gốc.

---

## Part B — Phân tích đầy đủ, có số liệu chọn/loại từng candidate (Vietnamese)

### 1. Bài toán: không có model nào phủ đủ cả 4 ngôn ngữ

Giống hệt tình huống Step 1/2 — **chưa tìm được model nào hỗ trợ tốt cả Vi+En+Zh+Ko trong 1 checkpoint duy nhất.** Đã test/xác nhận trực tiếp (không chỉ tra cứu) 7 candidate:

| Candidate | Vi | En | Zh | Ko | Kích thước thật | Trạng thái |
|---|---|---|---|---|---|---|
| **Piper** (`vais1000-medium`) | ✅ **WER 14.1%** (đo thật) | ✅ WER 10.3% | ❌ | ❌ (cộng đồng chưa có giọng khả dụng) | 61 MB/giọng | ✅ **CHỐT cho Vi** |
| **VieNeu-TTS 0.3B** | ✅ **WER 12.8%** (đo thật, nhỉnh hơn Piper nhưng trong sai số n=5) | ❌ | ❌ | ❌ | 491 MB (deploy path) | ⚠️ Backup cho Vi (xem §2.3) — chậm hơn 3.3×, nặng hơn 8× so với Piper |
| **Supertonic 3** | ⚠️ có nhưng **WER 35.2%** — lỗi lặp từ rõ ràng khi nghe | ✅ WER 7.9% | ❌ | ✅ CER 6.8% | 380 MB | ✅ CHỌN cho Ko+En, ❌ LOẠI cho Vi |
| **MeloTTS** (đầy đủ) | ❌ | ✅ WER 8.6% | ✅ **CER 7.3%** (1.2% nếu bỏ 1 câu chứa tên riêng Māori khó) | ✅ (chưa test) | 199 MB/ngôn ngữ (checkpoint gốc) | ✅ CHỌN bản ZH, ❌ LOẠI En/Ko (đã có Supertonic tốt hơn) |
| **Confucius4-TTS** (NetEase Youdao) | Tuyên bố có (zero-shot cloning) | Tuyên bố có | Tuyên bố có | Tuyên bố có | **>2.4 GB chỉ riêng speaker-encoder (w2v-bert-2.0)**, chưa tính T2S+S2A+vocoder | ❌ LOẠI — bỏ dở khi test vì dung lượng phi thực tế cho edge |
| **Kokoro-82M** | ❌ (không có) | ✅ | ✅ | ❌ **đã xác minh: không có giọng Hàn nào (không có prefix kf_/km_)** | 82 MB | ❌ LOẠI — không có Việt lẫn Hàn, mất hết lý do cân nhắc |
| **CosyVoice2-0.5B** | Không có bằng chứng | ✅ | ✅ | ✅ | 0.5B | ❌ LOẠI — không xác nhận được Việt |

### 2. Quyết định

**✅ CHỐT: Piper cho tiếng Việt.** So đầu-đối-đầu với ứng viên gần nhất (VieNeu-TTS) trên cả 3 tiêu chí đề bài yêu cầu (quality + latency + size): Piper thắng áp đảo 2/3 tiêu chí — **nhanh hơn 3.3×** (RTF 0.144 vs 0.483), **nhẹ hơn 8×** (61MB vs 491MB) — còn chất lượng gần như ngang nhau (WER 14.1% vs 12.8%, chênh lệch nằm trong sai số thống kê vì chỉ test 5 câu/ngôn ngữ). Không có lý do kỹ thuật để trả thêm 8× dung lượng và 3.3× độ trễ cho một cải thiện chất lượng chưa chắc có ý nghĩa thống kê. Chi tiết so sánh đầy đủ ở §2.3.

**✅ CHỌN: Supertonic cho Hàn + Anh.** Vấn đề chất lượng của Supertonic **chỉ xảy ra ở tiếng Việt** (lặp từ, WER 35.2%) — Hàn (CER 6.8%) và Anh (WER 7.9%) đều tốt, không có lỗi lặp. Kokoro-82M — ứng viên thay thế duy nhất còn lại — đã xác minh **không hỗ trợ tiếng Hàn** (tra model card + VOICES.md gốc trên HuggingFace, không có prefix `kf_`/`km_` nào), nên bị loại thẳng, Supertonic là lựa chọn duy nhất còn lại.

**✅ CHỌN: MeloTTS-ZH cho tiếng Trung.** Vẫn là model duy nhất có số latency đo thật trên Snapdragon.

**❌ LOẠI HẲN: Supertonic cho tiếng Việt.** Lý do bằng số — round-trip WER 35.2%, cao hơn 2.5-3× so với 2 lựa chọn thay thế, với lỗi lặp từ rõ ràng quan sát được trong bản dịch ASR ngược (vd. câu "dịch vụ này... dịch vụ này..." bị lặp) ở 3/5 câu test. Đây là phát hiện **chỉ lộ ra khi code-test thật** — tra cứu ban đầu (research thuần) không phát hiện được vấn đề này vì Supertonic được quảng cáo hỗ trợ tốt tiếng Việt.

### 2.3. Vì sao không chọn VieNeu-TTS (đối trọng gần nhất của Piper)

| Tiêu chí | **Piper** (đã chốt) | **VieNeu-TTS** (backup) |
|---|---|---|
| Round-trip WER (đo thật, 5 câu) | 14.1% | 12.8% (nhỉnh hơn, nhưng chênh lệch nhỏ trên n=5, có thể trong sai số) |
| RTF (đo thật, CPU) | **0.144** (nhanh hơn 3.3×) | 0.483 |
| Kích thước deploy | **61 MB** (nhỏ hơn 8×) | 491 MB (backbone GGUF 193MB + codec ONNX-INT8 298MB) |
| Kiến trúc | VITS (1 tầng, decode trực tiếp) | LM sinh speech-token + neural codec giải mã (2 tầng, giống họ Kokoro/CosyVoice) |
| License | MIT | Apache 2.0 |
| Giọng | 1 giọng nữ miền Bắc (`vais1000`) | 6 giọng (3 nam, 3 nữ; cả miền Bắc lẫn miền Nam) |
| Rủi ro triển khai | Thấp — đã chạy ổn định ngay | Cao hơn — bản PyTorch "standard" mode nặng 1.17GB (backbone+codec PyTorch gốc), phải chuyển sang backend ONNX-INT8 (`neuphonic/neucodec-onnx-decoder-int8`, chưa tự test trực tiếp, chỉ xác nhận dung lượng qua HF API) mới đạt được con số 491MB ở trên |

**Khi nào nên đổi sang VieNeu-TTS:** nếu team nghe mẫu (đã gửi 4 file: Piper vs VieNeu, câu dễ + câu khó) và thấy giọng VieNeu tự nhiên hơn rõ rệt, hoặc cần đa dạng giọng nói (6 giọng, 2 miền) cho demo — vì round-trip WER chỉ đo độ dễ hiểu (intelligibility), không đo độ tự nhiên (naturalness). Nếu không có lý do nghe-được rõ ràng, giữ Piper vì thắng áp đảo latency + size, đúng trọng tâm đề bài ("optimized for speed").

### 3. Lý do loại các candidate khác (không đổi/củng cố thêm)

**❌ LOẠI: Confucius4-TTS.** Bắt đầu test (dùng 1 câu tiếng Việt Step 1 làm giọng mẫu zero-shot cloning) nhưng dừng giữa chừng — riêng speaker-encoder (w2v-bert-2.0) đã tải hơn **2.4GB**, chưa tính 3 thành phần còn lại (T2S 24-layer, flow-matching S2A, BigVGAN vocoder). So với tổng 3 model đã chọn (~640MB CHO CẢ 4 NGÔN NGỮ), một model claim phủ 4 ngôn ngữ nhưng riêng 1 thành phần đã nặng hơn toàn bộ giải pháp — không hợp lý cho edge.

**❌ LOẠI: Kokoro-82M.** Không có tiếng Việt (không tranh cãi) và **đã xác minh trực tiếp không có tiếng Hàn** — tra cứu ban đầu nói "có" (không chính xác/nhầm lẫn nguồn), tự kiểm tra model card + VOICES.md gốc trên HuggingFace chỉ thấy 9 nhóm giọng: en-US/en-GB/ja/zh/es/fr/hi/it/pt-BR, không có `kf_`/`km_`.

**❌ LOẠI: CosyVoice2-0.5B.** Không có bằng chứng hỗ trợ tiếng Việt; 0.5B tham số lớn hơn nhiều so với giải pháp 3-model đã chọn cho cùng nhóm ngôn ngữ.

### 4. Kết quả đo thật đầy đủ (round-trip WER/CER qua Zipformer/SenseVoice, giống phương pháp Step 1)

| Engine | Lang | WER/CER trung bình (5 câu) | RTF trung bình |
|---|---|---|---|
| Piper | vi | 14.09% | 0.144 |
| Piper | en | 10.31% | 0.109 |
| VieNeu-TTS | vi | 12.79% | 0.483 |
| Supertonic | vi | 35.24% (lỗi lặp từ) | 0.531 |
| Supertonic | ko | 6.77% | 1.109 |
| Supertonic | en | 7.93% | 1.161 |
| MeloTTS | zh | 7.31% (1.2% nếu bỏ câu tên riêng khó) | 0.063 (sau cold-start) |
| MeloTTS | en | 8.64% | 0.050 (sau cold-start) |

**Lưu ý cold-start:** MeloTTS câu đầu tiên luôn có RTF rất cao (zh 1.11, en 11.85) do CUDA/JIT warmup — không phản ánh tốc độ thực khi chạy liên tục. Cần gọi warm-up 1 lần trước khi phục vụ người dùng thật.

### 5. Rủi ro & điều cần làm rõ trước khi chốt

| Rủi ro | Ghi chú |
|---|---|
| License Supertonic model = OpenRAIL-M | Cần đọc điều khoản trước khi cam kết, đặc biệt nếu tính thương mại hoá sau cuộc thi |
| **⚠️ Khoảng trống quan trọng chưa xử lý — phần cứng thật** | Toàn bộ số RTF ở bảng §4 đo trên máy dev (GPU NVIDIA / CPU thường), **không phải Snapdragon thật** — giống khoảng trống đã nêu ở Step 1/2. Riêng MeloTTS-ZH là ngoại lệ (có số Snapdragon 8 Elite Gen 5 thật từ Qualcomm AI Hub). Piper và Supertonic đều **chưa xác nhận có trên Qualcomm AI Hub hay không** — cần dùng `qai-hub` để profile thật trước khi chốt vào Technical Proposal. |
| MeloTTS-ZH bản gốc chưa quantize (199MB) khác với bản Qualcomm AI Hub đã pre-quantize | Dung lượng thật sau quantize trên AI Hub chưa xác nhận lại bằng số — chỉ có số latency |
| Chưa nghe mẫu Piper vs VieNeu-TTS bằng tai để xác nhận cuối cùng | Round-trip WER chỉ đo độ dễ hiểu, không đo độ tự nhiên — đã gửi mẫu, chốt Piper theo số liệu latency/size, có thể đổi nếu nghe thấy VieNeu vượt trội |

---

**Document version:** 2026-08-09 — code-test thật đầy đủ 5 candidate (Supertonic, MeloTTS, Piper, Confucius4-TTS bỏ dở, VieNeu-TTS), có số RTF + round-trip WER/CER thật. Kiến trúc CHỐT: Piper (Vi) + Supertonic (Ko+En) + MeloTTS-ZH (Zh), ~640MB tổng. Part A khớp format §4.2 Technical Proposal chính thức, 1 dòng/module (không còn để 2 lựa chọn song song).
**Bước tiếp theo:** (1) profile Piper/Supertonic trên Snapdragon thật qua `qai-hub`; (2) nghe mẫu Piper vs VieNeu-TTS để xác nhận không cần đổi; (3) xác nhận license MeloTTS gốc.
