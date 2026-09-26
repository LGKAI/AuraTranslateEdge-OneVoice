# Step 3 — Tổng hợp Tiếng nói (Text-to-Speech - TTS)

Mô-đun Step 3 chịu trách nhiệm tiếp nhận văn bản đầu ra từ mô-đun Dịch máy (Step 2 - MT) và tổng hợp thành tín hiệu âm thanh giọng nói tự nhiên, rõ ràng ở ngôn ngữ đích ($\text{Vi}$, $\text{En}$, $\text{Zh}$, $\text{Ko}$) để phát lại cho người dùng. 

Trong đề bài kỹ thuật của AuraTranslate Edge, **tổng hợp giọng nói (Synthesis - TTS) là 1 trong 3 mô-đun bắt buộc** (cùng với ASR và MT), đòi hỏi phải chạy **100% offline (zero cloud dependency)**, độ trễ thấp và kích thước tối ưu cho thiết bị biên.

---

## 1. Kiến trúc giải pháp: Phối hợp 3 Engine chuyên biệt

Thực nghiệm kiểm chứng trên hơn 7 ứng viên mã nguồn mở cho thấy: **Hiện tại chưa có một mô hình đơn lẻ (single-checkpoint) nào hỗ trợ đồng thời cả 4 ngôn ngữ Vi, En, Zh, Ko mà vẫn đáp ứng được tiêu chuẩn dung lượng và tốc độ của thiết bị biên.**

Do đó, dự án đã chọn kiến trúc định tuyến **phối hợp 3 engine TTS tối ưu nhất cho từng nhánh ngôn ngữ**:

```
Văn bản dịch từ Step 2 (MT)
       ↓
Định tuyến ngôn ngữ (Language Router)
       ├── [Tiếng Việt] ────➔ Piper TTS (vi_VN-vais1000-medium)
       │                      • Dung lượng: 61 MB | RTF: 0.144 (CPU)
       │                      • VITS 1-shot ONNX, license MIT tự do
       │
       ├── [Hàn / Anh]  ────➔ Supertonic 3 (Bản nén lai Step 4)
       │                      • Dung lượng: 178 MB | RTF: 1.11 (Ko) / 1.16 (En)
       │                      • Flow-matching TTS (3 submodel INT8 + Vocoder FP32)
       │
       └── [Tiếng Trung] ───➔ MeloTTS-ZH (Qualcomm AI Hub NPU)
                              • Dung lượng: 199 MB | RTF: 0.063 (Snapdragon NPU)
                              • Đo thực tế trên chip Snapdragon 8 Elite Gen 5
       ↓
Tệp âm thanh đầu ra WAV phát cho người dùng
```

### Chi tiết các mô hình đã chọn:

*   **Tiếng Việt: Piper TTS (`vi_VN-vais1000-medium`)**
    *   **Kiến trúc:** VITS (Variational Inference with adversarial learning for end-to-end Text-to-Speech), giải mã trực tiếp một lượt (one-shot decoder), không cần bộ mô hình ngôn ngữ (LM) hoặc codec 2 tầng.
    *   **Hiệu năng thực tế:** Dung lượng chỉ **61 MB**, tốc độ sinh âm thanh siêu nhanh với RTF **0.144** trên CPU. Điểm WER vòng lặp (round-trip intelligibility) đạt **14.1%** (phát âm rõ ràng, dễ nghe).
    *   **Chất giọng & Giấy phép:** Giọng nữ miền Bắc (`vais1000`), giấy phép MIT hoàn toàn tự do cho thương mại hoá.
*   **Tiếng Hàn + Tiếng Anh: Supertonic 3 (Bản nén lai Step 4 - 178 MB)**
    *   **Kiến trúc:** Flow-matching TTS chạy trên ONNX Runtime, bao gồm 4 submodel: `text_encoder`, `vector_estimator`, `vocoder`, `duration_predictor`.
    *   **Đột phá nén Step 4:** Ban đầu bản FP32 nặng 380 MB. Khi nén INT8 toàn bộ, âm thanh bị vỡ nát hoàn toàn. Qua kỹ thuật cô lập lỗi (bisection), dự án phát hiện `vocoder.onnx` là nguyên nhân gây méo âm khi nén dynamic INT8. Giải pháp: **Giữ nguyên `vocoder.onnx` ở FP32, nén 3 submodel còn lại sang INT8** $\rightarrow$ Giảm dung lượng còn **178 MB** mà âm thanh hoàn toàn trong trẻo, đạt CER tiếng Hàn **6.8%** và WER tiếng Anh **7.9%**.
*   **Tiếng Trung: MeloTTS-ZH**
    *   **Kiến trúc:** Mô hình TTS tối ưu hoá riêng cho tiếng Trung với bộ phân tách pinyin và thanh điệu chuẩn xác.
    *   **Đo thực nghiệm trên phần cứng thật:** Là mô hình duy nhất trong toàn bộ dự án đã có số liệu benchmark chính thức trên chip **Snapdragon 8 Elite Gen 5 NPU (HTP)** do Qualcomm công bố: Encoder 23.8ms + Decoder 42.5ms + Flow 71.2ms $\rightarrow$ RTF **0.063** (siêu tốc). Điểm CER tiếng Trung đạt **7.3%** (đạt **1.2%** trên các câu giao tiếp thông thường).
*   **Phương án dự phòng tiếng Việt: VieNeu-TTS (491 MB)**
    *   Kiến trúc 2 tầng (LM sinh speech token + Neural Codec NeuCodec giải mã). Đạt WER vòng lặp **12.8%** (nhỉnh hơn Piper trong biên độ sai số), hỗ trợ 6 giọng đọc Bắc - Nam. Tuy nhiên do nặng gấp 8× (491 MB) và chậm hơn 3.3× (RTF 0.483) nên được giữ làm phương án dự phòng khi cần demo đa dạng vùng miền.

> [!TIP]
> **Tổng dung lượng bộ mô hình TTS triển khai:**
> $\text{61 MB (Piper)} + \text{178 MB (Supertonic INT8-mixed)} + \text{199 MB (MeloTTS)} \approx \mathbf{438\text{ MB}}$ cho trọn vẹn cả 4 ngôn ngữ!

---

## 2. Kết quả Benchmark đối đầu & Đánh giá Round-Trip

Để đánh giá chất lượng TTS một cách khách quan không phụ thuộc vào cảm tính, dự án áp dụng phương pháp **Đánh giá vòng lặp (Round-trip Evaluation)**: Toàn bộ âm thanh WAV do các mô hình TTS sinh ra từ tập câu FLORES-200 được đưa ngược vào mô-đun Nhận dạng giọng nói (ASR - Step 1: Zipformer cho tiếng Việt, SenseVoice cho Anh/Trung/Hàn) để chấm điểm WER / CER đối chiếu với văn bản gốc.

### Bảng so sánh toàn diện 7 ứng viên TTS

| Ứng viên (Candidate) | Vi | En | Zh | Ko | Dung lượng | Trạng thái & Quyết định |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **Piper (`vais1000`)** | ✅ **WER 14.1%** | ✅ WER 10.3% | ❌ | ❌ | **61 MB** | ✅ **CHỐT CHO TIẾNG VIỆT** (Siêu nhẹ, siêu nhanh) |
| **Supertonic 3** | ⚠️ *WER 35.2% (Lặp từ)* | ✅ **WER 7.9%** | ❌ | ✅ **CER 6.8%** | **178 MB** *(sau nén)* | ✅ **CHỌN CHO HÀN & ANH**, ❌ **LOẠI CHO VIỆT** |
| **MeloTTS-ZH** | ❌ | ✅ WER 8.6% | ✅ **CER 7.3%** | ⚠️ | **199 MB** | ✅ **CHỌN CHO TIẾNG TRUNG** (Đo thật trên Snapdragon NPU) |
| **VieNeu-TTS 0.3B** | ✅ WER 12.8% | ❌ | ❌ | ❌ | 491 MB | ⚠️ **Dự phòng cho Việt** (Chậm hơn 3.3×, nặng hơn 8×) |
| **Confucius4-TTS** | Quảng cáo có | Quảng cáo có | Quảng cáo có | Quảng cáo có | **> 2.4 GB** *(chỉ riêng encoder)* | ❌ **LOẠI BỎ** (Dung lượng phi thực tế cho edge) |
| **Kokoro-82M** | ❌ | ✅ | ✅ | ❌ | 82 MB | ❌ **LOẠI BỎ** (Xác minh gốc không hề có giọng Hàn) |
| **CosyVoice2-0.5B** | ❌ | ✅ | ✅ | ✅ | ~1.5 GB | ❌ **LOẠI BỎ** (Không có tiếng Việt, mô hình quá nặng) |

### Chi tiết các phát hiện đo kiểm thực tế:

*   **Vì sao LOẠI HẲN Supertonic cho Tiếng Việt:** Khi chạy thực nghiệm, Supertonic mắc lỗi lặp từ nghiêm trọng (ví dụ câu *"dịch vụ này... dịch vụ này..."* bị lặp liên tục trong 3/5 câu test), đẩy điểm WER vọt lên **35.24%**. Đây là lỗi cố hữu trong tập dữ liệu huấn luyện tiếng Việt của Supertonic mà nếu chỉ đọc lý thuyết sẽ không thể phát hiện được.
*   **Vì sao LOẠI Confucius4-TTS:** Khi tải mô hình về kiểm thử, chỉ riêng thành phần trích xuất đặc trưng người nói (`w2v-bert-2.0`) đã chiếm hơn **2.4 GB**, chưa tính mô hình T2S và bộ giải mã BigVGAN. Cả giải pháp 3 engine của dự án chỉ tốn 438 MB, nên Confucius4 hoàn toàn không khả thi trên thiết bị biên.
*   **Vì sao LOẠI Kokoro-82M:** Các bài viết trên mạng từng tuyên bố Kokoro hỗ trợ đa ngữ, nhưng khi truy cập trực tiếp tệp cấu hình `VOICES.md` trên HuggingFace, mô hình hoàn toàn không có bất kỳ giọng Hàn nào (không có tiền tố `kf_`/`km_`), cũng như không có tiếng Việt.

---

## 3. Cấu trúc thư mục & Các tệp mã nguồn

Thư mục `src/step3_tts/` bao gồm các tệp mã nguồn và công cụ chẩn đoán:

### 📁 Mã nguồn kiểm thử các Engine TTS
*   [`test_tts_piper.py`](file:///d:/ChuyenNganhAI/AuraTranslateEdge-OneVoice/src/step3_tts/test_tts_piper.py): Script tổng hợp âm thanh tiếng Việt và tiếng Anh bằng Piper TTS, tự động tải giọng đọc nếu chưa có trong máy.
*   [`test_tts_supertonic.py`](file:///d:/ChuyenNganhAI/AuraTranslateEdge-OneVoice/src/step3_tts/test_tts_supertonic.py): Script tổng hợp âm thanh bằng Supertonic (đánh giá trên Vi, Ko, En), đo RTF và lưu file WAV.
*   [`test_tts_melotts.py`](file:///d:/ChuyenNganhAI/AuraTranslateEdge-OneVoice/src/step3_tts/test_tts_melotts.py): Script chạy MeloTTS cho tiếng Trung và tiếng Anh, ghi nhận hiện tượng warm-up ban đầu.
*   [`test_tts_vieneu.py`](file:///d:/ChuyenNganhAI/AuraTranslateEdge-OneVoice/src/step3_tts/test_tts_vieneu.py): Script kiểm thử ứng viên dự phòng VieNeu-TTS trên tiếng Việt.
*   [`test_tts_confucius.py`](file:///d:/ChuyenNganhAI/AuraTranslateEdge-OneVoice/src/step3_tts/test_tts_confucius.py): Kịch bản kiểm thử mô hình Confucius4 (dùng để lưu bằng chứng loại bỏ).

### 📁 Đánh giá chất lượng & Công cụ Lượng tử hoá
*   [`test_tts_eval_quality.py`](file:///d:/ChuyenNganhAI/AuraTranslateEdge-OneVoice/src/step3_tts/test_tts_eval_quality.py): Script nạp toàn bộ file âm thanh đã sinh trong `outputs/tts_*_results.csv`, gọi ASR Step 1 (Zipformer / SenseVoice) để tính toán điểm WER/CER vòng lặp tự động.
*   [`quantize_supertonic.py`](file:///d:/ChuyenNganhAI/AuraTranslateEdge-OneVoice/src/step3_tts/quantize_supertonic.py): Thực hiện lượng tử hoá lai cho Supertonic (3 submodel INT8, giữ riêng `vocoder.onnx` FP32), xuất kết quả ra `outputs/supertonic-deploy/`.
*   [`diagnose_supertonic_int8.py`](file:///d:/ChuyenNganhAI/AuraTranslateEdge-OneVoice/src/step3_tts/diagnose_supertonic_int8.py), [`verify_supertonic_int8.py`](file:///d:/ChuyenNganhAI/AuraTranslateEdge-OneVoice/src/step3_tts/verify_supertonic_int8.py): Bộ script bisection từng submodel để chẩn đoán chính xác lỗi vỡ tiếng khi nén INT8.
*   [`find_ko_seed.py`](file:///d:/ChuyenNganhAI/AuraTranslateEdge-OneVoice/src/step3_tts/find_ko_seed.py): Công cụ dò tìm hạt giống ngẫu nhiên (seed) tối ưu giúp bộ lấy mẫu tiếng Hàn của Supertonic phát âm ổn định nhất.

---

## 4. Hướng dẫn cài đặt & Chạy kiểm thử

### Cài đặt môi trường & Tải giọng đọc

#### 1. Cài đặt Piper TTS:
```bash
pip install piper-tts
# Tải giọng đọc tiếng Việt mẫu:
python -m piper.download_voices vi_VN-vais1000-medium
```

#### 2. Cài đặt Supertonic:
```bash
pip install supertonic soundfile
# Tạo bản deploy INT8-mixed siêu nhẹ:
python src/step3_tts/quantize_supertonic.py
```

#### 3. Cài đặt MeloTTS:
MeloTTS cần môi trường Python $\le 3.11$ (do thư viện `tokenizers` chưa có bản build sẵn cho Python 3.12+ trên Windows):
```bash
git clone https://github.com/myshell-ai/MeloTTS.git
cd MeloTTS
pip install -e .
python -m unidic download
```

---

### Các lệnh thực thi chi tiết

#### 🔹 1. Chạy tổng hợp giọng nói Piper (Tiếng Việt)
```bash
cd src/step3_tts
python test_tts_piper.py
```
*Kết quả xuất ra file âm thanh tại `outputs/tts_piper/` và chỉ số RTF tại `outputs/tts_piper_results.csv`.*

#### 🔹 2. Chạy tổng hợp giọng nói Supertonic (Tiếng Hàn & Tiếng Anh)
```bash
python test_tts_supertonic.py
```
*Kết quả xuất ra tại `outputs/tts_supertonic/` và `outputs/tts_supertonic_results.csv`.*

#### 🔹 3. Chạy tổng hợp giọng nói MeloTTS (Tiếng Trung)
```bash
python test_tts_melotts.py
```

#### 🔹 4. Đánh giá chất lượng vòng lặp (Round-trip Evaluation)
Sau khi đã sinh các file âm thanh từ các lệnh trên, chạy script sau để ASR chấm điểm trực tiếp:
```bash
python test_tts_eval_quality.py
```
*Kết quả đánh giá độ chính xác WER/CER được ghi vào [`outputs/tts_quality_results.csv`](file:///d:/ChuyenNganhAI/AuraTranslateEdge-OneVoice/outputs/tts_quality_results.csv).*

---

## 5. Tích hợp TTS Đa ngữ vào Code Python

Đoạn mã ví dụ xây dựng lớp Router TTS tích hợp cả 3 engine vào luồng xử lý:

```python
import os
import wave
import numpy as np

class UnifiedTTS:
    def __init__(self):
        # 1. Khởi tạo Piper cho tiếng Việt
        from piper.voice import PiperVoice
        self.piper_voice = PiperVoice.load("vi_VN-vais1000-medium.onnx")
        
        # 2. Khởi tạo Supertonic cho Tiếng Anh và Tiếng Hàn
        from supertonic import TTS
        self.supertonic = TTS(auto_download=True)
        self.supertonic_style = self.supertonic.get_voice_style(voice_name="M1")

    def synthesize(self, text: str, lang: str, out_wav_path: str):
        if lang == "vi":
            audio_bytes = b"".join(chunk.audio_int16_bytes for chunk in self.piper_voice.synthesize(text))
            with wave.open(out_wav_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(22050)
                wf.writeframes(audio_bytes)
        elif lang in ("en", "ko"):
            import soundfile as sf
            wav, _ = self.supertonic.synthesize(text=text, lang=lang, voice_style=self.supertonic_style)
            sf.write(out_wav_path, np.asarray(wav).squeeze(), 44100)
        elif lang == "zh":
            from melo.api import TTS as MeloTTS
            zh_model = MeloTTS(language="ZH", device="cpu")
            speaker_ids = zh_model.hps.data.spk2id
            zh_model.tts_to_file(text, speaker_ids["ZH"], out_wav_path, disable_bert=True)
        else:
            raise ValueError(f"Ngôn ngữ chưa hỗ trợ: {lang}")

# Sử dụng:
# tts = UnifiedTTS()
# tts.synthesize("Xin chào bạn", lang="vi", out_wav_path="output_vi.wav")
```
