# Step 2 — Dịch máy Đa ngữ (Machine Translation - MT)

Mô-đun Step 2 chịu trách nhiệm chuyển dịch văn bản hai chiều giữa **Tiếng Việt và 3 ngoại ngữ**: $\text{Vi} \leftrightarrow \text{En}$, $\text{Vi} \leftrightarrow \text{Zh}$, $\text{Vi} \leftrightarrow \text{Ko}$ (tổng cộng **6 chiều dịch**). Yêu cầu kỹ thuật cốt lõi là phải chạy **hoàn toàn offline (zero cloud dependency)** trên thiết bị biên, bảo toàn chuẩn xác thuật ngữ kỹ thuật/công xưởng, tốc độ sinh từ nhanh (< 1 giây/câu) để ghép nối liền mạch vào chuỗi Speech-to-Speech (S2S).

---

## 1. Mô hình đã chọn: NLLB-200-distilled-600M

Sau khi benchmark đối đầu trên tập ngữ liệu chuẩn quốc tế FLORES-200, dự án chốt sử dụng **Meta NLLB-200-distilled-600M**:

*   **Kiến trúc:** Seq2Seq Transformer (Encoder-Decoder chuyên biệt cho dịch thuật), gồm 600 triệu tham số (~2.3 GB FP32/FP16, nén INT8 qua CTranslate2 còn **594 MB**).
*   **Độ phủ ngôn ngữ:** Huấn luyện trên hơn 200 ngôn ngữ, hỗ trợ đồng thời cả 4 mã ngôn ngữ mục tiêu của dự án:
    *   `vie_Latn`: Tiếng Việt (chữ Quốc ngữ)
    *   `eng_Latn`: Tiếng Anh (chữ Latin)
    *   `zho_Hans`: Tiếng Trung (Hán tự Giản thể)
    *   `kor_Hang`: Tiếng Hàn (chữ Hangul)
*   **Chính sách Streaming (AlignAtt):** Kỹ thuật ngắt phát từ dựa trên ngưỡng ma trận Cross-Attention từ mô hình offline có sẵn (Papi et al., IWSLT 2024). Phương pháp này đạt **Zero Retraining** (không cần huấn luyện lại), giảm độ trễ phát từ (Average Lagging $\text{AL} \approx 2\text{s}$) mà vẫn bảo toàn điểm BLEU tiệm cận bản offline.

> [!TIP]
> **Tối ưu hoá Lượng tử hoá NPU & CTranslate2 (Step 4):**
> Mô hình NLLB-600M đã được nén INT8 thành công qua CTranslate2 (`outputs/nllb-ct2-int8/`), giảm dung lượng từ 2.3 GB xuống **594 MB** mà **điểm BLEU trên cả 6 chiều dịch không hề bị suy giảm** (thậm chí tăng nhẹ trong ngưỡng sai số beam search). Chi tiết xem tại [`docs/step4.md`](file:///d:/ChuyenNganhAI/AuraTranslateEdge-OneVoice/docs/step4.md).

---

## 2. Kết quả Benchmark đối đầu ở Step 2

Đo kiểm thực nghiệm trên tập chuẩn **FLORES-200** (30 câu đa ngữ × 6 chiều dịch = 180 câu):

| Chiều dịch | NLLB-600M *(CHỌN)* | Qwen3-0.6B | Qwen3-1.7B | Đánh giá & Nhận xét |
|---|:---:|:---:|:---:|:---:|
| **Vi $\rightarrow$ En** | **33.81** | 18.44 | 26.85 | NLLB vượt trội hoàn toàn (+15.37 BLEU) |
| **En $\rightarrow$ Vi** | **29.67** | 16.19 | 26.01 | NLLB dịch tiếng Việt tự nhiên, chuẩn ngữ pháp |
| **Vi $\rightarrow$ Zh** | 20.45 | 17.74 | **23.41** | Qwen3-1.7B nhỉnh hơn nhưng chậm hơn 4 lần |
| **Zh $\rightarrow$ Vi** | **21.07** | 10.39 | 17.58 | NLLB xử lý trọn vẹn Hán tự sang tiếng Việt |
| **Vi $\rightarrow$ Ko** | **8.05** | 3.18 | 5.75 | Tiếng Hàn chắp dính làm BLEU bề mặt thấp, nhưng ngữ nghĩa đúng |
| **Ko $\rightarrow$ Vi** | **18.60** | 5.63 | 13.97 | NLLB dịch chuẩn xác từ trợ từ Hangul |
| **Tốc độ xử lý (giây/câu)** | **0.4 – 0.9s** | 2.7 – 6.9s | 1.6 – 10.1s | **NLLB nhanh hơn 3× đến 15× so với Qwen3** |

### Lý do lựa chọn & Loại bỏ:
*   **✅ CHỌN NLLB-200-distilled-600M:** Thắng 5/6 chiều dịch về chất lượng và **thắng tuyệt đối về tốc độ ở mọi chiều**. Kiến trúc Seq2Seq sinh từ nhanh hơn 3–15 lần so với LLM sinh từ tự do.
*   **❌ LOẠI Qwen3-0.6B:** Thua rất xa NLLB ở tất cả 6 chiều dịch (ví dụ Vi $\rightarrow$ En: 18.44 vs 33.81) và tốc độ chậm hơn 3–8 lần.
*   **❌ LOẠI Qwen3-1.7B làm nhánh chính:** Dù nhỉnh hơn ở chiều Vi $\rightarrow$ Zh (23.41 vs 20.45) nhưng thua ở 5 chiều còn lại và tốc độ quá chậm (lên tới 10.1 giây/câu), không đáp ứng được yêu cầu thời gian thực của thiết bị biên.

---

## 3. Cấu trúc thư mục & Các tệp mã nguồn

Thư mục `src/step2_mt/` bao gồm các tệp mã nguồn cốt lõi:

*   [`test_mt_nllb.py`](file:///d:/ChuyenNganhAI/AuraTranslateEdge-OneVoice/src/step2_mt/test_mt_nllb.py): Kịch bản dịch máy NLLB-200, tích hợp giao diện dòng lệnh (CLI) dịch câu bất kỳ và module chạy Benchmark tự động đo điểm BLEU.
*   [`verify_nllb_int8.py`](file:///d:/ChuyenNganhAI/AuraTranslateEdge-OneVoice/src/step2_mt/verify_nllb_int8.py): Script kiểm chứng bản lượng tử hoá INT8 của NLLB qua runtime CTranslate2 siêu nhẹ trên CPU.
*   [`test_mt_qwen3.py`](file:///d:/ChuyenNganhAI/AuraTranslateEdge-OneVoice/src/step2_mt/test_mt_qwen3.py): Mã nguồn đo kiểm đối chứng họ mô hình Qwen3 (0.6B và 1.7B).
*   [`fetch_mt_data.py`](file:///d:/ChuyenNganhAI/AuraTranslateEdge-OneVoice/src/step2_mt/fetch_mt_data.py): Tải ngữ liệu song ngữ kiểm thử FLORES-200 trực tiếp từ kho lưu trữ của Meta AI về `data/mt/manifest.json`.

---

## 4. Hướng dẫn cài đặt & Chạy kiểm thử

### Cài đặt môi trường
Từ thư mục gốc của dự án, đảm bảo đã cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

> **Ghi chú về bộ nhớ đệm:** Mô hình NLLB-600M nạp trực tiếp từ Local Cache tại `~/.cache/huggingface/hub/` và tự động tận dụng GPU NVIDIA (`cuda:0`). Không cần tải lại qua internet khi đã có cache trong máy.

---

### Các lệnh thực thi chi tiết

#### 🔹 1. Dịch thử câu bất kỳ từ dòng lệnh (Interactive CLI)
Bạn có thể dịch bất kỳ văn bản nào giữa 4 ngôn ngữ (`vi`, `en`, `zh`, `ko`):

```bash
# Tiếng Việt -> Tiếng Anh:
python src/step2_mt/test_mt_nllb.py "Xin chào, tôi là trợ lý trí tuệ nhân tạo." --src vi --tgt en

# Tiếng Anh -> Tiếng Việt:
python src/step2_mt/test_mt_nllb.py "Artificial intelligence is transforming our daily lives." --src en --tgt vi

# Tiếng Việt -> Tiếng Trung:
python src/step2_mt/test_mt_nllb.py "Hôm nay thời tiết ở Hà Nội rất đẹp." --src vi --tgt zh

# Tiếng Việt -> Tiếng Hàn:
python src/step2_mt/test_mt_nllb.py "Rất vui được làm việc cùng bạn trong dự án này." --src vi --tgt ko

# Dịch câu mẫu mặc định:
python src/step2_mt/test_mt_nllb.py
```

#### 🔹 2. Chạy Benchmark đo BLEU nhanh (Quick Benchmark)
Để kiểm tra độ chính xác BLEU trên một số câu mẫu mà không cần chạy toàn bộ tập dữ liệu:

```bash
# Đo điểm BLEU trên 3 câu mẫu cho riêng chiều Vi -> En:
python src/step2_mt/test_mt_nllb.py --benchmark --limit 3 --direction vi->en

# Đo điểm BLEU trên 3 câu mẫu cho cả 6 chiều dịch:
python src/step2_mt/test_mt_nllb.py --benchmark --limit 3
```

#### 🔹 3. Chạy Full Benchmark trên toàn bộ tập FLORES-200
```bash
python src/step2_mt/test_mt_nllb.py --benchmark
```
* Script sẽ tự động dịch toàn bộ 30 câu × 6 chiều = 180 câu, đo đạc độ trễ và điểm BLEU đối chiếu với bản dịch chuẩn.
* Kết quả báo cáo được tự động ghi vào tệp [`outputs/mt_nllb_results.csv`](file:///d:/ChuyenNganhAI/AuraTranslateEdge-OneVoice/outputs/mt_nllb_results.csv).

#### 🔹 4. Chạy đối chứng mô hình Qwen3 (Đã loại)
```bash
python src/step2_mt/test_mt_qwen3.py Qwen/Qwen3-0.6B
```

---

## 5. Tích hợp NLLB-200 vào Code Python

Ví dụ đoạn mã ngắn gọn để tích hợp tính năng dịch thuật vào ứng dụng hoặc pipeline:

```python
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "facebook/nllb-200-distilled-600M"
NLLB_CODE = {"vi": "vie_Latn", "en": "eng_Latn", "zh": "zho_Hans", "ko": "kor_Hang"}

# Khởi tạo mô hình
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID).to(device).eval()

def translate(text: str, src: str = "vi", tgt: str = "en") -> str:
    tokenizer.src_lang = NLLB_CODE[src]
    inputs = tokenizer(text, return_tensors="pt").to(device)
    tgt_id = tokenizer.convert_tokens_to_ids(NLLB_CODE[tgt])
    out = model.generate(**inputs, forced_bos_token_id=tgt_id, max_new_tokens=200)
    return tokenizer.batch_decode(out, skip_special_tokens=True)[0]

# Thử nghiệm dịch:
output = translate("Chào mừng bạn đến với dự án AuraTranslate Edge", src="vi", tgt="en")
print(output)  # Welcome to the AuraTranslate Edge project
```
