# Hội thoại Đa ngữ, Xử lý Tên riêng và Tiếng nói Trộn ngôn ngữ (Code-Switching)

## 1. Chuyển đổi ngôn ngữ theo lượt nói: Chế độ `--pair vi,ko` và `--auto`

Mỗi lượt phát ngôn (utterance) được định tuyến độc lập, cho phép người nói thay đổi ngôn ngữ linh hoạt giữa các lượt nói, và ngôn ngữ đối tác có thể thay đổi tự do trong suốt phiên làm việc.

```bash
python -m s2s_edge live --pair vi,ko  --models models/models.json          # Việt <-> Hàn, tự động chọn chiều dịch theo từng câu nói
python -m s2s_edge live --auto --partner en --models models/models.json        # Việt <-> bất kỳ ngôn ngữ nào được nói (en / zh / ko)
python -m s2s_edge file --auto --in meeting.wav --out meeting_translated.wav   # Tệp ghi âm gồm nhiều ngôn ngữ xen kẽ
```

Hệ thống nhận dạng ngôn ngữ (Language Identification - LID) hoạt động mà **không tốn thêm mô hình phụ**:

1. **Mô hình CTC Tiếng Việt (Zipformer-150M)** là ASR có độ trễ thấp nhất (~163 ms trên chip QCS8550) sẽ chạy trước tiên. Mô hình hồi quy logistic dựa trên 6 đặc trưng độ tin cậy (`assets/lid_vi.json`: xác suất cực đại trung bình trên các khung không rỗng, tỷ lệ từ hợp lệ trong âm tiết tiếng Việt, tỷ lệ khung không rỗng, xác suất rỗng trung bình, số lượng từ) sẽ đưa ra quyết định: *"Là Tiếng Việt hay Không phải Tiếng Việt"*.
2. **Nếu không phải tiếng Việt:**
   - Nếu chỉ có 1 ngôn ngữ đối tác ➔ Tự động chọn ngôn ngữ đó.
   - Nếu ở chế độ tự động (`--auto`) ➔ Chạy SenseVoice ở chế độ `auto`, hệ thống tự động kiểm tra bảng chữ cái đầu ra để phân loại: Chữ Hangul ➔ `ko`, Hán tự ➔ `zh`, Ký tự Latinh ➔ `en`. Văn bản nhận dạng này được tái sử dụng trực tiếp làm bản ghi ASR, loại bỏ hoàn toàn việc phải chạy lại lượt ASR thứ hai.
3. Trong chế độ `--auto`, ngôn ngữ đối tác có tính "nhớ ngữ cảnh" (sticky): Câu tiếng Việt tiếp theo sẽ tự động được dịch sang ngôn ngữ vừa được nói ở lượt trước đó (tham số `--partner` dùng để thiết lập ngôn ngữ mặc định ban đầu).

### Kết quả đo kiểm thực nghiệm trên chuẩn đọc FLEURS (30 clips mỗi ngôn ngữ, `tools/lid_features.py`):

| Phép thử | Kết quả |
|---|---|
| Phân biệt Tiếng Việt vs Ngoại ngữ (Âm thanh sạch, 8–15s) | Kiểm định chéo 10-fold đạt **120 / 120 (100%)** chính xác; clip tiếng Việt có xác suất $\ge 0.70$, ngoại ngữ $\le 0.31$ |
| Phân loại en / zh / ko dựa trên bảng chữ cái SenseVoice (auto) | **90 / 90 (100%)** chính xác |
| Biên độ của ngưỡng đơn lẻ (`conf_nb`) | Rất hẹp: vi $\ge 0.765$, ngoại ngữ $\le 0.753$ ➔ Đây là lý do nhóm sử dụng mô hình kết hợp 6 đặc trưng |

### Kiểm thử độ bền vững trên tập tách biệt (Held-out, 10 clips mỗi ngôn ngữ, `tools/lid_robust.py`):

| Điều kiện kiểm thử | Nhận diện đúng tiếng Việt | Tuyệt đối không nhầm ngoại ngữ thành tiếng Việt |
|---|:---:|:---:|
| Trọn vẹn clip gốc | 10 / 10 | Đạt chuẩn (xác suất nhận nhầm tối đa 0.12) |
| Cắt ngắn 3 giây | 10 / 10 | Đạt chuẩn (xác suất tối đa 0.26) |
| Cắt ngắn 1.5 giây | 6 / 10 | Đạt chuẩn (xác suất tối đa 0.10) |
| 3 giây + Nhiễu đám đông babble ở 5 dB SNR | 7 / 10 | Đạt chuẩn (xác suất tối đa 0.08) |

Lỗi chỉ xảy ra theo một chiều: các câu tiếng Việt quá ngắn hoặc bị nhiễu nặng có thể bị bỏ sót, nhưng tuyệt đối không bao giờ nhận nhầm tiếng nước ngoài thành tiếng Việt. Do đó, bộ phân đoạn tự động ghép nối các khoảng lặng ngắn hơn 1.5s quanh các câu ngắn, và các đoạn âm thanh rác không rõ nghĩa sẽ bị loại bỏ thay vì cố dịch bừa.

---

## 2. Xử lý Tên riêng và Thuật ngữ Ngoại lai (Proper Names)

Hiện trạng và giải pháp khắc phục qua từng chặng pipeline:

| Chặng xử lý | Vấn đề phát sinh | Giải pháp khắc phục của hệ thống |
|---|---|---|
| **ASR** | Tên riêng bị nhận diện sai chính tả (do ASR không có từ điển tên riêng chuyên biệt) | Bổ sung mục `_asr_fix` trong từ điển thuật ngữ (`glossary`) để sửa các từ quen thuộc trong nhà máy (ví dụ: `"ếch xen"` ➔ `"Excel"`, `"ét pi ai"` ➔ `"SPI"`). |
| **MT (NLLB)** | Tự động sao chép hoặc chuyển tự chính xác (`Recep Tayyip Erdogan` ➔ `雷杰普·塔伊普·埃尔多安`, `Samsung` ➔ `三星`) | Giữ nguyên đầu ra chuẩn của NLLB. |
| **TTS Tiếng Việt (Piper)** | **Tự động bỏ qua mọi từ nằm ngoài bảng âm tiết tiếng Việt** (tên riêng, từ mượn tiếng Anh) | Tích hợp module `names.py`: Tự động phiên âm các từ Latinh này thành chuỗi âm tiết tiếng Việt mô phỏng cách đọc tự nhiên của người Việt (dựa trên âm vị CMUdict và quy tắc chữ cái), ví dụ: `Pamela Ferguson` ➔ `pe mơ lơ phơ gơ sơn`. Các từ tiếng Việt chuẩn được giữ nguyên. Kiểm thử trên 84 âm tiết của 31 tên riêng: chỉ có 6 âm tiết nằm ngoài từ điển Piper. |
| **TTS Tiếng Trung (MeloTTS)** | Bỏ qua chữ cái Latinh | Các tên Latinh còn sót lại được chính NLLB dịch chuyển tự sang Hán tự (chạy riêng tên tiếng Anh ➔ tiếng Trung) và lưu cache. |
| **TTS Tiếng Hàn (Supertonic)** | Bỏ qua chữ cái Latinh | Tương tự, chuyển tự sang chữ Hangul qua NLLB (Anh ➔ Hàn). |
| **TTS Tiếng Anh (Supertonic)** | Không bị lỗi | Đọc trực tiếp các tên chữ Latinh mượt mà. |

### Cấu hình Từ điển Thuật ngữ tùy chỉnh (Custom Glossary):
Người dùng có thể nạp tệp từ điển JSON (`--glossary glossary.example.json`) để cố định cách đọc và dịch cho từng tên riêng / thương hiệu nhà máy:
```json
{
  "Samsung": { "vi": "sam sung", "zh": "三星", "ko": "삼성" },
  "Foxconn": { "vi": "phốc con", "zh": "富士康", "ko": "폭스콘" }
}
```

---

## 3. Hiện tượng Trộn Ngôn ngữ trong CÙNG MỘT Câu nói (Code-Switching)

* **Tiếng Trung trộn Tiếng Anh trong 1 câu:** SenseVoice hỗ trợ nhận diện tự nhiên (native support).
* **Tiếng Việt chèn từ Tiếng Anh đàm thoại (*"check giúp em file Excel"*):** Bộ CTC tiếng Việt sẽ phiên âm từ tiếng Anh theo các âm tiết nghe na ná tiếng Việt, khiến mô hình dịch NLLB nhận phải văn bản vô nghĩa.
  - *Giải pháp hiện tại:* Thiết lập từ điển ánh xạ `_asr_fix` cho các thuật ngữ thường dùng tại nhà máy.
  - *Giải pháp triệt để dài hạn:* Cần thu thập tập dữ liệu tiếng Việt pha trộn tiếng Anh trong nhà máy để fine-tune ASR song ngữ chuyên biệt.
* **Nhiều người nói đè âm cùng lúc (Overlapping Speech):** Hiện tại hệ thống xử lý theo luồng âm thanh đơn kênh (Single stream, chưa tích hợp phân tách người nói Diarization).
