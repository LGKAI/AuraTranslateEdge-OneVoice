# OneVoice — web demo S2S

Server chỉ dùng thư viện chuẩn của Python, phục vụ chuỗi ASR → MT → TTS và trả từng bước dạng NDJSON để giao diện hiện tiến trình.

- **Engine thật AIBOX / ONNX (`S2S_ENGINE=real`)**: mặc định trên Linux/AIBOX hoặc khi đặt `S2S_MODELS` + `S2S_BACKEND`. Chạy đúng pipeline QNN/ONNX của hộp: VAD → ASR → lọc rác → dấu câu tiếng Việt → MT (route 3.3B/600M, tách câu, kính ngữ tiếng Hàn) → TTS.
- **Engine cục bộ trên máy Windows (`S2S_ENGINE=local`)**: mặc định khi chạy `python web_demo/server.py` trên Windows mà chưa cấu hình `S2S_MODELS`/`S2S_BACKEND`. Dùng trực tiếp model PyTorch/ONNX cài trên máy (`Zipformer-30M`, `SenseVoiceSmall`, `NLLB-200-600M`, `Piper` / `Supertonic`) kết hợp bộ xử lý `s2s_edge` (`clean_asr`, `vi_punct`, `split_sentences`, `clean_mt`, `ko_polite`, `tts_format`).
- **Engine giả (`S2S_ENGINE=mock`)**: giả lập câu mẫu và âm thanh để kiểm tra/quay nhanh giao diện khi chưa cần tải model: `set S2S_ENGINE=mock && python web_demo/server.py` (PowerShell: `$env:S2S_ENGINE="mock"; python web_demo/server.py`).

Trên hộp: `bash start.sh demo` (xem `HUONG_DAN_CHAY_BOX.md`). Laptop: `python tools/laptop_web.py --demo` rồi mở http://localhost:8000.

| API | Việc |
|---|---|
| `GET /api/health` | tên engine, các chiều dịch, thông tin model / MT (3.3B có đang chạy không) |
| `POST /api/translate?src=vi&tgt=en` | body = WAV; trả `asr` → `mt` → `tts` (audio base64) → `done` |
| `POST /api/text?src=vi&tgt=en` | body = `{"text": "..."}`; chữ đã sửa → `mt` → `tts` → `done` (nút ✎ "sửa chữ rồi dịch lại") |

Giao diện: mic (bấm, hoặc giữ Space để nói), tải file / kéo thả, nghe lại âm thanh gốc và bản dịch, ✎ sửa chữ nhận dạng rồi dịch lại, sao chép bản dịch, giao diện sáng/tối, bản điện thoại.

Test (không cần model, engine thật với đồ thị giả): `python web_demo/test_engine.py`.
