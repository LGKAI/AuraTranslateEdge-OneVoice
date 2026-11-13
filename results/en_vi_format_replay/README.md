# Chạy lại S2S: kiểm tra format en-vi

Đã chạy 3 audio gốc trong en-vi/BTC qua Translator của OneVoice_integrated, không cung cấp transcript, không sửa code production. Model ASR/MT/TTS dùng models/models.cpu.json với ONNX Runtime CPU; nhánh neural dùng thêm model chuẩn hóa trên Apple Metal.

Hai cấu hình: default (S2S_NEURAL_TN=0) và neural (S2S_NEURAL_TN=1). Đây là lần chạy mới; không thay output gốc. Thời gian phụ thuộc tải máy và lần đầu nạp model, không phải benchmark ổn định. Không chấm nghe thủ công.

## Kết quả từ văn bản thực sự đưa vào TTS

- EV-3775: default có “năm : ba mươi sáng”; neural có “năm giờ ba mươi sáng”. Cải thiện format giờ.
- EV-5301: default có “tám : không”; neural có “tám giờ”. Trong neural, bước chuẩn bị nguồn đã đổi 800 thành 8:00 trước MT; kết quả không thể quy riêng cho bước TTS. Từ “báo động” vẫn chưa chuẩn ngữ cảnh báo thức.
- EV-0358: default đọc ZIP 67297 thành số đếm; neural đọc “sáu bảy hai chín bảy” đúng chính sách. Tuy nhiên ByT5 đổi số nhà 734 thành “năm ba mươi bốn”: lỗi mới làm sai giá trị số, chưa đạt toàn câu. Trace cho thấy CRF chỉ đánh dấu ZIP, 734 không được bảo vệ; ByT5 sinh sai dù không còn chữ số. Kiểm tra không còn chữ số chưa đủ bảo đảm bảo toàn giá trị.

## File đối chiếu

Mỗi thư mục default/ và neural/ có 3 WAV mới, JSON ghi ASR trước/sau chuẩn hóa, bản dịch, tts_text thực tế, thời gian và decoder. neural/normalization.jsonl ghi từng bước chuẩn hóa. verification.json kiểm tra WAV hữu hạn, số mẫu, hash input/output và code/config.
