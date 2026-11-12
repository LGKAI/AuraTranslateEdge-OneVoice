# Các cải tiến chất lượng đã làm (tóm tắt)

Cập nhật 11/11/2026. Pipeline: ASR (Zipformer tiếng Việt, SenseVoice en/zh/ko) → làm sạch văn bản → MT (NLLB-200-distilled-600M, beam 4) → TTS (Piper vi, MeloTTS zh, Supertonic en/ko), chạy trên HSPTEK AIBOX 8550 (QCS8550).

## 1. Đã đưa vào pipeline

| # | Cải tiến | Làm gì | Hiệu quả đo được | Chạy ở đâu trên hộp |
|---|---|---|---|---|
| 1 | **Khôi phục dấu câu và chữ hoa cho ASR tiếng Việt** (vibert-capu, ONNX int8 111 MiB) | ASR tiếng Việt ra chữ thường không dấu; mô hình nhỏ thêm `, . ? :`, viết hoa, đổi số đọc thành chữ số trước khi dịch | chrF của MT trên chữ ASR thật: vi→ko 25,6 → **34,1**, vi→zh 31,0 → **36,3**, vi→en ngang (55,6 → 55,1); trên 280 câu chuẩn bỏ dấu: vi→en +1,6, vi→ko +3,0 | CPU (onnxruntime), khoảng 12–26 ms mỗi câu trên laptop; trên CPU ARM chưa đo |
| 2 | **Phát âm thanh gối đầu theo câu** (chế độ live) | tách đoạn nói tại dấu câu, dịch và đọc từng câu, phát câu 1 trong lúc làm câu 2 | thời điểm bắt đầu nghe giảm khoảng một nửa (CPU: 27,9 s so với 52,4 s; ước tính trên chip khoảng 1,5 s so với 3 s cho đoạn 2 câu) | mã máy chủ; không đổi mô hình |
| 3 | **MT tự hạ cấp** (`MtFallback`) | thử beam lớn nhất trước; nếu thiếu bộ nhớ, lỗi chạy hay đầu ra sai bố cục thì giải phóng đồ thị, hạ xuống beam nhỏ hơn rồi greedy, dịch lại đúng câu đó | không mất câu khi có lỗi NPU; đã thử với lỗi giả lập | mã máy chủ |
| 4 | **Sửa lỗi thứ tự đầu ra TTS tiếng Trung trên chip** | chip trả (độ dài, audio) thay vì (audio, độ dài); TTS giờ chọn đầu ra theo kích thước và kiểu dữ liệu | tránh câm hoàn toàn ở đầu ra tiếng Trung trên hộp | mã máy chủ |
| 5 | **Glossary văn phòng** (bản nháp, 34 thuật ngữ) | cách đọc từ Latin cho Piper (email, deadline, file, Zoom…), dạng chữ cho zh/ko | Piper hết đọc "file" thành "phai lơ", "deadline" thành "đe đơ lain" (mới kiểm ở mức văn bản) | mã máy chủ |
| 6 | **Bỏ câu rỗng hoặc vô nghĩa** | ASR không nghe ra gì thì không phát gì, hiện "xin nói lại" | tránh phát câu vô nghĩa khi demo | mã máy chủ |

## 2. Đang thí nghiệm

| Thí nghiệm | Kết quả đến giờ |
|---|---|
| Chỉnh tham số giải mã MT (bộ dev 552 câu) | tắt hoặc nới chặn lặp 3-gram: +0,57 chrF trung bình, **vi→ko +3,1**; `length_penalty` gần như không đổi. Đang xác nhận trên holdout |
| Chọn lại ứng viên của beam (noisy channel, dịch ngược) | đang đo mức trần; các ứng viên của beam 4 rất giống nhau nên kỳ vọng thấp |

## 3. Đã thử và loại (có số)

| Phương án | Lý do |
|---|---|
| NLLB-1.3B (beam 2) | trên toàn bộ 920 câu: trung bình 43,7 so với 44,0 của 600M; kém có ý nghĩa ở en→vi (−2,7) và vi→ko (−3,8) |
| Dịch bắc cầu qua tiếng Anh | kém dịch thẳng 3–10 chrF |
| Mô hình ngôn ngữ, ưu tiên từ cho ASR | ASR tiếng Việt đã đạt WER khoảng 0,8 %; tên riêng không biết trước |
| Bộ lọc câu dịch hỏng | câu hỏng thật chỉ khoảng 0,2 %, không lặp vòng lần nào trên 2.020 câu |
| Lọc từ đệm, sửa dấu cách tiếng Hàn | gần như không có từ đệm; sửa dấu cách không tăng điểm chung |

## 4. Thí nghiệm trên CPU có đúng khi lên chip không

- **MT và ASR: có.** Inference trên chip QCS8550 (AI Hub) cho cùng đầu vào khớp CPU: MT cos 1,0 và token 100 %; ASR tiếng Việt chọn đúng ký tự ở 100 % khung; SenseVoice khớp tuyệt đối. Thí nghiệm giải mã dùng đúng vòng beam của chip (kiểm khớp 5/5 câu với mã ONNX).
- **TTS: nội dung giống, âm thanh hơi khác.** Piper gần trùng (cos 0,9995). MeloTTS và Supertonic có dạng sóng khác do tính fp16, nhưng ASR nghe lại ra đúng câu.
- **Các cải tiến ở mục 1** chạy bằng Python trên CPU của hộp (dấu câu, beam, tách câu, hạ cấp, glossary), nên hành vi giống hệt laptop. Chỉ thời gian chạy là phải đo lại trên CPU ARM.
- **Chưa kiểm trên hộp thật:** toàn bộ chỉ mới kiểm trên chip ảo của AI Hub (Android). Giai đoạn A2/AP trong `tools/runbook.sh` sẽ xác nhận trên hộp Ubuntu.
