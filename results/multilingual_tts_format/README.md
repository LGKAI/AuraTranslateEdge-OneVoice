# Kiểm tra format TTS Anh/Hàn/Trung

Đã gọi trực tiếp Registry.tts(lang).synthesize() với models/models.cpu.json trên ONNX Runtime CPU. Input là câu thử tự soạn gồm giờ 3:30, ngày 21/12/2027, tiền 18.25, nhiệt độ 36.8, ZIP 04782, điện thoại 0908246135, chuyến bay VN247. Đây là test riêng TTS, không phải S2S, không phải đánh giá nghe thủ công.

## Mặc định

Tạo thành công cả 3 WAV 44.100 Hz, dữ liệu hữu hạn. default/{en,ko,zh}.json ghi văn bản chuẩn hóa, các đoạn thực sự gọi synth, thời gian và đường dẫn audio.

- Cả 3 ngôn ngữ: ZIP và điện thoại bị đổi thành số đếm, mất số 0 đầu; phần 247 của VN247 cũng thành số đếm thay vì từng ký tự theo chính sách người dùng.
- Anh: 3:30 thành three : thirty; ngày thành twenty-one / twelve / two thousand twenty-seven; $18.25 giữ ký hiệu $ và đọc dạng eighteen point two five thay vì chuyển sang dollars/cents. Chia đoạn đã tách hundred thành h / undred trong câu số điện thoại, tạo rủi ro phát âm.
- Hàn: 3:30 thành 삼 : 삼십, thiếu 시/분 và chưa dùng 세 시 trong ngữ cảnh giờ. Số thập phân 36.8 được chuyển đúng giá trị 삼십육 점 팔. Ngày tháng và 18.25달러 không thấy sai giá trị chỉ từ văn bản chuẩn hóa.
- Trung: 3:30 thành 三:三十, thiếu 点/分; năm 2027 chuyển dạng số đếm 二千零二十七 thay vì cách đọc năm thường dùng 二零二七 (không đổi giá trị). Số thập phân và số tiền giữ giá trị; ngày tháng có đủ 年/月/日. Chia đoạn cắt giữa cụm giờ và giữa số điện thoại.

## Nhánh model tiếng Anh

S2S_NEURAL_TN=1 thất bại 2 lần với chính câu thử tổng hợp, lỗi English learned span detection failed source validation; không tạo audio. neural/en.json ghi lỗi của lần chạy sau. Điều này không phủ nhận các regression ngắn đã qua trước đó, nhưng bác bỏ kết luận hệ thống ổn với mọi câu. Hàn và Trung hiện không có nhánh learned TN riêng trong synthesize(), vẫn dùng hàm normalize mặc định ngay cả khi bật cờ.

Chưa chấm phát âm, ngữ điệu, chất lượng cảm nhận; lỗi khẳng định ở đây là lỗi format nhìn thấy trong văn bản/chunk của pipeline và lỗi runtime thực tế.
