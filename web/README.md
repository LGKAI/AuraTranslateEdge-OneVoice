# OneVoice mic web

Mở http://localhost:8766 trên cùng máy Mac. Chọn chiều, bấm mic, cho phép microphone rồi bấm lại để dừng và dịch. Có thể tải file audio; trình duyệt chuyển về PCM WAV mono 16 kHz trước khi gửi. Mỗi bản thu tối đa 30 giây.

Nếu server chưa chạy, double-click `tools/run_s2s_web.command`. Launcher dùng môi trường `/private/tmp/s2s-original-env/bin/python`, tạo tiến trình server riêng và mở trang. Backend bind 127.0.0.1; không phải URL public cho điện thoại/máy khác. PID nằm ở `results/mic_web/server.pid`.

Thu âm dùng getUserMedia + AudioWorklet, không dùng Web Speech Recognition và không truyền transcript từ trình duyệt. Worker gọi Translator/Engine thật, bật chuẩn hóa học máy. Các job được xử lý tuần tự; ONNX sessions được giải phóng sau mỗi job, model chuẩn hóa được giữ trong worker. Audio input/output, transcript và audit lưu trong `results/mic_web/<job-id>/`.

Kiểm tra: `tools/check_s2s_web.py` đã gửi hai WAV qua HTTP và nhận audio ASR→MT→TTS ở cả hai chiều; bản thu không hợp lệ bị từ chối. UI đã được mở trong Chrome. Thử mic tự động bằng Chrome fake-audio trả audio im lặng, nên không dùng thử đó để khẳng định mic phần cứng đã được xác nhận; người dùng cho phép và thử mic thật trong Chrome. Việc gửi file audio qua API đã được xác nhận độc lập.

Log server: `logs/s2s_web.log`. Server chạy độc lập để không bị đóng khi foreground tool kết thúc. Nếu cần dừng, dùng PID đúng trong file PID; chỉ dừng tiến trình của server này.
