# Tích hợp ASR từ asr-improvement

Nguồn: `1e5231886dbf4af5bb0a6ff722bfa16223f8f3a6`. Đã đọc báo cáo `reports/tts_format_s2s_eval_2026-11-13.md`.

## Đã tích hợp
- VAD: pad 120 ms, silence 320 ms, merge gap 0.32 s, short 1.1 s.
- Viết hoa/dấu hỏi/cảm thán/ngắt mệnh đề VI từ upstream; áp dụng sau CAPU.
- ORT: tắt optimizer với graph >200 MB/external data, retry không optimization khi load lỗi.
- Không thay ASR weights, wrapper asr.py, model registry, TTS hybrid hoặc nhánh chuẩn bị nguồn neural. Không merge toàn branch, không reset thay đổi đang có.

## Điều chỉnh bắt buộc sau kiểm tra
- Upstream biến "tôi không biết khi nào" thành cảm thán: chặn theo cấu trúc câu phủ định đã có.
- Upstream biến "Năm, hai nghìn, không." / "Gọi số không." thành câu hỏi: bảo vệ zero trong ngữ cảnh số. Không khôi phục số/chữ đã mất ở ASR.

## Kết quả
- 22 nhóm unit test qua; gồm 15 test cũ, regression dấu câu, VAD và fallback ORT (mock).
- 6/6 S2S thực hoàn tất cả trước/sau; cùng hash audio đầu vào. JSON giữ ASR/MT/tts_text/chunk, WAV có dữ liệu hữu hạn.
- Hash TTS hybrid, wrapper ASR và registry giữ nguyên. Đây không phải kiểm chứng chất lượng nghe hoặc cải thiện WER.

- vi->en: 1 -> 1 đoạn; 11.55 -> 11.54 giây (một lượt, không phải benchmark ổn định).
- vi->ko: 1 -> 1 đoạn; 6.66 -> 7.04 giây (một lượt, không phải benchmark ổn định).
- vi->zh: 1 -> 1 đoạn; 11.14 -> 12.24 giây (một lượt, không phải benchmark ổn định).
- en->vi: 3 -> 7 đoạn; 6.21 -> 11.89 giây (một lượt, không phải benchmark ổn định).
- ko->vi: 3 -> 6 đoạn; 4.40 -> 8.06 giây (một lượt, không phải benchmark ổn định).
- zh->vi: 2 -> 5 đoạn; 4.12 -> 6.89 giây (một lượt, không phải benchmark ổn định).

## Những gì chưa được sửa
- Chuỗi giờ/ngày/điện thoại VI vẫn bị ASR/CAPU tách sai, MT vẫn có thể đổi nghĩa. Không có bằng chứng mọi đầu ra tốt hơn.
- VAD tách nhiều đoạn hơn ở EN/KO/ZH trong tập nhỏ và làm tăng latency; cần test holdout trước khi dùng làm cấu hình tối ưu.
- Bộ grammar viết hoa/dấu câu upstream dùng quy tắc ngôn ngữ và có danh sách tên; không phải model ASR mới, không đảm bảo đúng mọi văn bản.
- Không có 3 WAV VE-0371/VK-0755/VM-4870 tại đường dẫn upstream để tái lập báo cáo 3 clip. Chỉ kiểm kỳ vọng văn bản và fixture hiện có.
- Với S2S_NEURAL_TN=1 (web đang dùng), Qwen vẫn đảm nhiệm nguồn và bỏ qua restore_vi. VAD/ORT mới vẫn áp dụng. Đặt S2S_NEURAL_TN=0 khi cần dùng đúng đường CAPU + quy tắc upstream. Không tự đổi lựa chọn này của người dùng.

## Bằng chứng và chạy lại
- `results/asr_integration/{before,after}/summary.json` và WAV cùng thư mục.
- `logs/asr_integration_{before,after,tests}.log`; `logs/asr_integration_tests_upstream_failure.log` giữ lỗi đầu tiên.
- `reports/asr_integration/verification.json`; snapshot rollback `before_source/`.
- Lệnh: `/private/tmp/s2s-original-env/bin/python -m unittest discover -s tests -v`.
- Chạy lại sau tích hợp: `/private/tmp/s2s-original-env/bin/python tools/evaluate_asr_integration.py after` (ghi đè kết quả after).
- Rollback chỉ 4 file từ before_source, không reset repo. Chưa push thay đổi tích hợp lên GitHub.
