# Kế hoạch và tiêu chí nghiệm thu chuẩn hóa TTS hybrid

Phạm vi: VI/EN/KO/ZH, số nguyên/thập phân có dấu, %, tiền, nhiệt độ/đơn vị thông dụng, giờ, ngày, mã chữ-số/ZIP/điện thoại. Không sửa nghĩa ASR/MT, không chấm tên riêng hoặc MOS.

1. Lưu baseline hiện có; dùng bộ parser chung tạo span có kiểu và vị trí, giữ nguyên chuỗi gốc và cấu trúc giá trị. Không thay riêng chuỗi test.
2. Xác định loại bằng grammar và ngữ cảnh theo ngôn ngữ; CRF sẵn có chỉ hỗ trợ phân loại phần còn mơ hồ ở EN/VI. Model không sinh lại số hay toàn câu. Chuỗi mơ hồ/không hợp lệ phải có cảnh báo và chính sách đọc bảo toàn, không đoán âm thầm.
3. Sinh lời đọc từ giá trị có cấu trúc và tài nguyên ngôn ngữ. Giờ Hàn dùng số thuần Hàn; năm Trung đọc từng số; điện thoại/mã giữ từng ký tự và số 0 đầu; tiền có đơn vị.
4. Bộ kiểm tra kiểm soát span không chồng lấn, raw đúng input, giá trị/chuỗi ký tự được bảo toàn trong biểu diễn trung gian, có trace. Không dùng điều kiện “không còn chữ số” làm bằng chứng bảo toàn nghĩa.
5. Tích hợp trước G2P/TTS cho cả 4 ngôn ngữ; giữ tùy chọn legacy/neural để so sánh. Chia đoạn theo ranh giới từ/Unicode hợp lệ, không cắt giữa từ; bảo vệ cụm đã chuẩn hóa khi còn đủ cửa sổ graph; cụm quá dài tách ở ranh giới an toàn và ghi nhận giới hạn.
6. Kiểm thử độc lập kỳ vọng lời đọc, đối chứng số lượng/mã, biên ngày/giờ, separator theo locale, số 0, số âm và nhiều format trong cùng câu. Sinh nhiều giá trị ngẫu nhiên để kiểm tra tính tổng quát; không train theo các ca đánh giá.
7. Replay tất cả 50 bản dịch en-vi (không đổi file gốc), chạy TTS thực 4 ngôn ngữ, chạy lại S2S trên audio gốc ca lỗi và audio VI->EN/KO/ZH. Lưu trace, WAV, thời gian, hash và nhận xét trước/sau. Phân biệt kiểm tra format, chạy thành công và chất lượng nghe.

Tiêu chí: các lỗi 5:30/08:00/ZIP/734 cũ không tái diễn; câu tổng hợp EN không phụ thuộc thành công LLM; không thêm lỗi giá trị vào 50 câu; mọi lỗi còn lại được báo cáo. Không cam kết đúng mọi văn bản hoặc latency thiết bị NPU; không dùng benchmark phát triển làm độ chính xác sản xuất.

Tham khảo thiết kế: https://github.com/NVIDIA/NeMo-text-processing/blob/main/nemo_text_processing/hybrid/README.md ; https://unicode-org.github.io/icu/userguide/format_parse/numbers/rbnf.html . Đây là triển khai grammar/codec riêng trong dự án, không tuyên bố đã tích hợp toàn bộ NeMo/ICU.

## Trạng thái hoàn thành

Đã thực hiện cả 7 bước. 10 nhóm unittest qua, 55 ví dụ kỳ vọng qua, 1.600 ca cấu trúc sinh tự động qua; 49 audio replay MT, 4 audio TTS và 6 audio S2S đã lưu. Có thêm HTTP audio thật và kiểm tra cùng Translator xử lý ca hợp lệ rồi ca rỗng để tránh log TTS cũ. Báo cáo: `results/hybrid_format/README.md`. Đánh giá audio còn sai khác qua ASR nghe lại, chưa có chấm nghe của người; không đánh dấu chất lượng phát âm là hoàn hảo.
