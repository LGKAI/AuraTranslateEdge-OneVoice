# Spec cho vòng cải thiện kế tiếp

Executor là Antigravity. Codex lập kế hoạch/review; người dùng nghe xác nhận audio. Workspace duy nhất: OneVoice_E2E_HybridTTS. Không tạo bản sao pipeline mới hoặc ghi đè thay đổi chưa commit.

## Mục tiêu

1. Chặn regression ranh giới mã và có bằng chứng tái lập được.
2. Format theo lớp dữ liệu/ngữ cảnh, không thay riêng một chuỗi hoặc thêm số bị mất.
3. Bảo toàn những entity nguồn xác định được; tách rõ unknown, mismatch và đã xác minh.

## Ràng buộc chung

- Giữ graph/weights và dataset cố định trong một so sánh; đổi một biến mỗi ablation.
- VI/EN/KO/ZH; Python 3.11 hiện được dùng. Không thêm dependency/model lớn nếu chưa có kế hoạch riêng.
- Không lấy transcript tham chiếu thay đầu vào ASR; không tự suy ra 32 khi ASR trả 2.
- Mã giữ chữ số, thứ tự, số 0 đầu và dấu theo policy đã công bố. Canonical bỏ grouping phải có mapping với raw, không gọi đó là byte-identical.
- Dấu âm, thập phân, loại dữ liệu, đơn vị và meridiem là ngữ nghĩa; chỉ giữ dãy chữ số không đủ.
- Scientific/range chưa hỗ trợ có thể bảo toàn ký tự kèm warning; không tách thành số khác rồi báo thành công.
- Unknown/unverifiable không tính pass và không được lặng lẽ sửa output MT.
- Ngày mơ hồ không tự chọn DMY/MDY. Không xem mọi chuỗi dài là phone hoặc mọi dấu gạch nối là âm.
- Không đưa placeholder tùy ý qua NLLB pruned vocab rồi giả định model giữ nó.
- Nếu span vượt graph budget, trả lỗi/unsupported rõ, không cắt âm thầm hoặc nuốt exception.
- Log theo run ID, input hash, source/model/config fingerprint, seed/locale, phiên bản thư viện, thiết bị; không publish dữ liệu bộ đề/thu âm cá nhân.

## Tiêu chí đề xuất cho vòng này

- Regression đã xác định có expected rõ phải qua toàn bộ; không phát sinh silent content loss trên tập được đánh giá.
- Bộ mutation có nhãn phải bắt 100% mutation xác định được; đây là mục tiêu test, không tuyên bố accuracy thế giới thực.
- Luôn báo số mẫu và unknown coverage; không có golden transcript người thật thì không công bố WER/MOS.
- Default mới phải giữ đúng ZIP ở ca regression và không làm giảm metric numeric/identifier so baseline trên holdout. Nếu kết quả hòa, ưu tiên cấu hình nhẹ hơn.
- Latency chỉ chặn promotion khi tăng >10% warm median trên phép so paired cùng thiết bị mà không có cải thiện numeric được xác nhận. Đây là ngưỡng đề xuất của kế hoạch, không phải yêu cầu sản phẩm đã được đo. Báo cả cold, stage time, số đoạn; p95 tập nhỏ chỉ mang tính mô tả.

## Chưa thuộc phạm vi

Huấn luyện/đổi weights ASR, ITN VI đầy đủ, bảo đảm mọi định dạng trên thế giới, tự động sửa bản dịch không có alignment đáng tin, chứng nhận audio bằng back-ASR hoặc tối ưu NPU khi chưa có thiết bị. Các việc này cần kế hoạch riêng sau bằng chứng vòng này.
