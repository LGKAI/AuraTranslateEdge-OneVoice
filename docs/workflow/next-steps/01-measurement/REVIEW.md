# Review Plan 01 - cần sửa trước nghiệm thu

Ngày 13/11/2026. Codex và hai agent `review_profiles_done`, `review_measurement_done` review read-only code, diff, execution report, log và JSON. Không chạy lại test/inference, không đổi runtime hoặc server.

## Kết luận

**Chưa nghiệm thu hoàn toàn Plan 01.** Implementation đã tiến bộ rõ về profile, telemetry, tách ablation và bằng chứng HTTP, nhưng còn 5 vấn đề P1. Không cần làm lại toàn bộ phần đã đúng; sửa từng lỗi và bổ sung regression tương ứng trước khi chuyển sang Plan 02. Chưa đề nghị đổi ORT default sang basic từ một máy/tập nhỏ.

## Finding

### R1 / P1 - TTS profile được báo nhưng không bind vào instance

`s2s_edge/s2s_edge/registry.py:47-68,100-104,155-158` nhận `tts_format`, ghi vào profile nhưng `Registry.tts()` không truyền lựa chọn đó cho backend. `tts_format.py:9-17` tiếp tục đọc biến môi trường lúc synthesize.

**Ảnh hưởng theo code:** `Registry(..., tts_format='legacy')` trong env hybrid có thể báo legacy nhưng tạo đầu vào bằng hybrid. Những harness tự sửa env trước khi tạo Registry có thể che lỗi này. Chưa chạy tái lập trong review.

**Cần sửa:** bind effective profile vào instance TTS/Engine, tránh sửa env toàn cục để chữa. Đồng thời thống nhất source_mode snapshot: Engine không nên chuyển nhánh theo env về sau trong khi metadata vẫn là profile cũ.

**Regression cần:** explicit argument khác env; hai Registry khác mode sống đồng thời; env đổi sau khởi tạo. Kiểm text/trace thực và đúng instance, không chỉ profile dict.

### R2 / P1 - Launcher tái sử dụng server cũ mà không so profile/code

`tools/start_s2s_web.py:29-31,42-44` chỉ kiểm health.ok. Server đã chạy thì build_launch_env và toàn bộ kiểm tra caller bị bỏ qua.

**Ảnh hưởng:** gọi launcher với mode=0/legacy trong khi server cũ đang mode=1/upstream chỉ trả URL, không áp dụng hoặc báo mismatch. Hai server test ở 8767/8768 chưa chứng minh cổng 8766 hiện có được chuyển sang cấu hình mới. Review không kiểm tra trạng thái live của server 8766.

**Cần sửa:** so effective profile và loaded-code fingerprint của server với yêu cầu caller. Khớp thì reuse; lệch thì trả trạng thái/exit khác 0 rõ ràng. Chỉ reload khi được phép và không có job, không tự tắt server bận. Phân biệt code fingerprint đã load với file trên đĩa.

**Regression cần:** existing server matching; profile mismatch; stale code; pending job >0; env không hợp lệ ngay cả khi server đang healthy.

### R3 / P1 - Ablation không dùng hợp đồng immutable/provenance

`tools/ablate_profiles.py:145-147,243-245` dùng exist_ok=True; `:235,:328` ghi JSON vào đường cố định, WAV cũng dùng lại tên. Kết quả không có source/model/manifest fingerprint và drift check tương đương harness mới.

**Ảnh hưởng:** chạy lại các lệnh trong báo cáo có thể ghi đè chính evidence dùng chọn profile. Không thể suy provenance của ablation từ manifest của một run S2S khác.

**Cần sửa:** run ID riêng, từ chối overwrite, manifest của đúng ablation và kiểm candidate drift. Không gắn hash hiện tại vào kết quả lịch sử như thể đã ghi lúc chạy. Giữ nguyên kết quả cũ.

**Regression cần:** existing run bị từ chối; source/config drift bị đánh dấu failed; run manifest chứa input/model/profile/source và seeded order. Chỉ cần smoke nhỏ cho contract trước; đo lại tập cần thiết khi chuẩn bị kết luận profile.

### R4 / P1 - Baseline verify=False chưa làm run thất bại

`tools/evaluate_asr_integration.py:254-274` trả verified=False khi thiếu/tamper baseline. `:303-310` chỉ đưa vào manifest; điều kiện run_status/exit cuối chỉ xét case errors và source drift.

**Ảnh hưởng theo code:** một comparative run có baseline không hợp lệ vẫn có thể exit 0 và run_status ok. Không thấy bằng chứng baseline thực hiện tại đã bị sửa; đây là lỗi enforcement.

**Cần sửa:** run đã khai báo reference phải kiểm baseline trước inference và fail khi thiếu/mismatch/không có evidence cần thiết. Nếu muốn standalone thì dùng mode explicit và không gọi nó là so sánh baseline đã xác minh.

**Regression cần:** tamper/missing baseline → exit khác 0, engine không được gọi; valid baseline → đi tiếp; standalone được gắn nhãn không comparative.

### R5 / P1 - Lexical substring không phải numeric/identifier correctness

`tools/ablate_profiles.py:64-81` tìm expected token dưới dạng substring. Theo code, expected 32 có thể match 132, hoặc 3 match 330; không kiểm type/unit/meridiem. Expected list cũng chưa bao phủ mọi entity trong câu.

**Ảnh hưởng:** checked_entity_passes chỉ là lexical-presence proxy. Câu “legacy thắng tuyệt đối theo SPEC entity gate” trong EXECUTION_REPORT quá mạnh. Đây không phủ nhận finding ZIP bị tách của upstream, vốn có đối chiếu segment/raw ASR rõ.

**Cần sửa tối thiểu:** đổi tên metric và hạ kết luận ngay trong báo cáo; định nghĩa coverage và correct/wrong/unknown. Không dùng proxy làm gate semantic. Nếu triển khai scorer entity thì dùng type/boundary/identity, xử lý duplicate/missing entities và không chấm unknown thành pass.

**Regression cần:** 32 so 132, 3 so 330, 0 đầu, sign/decimal/unit, hai entity cùng giá trị và missing flight/date/time label. Lời đọc chưa parse được phải unknown, không buộc pass/fail để có số đẹp.

## Những phần đã có bằng chứng tốt

- Log `results/next_steps/logs/t5_final_unit_tests.log` ghi 38 test methods OK; reviewer không tự chạy lại.
- Hai summary `plan01_top1_legacy_basic` và `plan01_top2_legacy_disable` đều ghi 6 completed/0 failure/run_status ok, VAD legacy.
- Reviewer so hash các source trong manifest top1 với file hiện tại: không thấy mismatch. Đây là so file, không phải chạy inference hoặc kiểm hash lại nhiều GB model.
- Đã tách native EN khỏi synthetic; thiếu native VI/KO/ZH được báo rõ. Không lấy synthetic làm bằng chứng native holdout.
- Quyết định giữ VAD legacy mặc định phù hợp bằng chứng ZIP và latency hiện có. ORT auto vẫn giữ baseline, basic mới là ứng viên được đo.

## Bàn giao sửa review

1. Sửa R1/R2 và bổ sung tests về actual effective mode/reuse server. Không đổi production config chỉ để làm test pass.
2. Sửa R3/R4, dùng engine mock cho contract; giữ nguyên tất cả evidence lịch sử. Sau đó chạy một ablation smoke có manifest mới.
3. Sửa nhãn/kết luận R5; nếu chưa có semantic scorer, giữ kết luận giới hạn ở proxy và các regression được xác nhận cụ thể.
4. Antigravity chạy targeted + regression và cập nhật EXECUTION_REPORT cùng diff/hash. HTTP thật chỉ chạy cổng cô lập, không restart cổng người dùng để phục vụ review.
5. Gửi lại cho Codex. Chưa push/merge hoặc coi Plan 01 hoàn tất nghiệm thu từ review này.

Các lỗi formatter và source preservation trong Plan 02/03 vẫn giữ nguyên scope, không yêu cầu sửa lẫn vào patch đóng 5 finding này.
