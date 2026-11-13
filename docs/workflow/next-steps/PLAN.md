# Roadmap sau review ASR và format

**Executor:** Antigravity theo phân công của người dùng; Codex chỉ plan/review. Không yêu cầu tự chuyển sang Codex native/subagent execution.

**Goal:** giảm lỗi số/mã có bằng chứng, không đánh đổi bằng thay đổi mặc định chưa kiểm chứng.
**Architecture:** profile có thể đối chứng, formatter có span/giá trị, guard và splitter bảo toàn trước MT, audit giá trị qua từng chặng.
**Tech stack:** runtime Python hiện có, ONNX CPU, CRF và neural source mode tùy cấu hình; chưa đổi weights.
**Spec:** `SPEC.md`. **Review:** `REVIEW.md`. **Baseline:** `baseline_manifest.json`; HEAD chỉ là nền Git, working tree còn thay đổi chưa commit.

## Thứ tự triển khai

1. **P0 - An toàn và đo đúng:** `01-measurement/PLAN.md`. Đóng băng baseline, VAD mới opt-in, profile web/CLI rõ ràng; cô lập VAD với ORT. Hoàn tất trước khi đổi default hoặc so kết quả các bước sau.
2. **P0 - Formatter:** `02-format/PLAN.md`. Sửa bốn nhóm lỗi đã tái lập; đóng băng 17 edge cases và bộ đối chứng. Độc lập với source guard, không trộn sửa ASR/MT vào commit parser.
3. **P0 - Nguồn trước MT:** `03-source-preservation/PLAN.md`. Guard ngữ nghĩa số, CAPU không mất từ, splitter không cắt numeric span; không khôi phục điều ASR chưa nhận ra.
4. **P1 - Audit và đánh giá người dùng:** ghi ledger trước/sau MT, test pipeline cùng profile web và nghe xác nhận. Chưa promote profile mới khi chỉ có unittest hoặc back-ASR.

Antigravity làm từng kế hoạch và lưu `EXECUTION_REPORT.md` cùng thư mục. Đưa báo cáo + diff cho Codex review trước khi mở rộng bước tiếp theo. Không tự push/merge hoặc restart server đang có job.

## Review focus xuyên suốt

- ZIP/phone nằm đúng cạnh khoảng dừng: không cắt 67297 thành 6 và 729,7 (01).
- Dấu câu CJK, dấu âm và phone thứ hai trong danh sách: không ghép/đổi giá trị (02).
- Model chỉ đổi dấu nhưng vẫn giữ chữ số: guard phải nhận ra ngữ nghĩa khác (03).
- Câu dài qua CAPU overlap và numeric span sát graph budget: không mất/trùng token (03).
- Cùng một server/profile nhưng reused cache hoặc audio cap retry: trace phải khớp code/config/chunk thực (01,02).

## Bàn giao kết quả

- Log lệnh đầy đủ và exit status; phân biệt unit, TTS graph, S2S, HTTP và mic thật.
- Mỗi ca lưu input/ASR raw/prepared/MT/tts_text/actual chunks/output + hash/metadata.
- Báo riêng correct/wrong/unknown entity theo language/type; phát âm cần người nghe, không quy ASR nghe lại thành gold.
- Không ghi đè `results/asr_integration/before`; rollback theo patch/hash, không reset cả repo.

## Gate kết thúc vòng

Chỉ xem vòng hoàn tất khi từng tiêu chí trong SPEC được đối chiếu, regression P0 không còn, cấu hình web đang dùng có test phù hợp, và hạn chế/chưa kiểm chứng được ghi rõ. Nếu thiếu audio holdout bản ngữ thì vẫn có thể hoàn tất implementation nhưng **chưa kết luận nâng chất lượng ASR hoặc promote toàn cục**.
