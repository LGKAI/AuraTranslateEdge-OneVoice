# Workflow Codex + Antigravity

## Vai trò theo lựa chọn của người dùng

- **Codex:** đọc code/diff/tài liệu/log, phân tích nguyên nhân, lên kế hoạch và review kết quả. Được viết tài liệu bàn giao; không tự sửa production code, chạy test/benchmark/inference, đổi model hoặc push khi người dùng chỉ giao lập kế hoạch/review.
- **Antigravity:** thực hiện thay đổi và chạy test theo kế hoạch; ghi lại bằng chứng, khác biệt so với kế hoạch và các lỗi còn lại.
- **Người dùng:** quyết định mục tiêu, điều chỉnh phạm vi; nghe audio để xác nhận phát âm/ngữ điệu khi cần. Một yêu cầu rõ ràng giao lại phần thực thi cho Codex có thể thay đổi phân công trên.

Skill đã cài trên Codex: `writing-plans`, `systematic-debugging`, `verification-before-completion` từ obra/superpowers. Phân công của người dùng được ưu tiên hơn các gợi ý mặc định trong skill về tự thực thi hoặc dùng subagent. Skill này không tự cài vào Antigravity và không tạo kết nối giữa hai ứng dụng.

## Vòng làm việc

1. Người dùng giao mục tiêu cho Codex.
2. Codex đọc code và ghi kế hoạch tại `docs/workflow/<task-id>/PLAN.md`, không sửa runtime.
3. Người dùng đưa kế hoạch cho Antigravity; Antigravity thực thi và ghi `EXECUTION_REPORT.md` cùng thư mục.
4. Codex đọc diff và báo cáo, ghi `REVIEW.md`: đạt yêu cầu nào, lỗi nào còn lại, bằng chứng nào thiếu. Không chỉ dựa vào câu “đã xong”.
5. Antigravity sửa các điểm cần thiết rồi gửi lại báo cáo; người dùng quyết định bước tiếp theo. Commit/push/merge theo phạm vi được giao, không suy ra tự động từ việc review.

Mỗi thời điểm chỉ một bên sửa production code. Dùng commit SHA/branch để bảo đảm cả hai đang xét cùng phiên bản; không tạo thêm bản sao pipeline chỉ để bàn giao. Các ứng dụng không tự đọc được chat nội bộ của nhau: file và Git là nguồn bàn giao chung.

## Mẫu PLAN.md - Codex viết

```markdown
# <task-id>: <mục tiêu>

## Phiên bản và phạm vi
- Repo/branch/base commit:
- Thay đổi đang có trước nhiệm vụ (git status/diff):
- Mục tiêu và hành vi mong muốn:
- Ngoài phạm vi:
- Giới hạn về model, dependency, latency và thiết bị:

## Căn cứ từ code
- File/hàm/đường đi dữ liệu liên quan:
- Bằng chứng lỗi; giả thuyết nào chưa được kiểm chứng:

## Công việc thực thi
- [ ] T1: thay đổi, file, interface và tiêu chí nghiệm thu cụ thể.
- [ ] T2: ca kiểm tra hồi quy, dữ liệu đối chứng và kết quả kỳ vọng.
- [ ] T3: kiểm tra tích hợp cần thiết, cách ghi trace và cách quay lại.

## Lệnh test do Antigravity chạy
- Command + thư mục chạy + môi trường/thiết bị:
- Kết quả kỳ vọng và điều kiện thất bại:
- Đường dẫn lưu log/JSON/audio:
- Những điều test này không chứng minh được:

## Bàn giao
- File được phép sửa; thay đổi cần báo lại:
- Những câu hỏi hoặc bằng chứng còn thiếu:
```

Kế hoạch phải cụ thể nhưng không viết thay toàn bộ implementation. Chỉ yêu cầu test phù hợp với thay đổi; không chạy lại benchmark lớn nếu kết quả có sẵn vẫn hợp lệ và chưa có lý do mới.

## Mẫu EXECUTION_REPORT.md - Antigravity viết

```markdown
# <task-id>: Kết quả thực thi
- Branch/base commit/head commit:
- git diff --stat; danh sách file đã sửa:
- Thay đổi khác kế hoạch và lý do:

## Bằng chứng test
- Command thực tế + cwd + thời gian chạy:
- Phiên bản dependency/model, cấu hình và thiết bị:
- Exit code, số ca pass/fail/skip:
- Đường dẫn raw log, JSON và audio:
- Kết quả trước/sau; test chưa chạy và lý do:

## Còn lại
- Lỗi chưa sửa, giới hạn, rủi ro cụ thể:
- Cách rollback:
```

Với S2S/TTS, báo cáo thêm: input WAV/hash, ASR thô, bản dịch, text thực sự vào TTS, chunk và output WAV/hash. Phân biệt test riêng format, TTS trực tiếp, S2S thật và mic thật. ASR nghe lại audio không phải điểm nghe của người và không chứng minh ngữ điệu/phát âm hoàn hảo.

## Quy tắc review bằng chứng

- Ghi rõ **“đã đọc code”**, **“theo log Antigravity”**, **“người dùng đã nghe xác nhận”** hoặc **“chưa xác minh”**.
- Nếu chỉ có log từ Antigravity, không nói Codex đã tự chạy test.
- Log phải khớp commit/config được review; log cũ không chứng minh phiên bản mới.
- Không có raw log hoặc thiếu test quan trọng: ghi thiếu bằng chứng và yêu cầu bổ sung vào bước thực thi, không tự đánh dấu pass.
- Không để plan và report tự thay đổi mục tiêu để hợp thức hóa kết quả.

## Prompt bàn giao nhanh

**Cho Codex:** “Đọc code và lập kế hoạch cho <mục tiêu>. Dùng workflow trong docs/workflow/README.md. Chỉ lập kế hoạch, không sửa runtime hoặc chạy test.”

**Cho Antigravity:** “Đọc docs/workflow/README.md và docs/workflow/<task-id>/PLAN.md. Thực hiện trong phạm vi kế hoạch, chạy test phù hợp và lưu EXECUTION_REPORT.md cùng raw log. Báo rõ mọi thay đổi ngoài kế hoạch.”

**Cho Codex review:** “Đọc PLAN.md, EXECUTION_REPORT.md và diff của <commit>. Review theo tiêu chí nghiệm thu; phân biệt lỗi code với thiếu bằng chứng. Không tự thực thi.”
