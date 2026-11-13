# Tích hợp asr-improvement vào pipeline hiện tại

Executor: Codex, theo yêu cầu tích hợp trực tiếp lần này; không dùng subagent. Báo cáo nguồn: reports/tts_format_s2s_eval_2026-11-13.md.

## Phạm vi và căn cứ

Nguồn: origin/asr-improvement, commit 1e5231886dbf4af5bb0a6ff722bfa16223f8f3a6. ASR weights không thay đổi. Nâng cấp nằm ở audio.segment, textnorm.vi_punct, conversation.restore_vi, backends.OrtGraph. Chọn lọc thay đổi, không merge nguyên branch có layout/history khác. Giữ TTS hybrid, chuẩn bị nguồn neural và mọi thay đổi chưa commit hiện tại.

## Tasks và tiêu chí

- [x] T1: lưu snapshot 4 module và hash của TTS/model registry; chạy baseline audio thực 6 chiều, dùng cùng graph và fixture. File mới: tools/evaluate_asr_integration.py; results/asr_integration/before/.
- [x] T2: đưa VAD và vi_punct từ nguồn vào; chỉ thêm vi_punct(m(text)) tại restore_vi, không thay Engine hoặc TTS. Đưa chính sách ORT fallback từ nguồn; ghi nhận nếu có ảnh hưởng latency. Giữ nguyên phần normalize() đọc số cũ.
- [x] T3: tests/test_asr_integration.py kiểm câu hỏi/ngắt câu, nội dung số không bị thay, tích hợp model-punctuation bằng mock và VAD cấu hình mới. Chạy chung với 15 test hiện có. Nếu thiếu 3 clip của báo cáo upstream, không gọi test fixture khác là tái lập báo cáo đó.
- [x] T4: chạy lại cùng 6 chiều, ghi before/after ASR/MT/tts_text/audio, thời gian, hash; so TTS và registry không đổi. File reports/asr_integration/REVIEW.md ghi cả hạn chế, không suy ra WER/MOS trên tập nhỏ.

## Verification

Python: /private/tmp/s2s-original-env/bin/python. Test: python -m unittest discover -s tests -v. E2E: python tools/evaluate_asr_integration.py before hoặc after, chạy CPU và S2S_NEURAL_TN=0 để thực sự đi qua restore_vi. Kiểm riêng S2S_NEURAL_TN=1 vẫn giữ nhánh nguồn có sẵn; không trộn kết quả hai cấu hình. Mỗi lần chạy phải lưu raw log + JSON + WAV.

## Rollback

Chỉ khôi phục 4 file từ reports/asr_integration/before_source; không reset/checkout toàn repo. Không commit/push remote trong nhiệm vụ fetch và tích hợp này. Runtime web đang chạy cần restart khi rảnh để nạp code mới; không mở Chrome.

Hoàn tất; xem reports/asr_integration/REVIEW.md cho hạn chế và so sánh thực tế.
