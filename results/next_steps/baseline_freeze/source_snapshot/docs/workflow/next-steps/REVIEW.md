# Review độc lập kế hoạch tích hợp và backlog

Ngày 13/11/2026. Ba agent: `audit_asr_plan`, `plan_numeric_pipeline`, `audit_format_backlog`; Codex đối chiếu các finding chính với code/JSON. Tất cả làm read-only, không chạy test/inference và không sửa runtime trong lượt review này.

## Kết luận

Kế hoạch cũ đạt mục tiêu **tích hợp có kiểm soát và chạy được**, chưa đạt mục tiêu **chứng minh nâng chất lượng và chọn mặc định an toàn**. Không nên lấy 22 unittest methods hoặc 6 ca chạy hết làm chứng nhận ASR/numeric accuracy/MOS. Các thay đổi ASR vẫn chưa commit; HEAD không đại diện toàn bộ code đang chạy.

## Finding ưu tiên

1. **P0 - Regression ZIP có bằng chứng audio-run JSON.** `results/asr_integration/after/summary.json`, ca en→vi: đoạn `[16.58,18.48]` nhận `And my ZIP code is6.`, đoạn `[18.6,19.94]` nhận `729,7.`; bản before giữ nguyên `67297` trong một đoạn. Cấu hình `audio.py:51` tách 3→7 đoạn. Đây là lỗi tính toàn vẹn mã, không chỉ tăng thời gian. Cần profile VAD opt-in và baseline an toàn trước khi promote.
2. **P0 - Guard neural chưa bảo vệ ngữ nghĩa dấu số.** `neural_textnorm.py:66-73` bỏ `. : / -` trong phép so content. Suy luận từ code: biến đổi `15.50→15:50`, `-12→12` có thể lọt. Chưa chạy các mutation này trong review; Antigravity phải tái lập bằng test. Cần kiểm cả canonical_identifiers trước guard.
3. **P0 - Có đường mất từ và phá numeric span trước MT.** `punct_vi.py:197-199` nuốt exception merge; suy luận có thể bỏ suffix. `mt.py:9-10,33-39,56-59` chia dấu chấm/phẩy/hai chấm ngay trong số khi source mode=0; báo cáo Antigravity đã ghi `$15.50`/`37.5` bị tách. Không vá mù mọi `37. 5` ở output MT.
4. **P0 - Bốn nhóm lỗi formatter vẫn hiện diện.** `hybrid_textnorm.py:21,166-172,290,295-309`: phone quốc tế/ngoặc/danh sách; fullwidth comma CJK ghép số; Unicode boundary khiến scientific/range không được cảnh báo; AM/PM nuốt dấu chấm câu. Bổ sung cue đơn thuần không giải quyết gốc. Bản sửa AM/PM đã có ở `origin/feat/tts-hybrid-format`, cần port diff nhỏ và test, không copy nguyên module khác layout.
5. **P1 - Test không khớp đường web.** Harness ép neural source=0; launcher `tools/start_s2s_web.py:10` ép 1 kể cả khi caller chọn 0. `conversation.py:78-82` mode=1 bỏ qua restore_vi mới. Cần explicit profile và log effective config, không tuyên bố đã test web bằng kết quả CLI mode=0.
6. **P1 - Chưa cô lập nguyên nhân latency.** Thay đồng thời VAD và ORT; `backends.py:31-37` ảnh hưởng graph MT/TTS lớn, không chỉ ASR. Cùng engine, thứ tự cố định và một lượt không cho phép quy hết chậm đi cho VAD. Thực hiện ablation độc lập, ghi load/warm-up và từng stage.
7. **P1 - Baseline/provenance/rollback yếu.** `evaluate_asr_integration.py:12-26`: nhãn before không chọn code cũ, có thể ghi đè baseline và vẫn exit 0 khi có case error. Kế hoạch rollback bốn module không bao hết metadata/harness; copy mù có thể đè sửa mới. Cần immutable run ID, source/config/model fingerprint, exit status và rollback patch theo hash.
8. **P1 - Trace chưa chắc là chunk cuối cùng vào graph.** `tts_format.py:70-79` retry với record=None; `tts.py` lấy lengths từ planned_chunks. G2P sau TN và str.find lại span cũng có thể làm lệch vị trí bảo vệ khi có cụm lặp. Cần phân biệt planned/attempted/synthesized và provenance ranges.

## Cách đọc bằng chứng

- Log hiện có ghi 22 test methods OK; reviewer không tự chạy lại. VAD sinusoid và CAPU mock chỉ kiểm hợp đồng hẹp, không phải audio người thật.
- Before/after có cùng input hash nhưng KO/ZH audio là output tổng hợp từ pipeline cũ; chưa phải holdout người bản ngữ.
- Latency EN→VI 6,21→11,89 s; KO→VI 4,40→8,06 s; ZH→VI 4,12→6,89 s chỉ là số một lượt, không phải benchmark có kiểm soát.
- Chưa có 3 WAV upstream để tái lập báo cáo 3 clip. Không tuyên bố đã tái lập nó.

## Quyết định kế hoạch

Ưu tiên containment/provenance → sửa P0 format và P0 source guard/segmentation theo hai kế hoạch độc lập → audit numeric MT và nghe xác nhận → đánh giá holdout rồi mới chọn default. Xem `PLAN.md` và `SPEC.md` cùng thư mục. Review này chưa thay đổi cấu hình đang chạy.

## Review lại kế hoạch mới

Hai agent đọc lại roadmap và các kế hoạch 01/03. Đã chỉnh theo phản biện: tách baseline bất biến khỏi candidate manifest để không vô tình thay đối chứng; opaque numeric span phải giữ raw thay vì đổi dấu rồi chỉ warning; chốt nghĩa GuardReport.valid và CAPU.last_validation. Agent không nêu blocker scope lớn khác. Đây là review tài liệu, không phải xác nhận implementation/test của các task chưa thực hiện.
