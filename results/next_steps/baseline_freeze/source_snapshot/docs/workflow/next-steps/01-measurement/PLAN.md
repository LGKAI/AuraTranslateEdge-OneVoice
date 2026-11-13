# P0: profile an toàn và phép đo có thể tái lập

**Executor:** Antigravity; chỉ task ở đây. **Spec:** `../SPEC.md`. **Goal:** chặn regression VAD, đo độc lập từng thay đổi. **Architecture:** explicit profile → effective config → immutable run → paired evaluation. **Tech:** Python/ONNX hiện có.

## T1 - Freeze và sửa hợp đồng harness

**Files:** `tools/evaluate_asr_integration.py`; thêm `tests/test_asr_evaluation_contract.py`. Evidence mới dưới `results/next_steps/<run-id>/`, không sửa baseline before/after đã có.
**Interface:** CLI nhận `--run-id`, `--manifest`, `--source-mode {0,1}`, `--vad-profile {legacy,upstream}`, `--ort-policy {basic,disable,auto}`; cấu hình thực tế ghi vào result, không ép env âm thầm. `write_run_manifest(...) -> dict` gồm input/model/source hashes, code diff fingerprint, effective options và environment. Không cần hash nhiều GB weights lại cho mỗi case: cache theo manifest đã xác thực, invalidation khi file đổi.

- [ ] Test fail: run ID đã có phải từ chối ghi đè; label before không được nhận là code cũ; có case error thì process exit khác 0 và vẫn giữ partial result.
- [ ] Freeze working tree đang có bằng manifest và patch riêng; không commit/stash/reset thay đổi người dùng. Tách baseline đối chứng bất biến (input/model/reference-run fingerprint) khỏi manifest của từng candidate. Candidate được phép khác source baseline theo diff đã ghi; chỉ dừng nếu khác expected candidate revision hoặc source/config drift trong cùng run. Không thay baseline bằng candidate để hợp thức hóa kết quả.
- [ ] Implement CLI/manifest/error exit; cập nhật metadata mọi ca gồm load, warm-up, stages, số đoạn và failure count. Chạy unit hợp đồng bằng `python -m unittest discover -s tests -p test_asr_evaluation_contract.py -v`.
- [ ] Rollback phải kiểm expected source hash trước khi áp patch, bao gồm config/docs đã đổi; không copy bốn snapshot mù.

**Đạt:** evidence phân biệt code cũ/mới thật và không thể báo pass khi ca lỗi. Thay đổi task này không yêu cầu chạy inference.

## T2 - Profile VAD và ORT có thể chọn độc lập

**Files:** `s2s_edge/s2s_edge/audio.py`, `backends.py`, `pipeline.py`, `conversation.py` (call sites), `tools/run_e2e.py`, `tools/s2s_web.py`, `tools/start_s2s_web.py`; thêm `tests/test_runtime_profiles.py`.
**Interface:** `get_vad_options(profile: str) -> dict`; không đổi giá trị caller truyền rõ. Legacy: pad=150 ms, silence=500 ms, merge_gap=1.5 s, short=3.0 s; upstream: 120/320/0.32/1.1. Default an toàn tạm thời legacy; upstream opt-in. ORT `basic` vẫn có retry disable khi optimizer lỗi; `auto` tương ứng chính sách upstream >200 MB/external-data. Ghi policy và attempt thực, không chỉ ý định cấu hình.

- [ ] Regression trước: effective profile giống nhau ở CLI/web và batch/stream; launcher phải tôn trọng env caller đặt 0, không ép source mode=1.
- [ ] Profile đi qua cùng một chỗ resolve; không để một số call site vẫn dùng defaults khác. Health/result ghi profile, source mode và fingerprint, không thay nhãn UI bằng phỏng đoán.
- [ ] Test caller override, invalid profile báo rõ, ORT fallback và không tự sửa option đã chọn. `python -m unittest discover -s tests -p test_runtime_profiles.py -v`.
- [ ] Trên cùng WAV regression EN, VAD legacy phải giữ span ZIP đầy đủ; thêm ca pause nằm giữa chữ số, sát câu và chuỗi dài. Đánh giá kết quả ASR thực, không chỉ assert số đoạn. Không thêm heuristic cho giá trị 67297.

**Đạt:** chưa promote VAD upstream mặc định; có thể tái lập mỗi nhánh mà không sửa source thủ công. Khi chỉ containment VAD, giữ ORT policy hiện tại và ghi nó; basic/disable/auto là các biến ablation riêng, không đổi đồng thời rồi quy nguyên nhân cho VAD. Không tự đổi lựa chọn neural-source của người dùng; chỉ bỏ việc launcher ghi đè và ghi nhận nó.

## T3 - Ablation và quyết định

- [ ] Chọn fixture có nhãn từ dữ liệu sẵn có và freeze split; tách rõ audio tổng hợp với audio người thật. Dev minimum 3 clip/language nếu có; thiếu dữ liệu ghi thiếu, không lấy giọng tổng hợp gọi là native holdout.
- [ ] Đo VAD legacy/upstream với ORT cố định; đo ORT basic/disable/auto trên graph liên quan khi VAD cố định. Chỉ hai cấu hình tốt nhất mới chạy lại S2S sáu chiều; tránh toàn bộ factorial đắt tiền khi chưa cần.
- [ ] Load/cold tách warm; ít nhất 1 warm-up và 3 lượt paired, đảo thứ tự profile bằng seed. Báo stage time/segment count, median và sample count; p95 tập nhỏ chỉ mô tả. Giữ cùng graph/threads/device và không chạy hai inference job cùng lúc.
- [ ] Test HTTP thật ít nhất VI→EN và EN→VI với mode mà web sẽ dùng; source mode còn lại có integration check riêng, không dùng kết quả thay thế nhau. Metal không có thì đánh dấu mode=1 chưa kiểm, không silently chuyển 0.
- [ ] Áp gate SPEC: không regression mã ở fixture đã xác nhận; chỉ promote upstream nếu có lợi numeric/quality được xác minh và chấp nhận latency. Nếu không, giữ legacy và báo kết luận không tăng chất lượng.

**Output:** `EXECUTION_REPORT.md`, manifest, unit log, ablation JSON và sample WAV. Trạng thái audio nghe chưa xác nhận phải để riêng. Mỗi task có diff độc lập, không tự push/merge.
