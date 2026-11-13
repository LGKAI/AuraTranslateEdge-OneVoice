# 01-measurement: Kết quả thực thi

- **Task ID / Plan:** `01-measurement` (`docs/workflow/next-steps/01-measurement/PLAN.md`) + sửa 5 P1 findings hậu review (`docs/workflow/next-steps/01-measurement/REVIEW.md`: R1–R5)
- **Branch / Base commit (HEAD) / Head commit:** Branch `feat/onevoice-e2e-tts-hybrid`, Base & Head commit `43ea359b42ae274f32f6e24eeb9b5018dc2a93ea` (baseline trước Plan 01 là uncommitted working tree được kiểm tra khớp 100% với `docs/workflow/next-steps/baseline_manifest.json` và đóng băng bất biến tại `results/next_steps/baseline_freeze/`).
- **`git diff --stat` (tracked files so với `HEAD`, gồm cả baseline pre-Plan-01 chưa commit):**
  `21 files changed, 762 insertions(+), 118 deletions(-)` (trong đó các file pre-Plan-01 `README.md`, `results/mic_web/server.pid`, `sources.json`, `s2s_edge/s2s_edge/textnorm.py`, `s2s_edge/tools/eval_vi_punct.py`, `tools/build_nllb_split.py`, `tools/check_mt_numerics.py`, `tools/train_identifier_tagger.py`, `tools/verify_neural_fix_run.py` được giữ nguyên, không chỉnh sửa trong Plan 01 hay vòng R1–R5).
- **Phạm vi R1–R5 (bổ sung & khắc phục theo `REVIEW.md`):**
  1. **R1 — Gắn `tts_format` và `source_mode` vào instance thay vì đọc `os.environ` tại thời điểm chạy (`tts_format.py`, `tts.py`, `registry.py`, `conversation.py`, `__main__.py`, `run_e2e.py`, `s2s_web.py`, `evaluate_asr_integration.py`):**
     - `s2s_edge/tts_format.py`: thêm `resolve_mode(tts_format=None)`, hỗ trợ `mode(tts_format=None)` và `prepare(text, lang, mode=None)`.
     - `s2s_edge/tts.py`: `TtsVi`, `TtsZh`, `TtsEnKo` bind `self.tts_format = _tts_format.resolve_mode(tts_format)` trong `__init__`; bỏ đọc `os.environ` trong `synthesize()`/`_synth()`, truyền `mode=self.tts_format` cho `_tts_format.prepare()`.
     - `s2s_edge/registry.py`: `Registry` lưu `self.tts_format` / `self.tts_format_explicit` và truyền `tts_format=self.tts_format` cho `Registry.tts(lang)`; mở rộng `runtime_code_fingerprint(extra_files=None)` thêm `server_files_sha256`.
     - `s2s_edge/conversation.py`: `Engine.__init__` snapshot `self.source_mode` từ `Registry`, `Engine.neural_enabled()` dùng `self.source_mode == "1"` thay vì đọc lại `os.environ` (chỉ fallback `neural_textnorm.enabled()` với fake registry thiếu thuộc tính).
     - `s2s_edge/__main__.py`, `tools/run_e2e.py`, `tools/evaluate_asr_integration.py`: thêm/truyền `--tts-format` vào `Registry(...)`; `evaluate_asr_integration.resolve_effective_options()` không mutate `os.environ` toàn cục nữa.
  2. **R2 — `tools/start_s2s_web.py` & `tools/s2s_web.py` kiểm tra profile + code fingerprint khi tái sử dụng server:**
     - `tools/s2s_web.py`: `Service` snapshot `self.profile`, nhận `--tts-format`, `Service.run` dùng profile đã snapshot; `/api/health` trả thêm `pid`, `effective_profile`, `loaded_code_fingerprint` (`s2s_edge_sha256`, `server_files_sha256`, `git_head`).
     - `tools/start_s2s_web.py`: `build_launch_env()` validate env & profile **trước** khi kiểm `/api/health`; `compare_server_state()` đối chiếu cả 4 trường profile (`source_mode`, `vad_profile`, `ort_policy`, `tts_format`) và `loaded_code_fingerprint` với đĩa (thiếu trường = mismatch); match → reuse (exit 0); mismatch/stale code → exit khác 0 (`2`) nếu không có `--reload`; có `--reload` → chỉ `SIGTERM` khi `pending == 0` và `pid` hợp lệ (từ chối exit `3` nếu `pending > 0` hoặc thiếu `pid`, không giết server đang bận).
  3. **R3 — `tools/ablate_profiles.py` tái lập, manifest, chống ghi đè & phát hiện drift:**
     - Bắt buộc `--run-id`, từ chối `--run-id` bảo vệ (`before`, `after`, `baseline_freeze`, `manifests`, `t3_ablation`) và từ chối ghi đè thư mục đã tồn tại (`exist_ok=False`); giữ nguyên toàn bộ `results/next_steps/t3_ablation/`.
     - Tính sẵn `seeded_orders` trước inference, ghi `run_manifest.json` (`input_hashes`, `model_hashes`, `source_hashes`, `split_sha256`, `candidate_fingerprint`, `baseline_reference`, `effective_options`, `seeded_orders`), ép kiểm baseline trước khi tạo `Engine` (trong comparative mode), kiểm source/config drift sau chạy và exit khác 0 khi có drift/lỗi; hỗ trợ `--standalone` và dependency injection (`engine_factory`, `segment_fn`, `source_paths`, `model_paths`).
  4. **R4 — Cổng kiểm chứng baseline thực chất trước inference trong `tools/evaluate_asr_integration.py`:**
     - `_verify_immutable_baseline()` từ chối `evidence_files` rỗng hoặc manifest lỗi; comparative mode (mặc định) fail non-zero (`2`, `run_status="failed_baseline_verification"`) **trước** khi gọi `engine_factory` / `_create_engine`.
     - Thêm cờ `--standalone` ghi rõ `comparison_mode="standalone_non_comparative"` và `baseline_verified=false` trong `run_manifest.json` và `summary.json`.
  5. **R5 — Typed boundary-aware entity scorer (`tools/entity_scorer.py`) & hạ kết luận trung thực:**
     - Phân tách rõ `lexical_proxy` (`is_semantic_gate=false`) khỏi `entity_eval` (`typed_boundary_entity_v1`: kiểm boundary số tránh `32`↔`132` và `3`↔`330`, giữ số `0` đầu cho phone/code, giữ dấu âm, thập phân, đơn vị nhiệt/cân nặng/%, tiền tệ, giờ AM/PM, kiểm số lượng trùng lặp multiplicity, đo `coverage` so với các thực thể tách từ `written`, giữ nguyên `unknown` cho label thiếu hoặc lời đọc chưa parse được `unparsed_spoken` — không ép thành pass hay fail).
     - Chấm lại chỉ đọc (read-only) `results/next_steps/t3_ablation/vad/vad_ablation.json` sang `results/next_steps/r5_vad_rescore.json` (không chạm file lịch sử).
- **Danh sách file đã sửa / thêm mới tổng hợp toàn Plan 01 (bao gồm R1–R5):**
  - **Sửa (12 files tracked + pre-Plan-01 untouched; diff lưu tại `results/next_steps/patches/{t1,t2,r1_r5}/`):** `s2s_edge/s2s_edge/{__main__,audio,backends,conversation,pipeline,registry,tts,tts_format}.py`, `tools/{run_e2e,s2s_web,start_s2s_web,evaluate_asr_integration}.py`, `tests/test_asr_integration.py`.
  - **Thêm mới (cần quản lý riêng qua rollback manifest 44 files `results/next_steps/plan01_expected_state_manifest.json`):** `tools/{rollback_baseline,ablate_profiles,entity_scorer}.py`, `tests/{test_asr_evaluation_contract,test_runtime_profiles,test_entity_scorer}.py`, `results/next_steps/manifests/{asr_integration_6dir,t3_ablation_split,t3_r3_smoke_split}.json`, cùng bằng chứng/log/patch dưới `results/next_steps/` (`baseline_freeze/`, `t3_ablation/` giữ nguyên trạng, `plan01_r3_smoke_vad/`, `r5_vad_rescore.json`, `patches/r1_r5/`, `logs/r*`).
  - **Ranh giới Plan 02 / Plan 03 giữ nguyên 100%:** Không chỉnh sửa `hybrid_textnorm.py`, `neural_textnorm.py`, `punct_vi.py`, `mt.py`; giữ `DEFAULT_ORT_POLICY = "auto"`; không chạm/restart server cổng `8766` (`PID 24863`); không ghi đè `results/asr_integration/{before,after}` hay `results/next_steps/t3_ablation/`.

- **Thay đổi khác kế hoạch và lý do:**
  1. **Sửa `tests/test_asr_integration.py` (`test_vad_and_silence`, line 38–48):** Trong kế hoạch T2, mặc định VAD trở về `legacy` (`min_sil_ms=500`, hợp nhất khoảng lặng <= 1.5s) và `upstream` (`min_sil_ms=320`) là opt-in; test cũ giả định mặc định tách 380ms nên được cập nhật để assert cả hai hành vi (`legacy` merge vs `upstream` split).
  2. **HTTP unit test trong `tests/test_runtime_profiles.py` dùng `s2s_web.build_handler` in-memory thay vì mở socket TCP:** Trong macOS sandbox mặc định, `bind()` loopback bị chặn (`PermissionError: [Errno 1] Operation not permitted`), nên unit test kiểm tra HTTP handler trực tiếp không cần bind cổng, còn kiểm tra HTTP server thật (E2E) được chạy ở Step 3d ngoài sandbox trên cổng cô lập `8767`/`8768`.
  3. **Sửa `tools/run_e2e.py` và `tools/s2s_web.py` thêm `--out-dir`:** Cần thiết để chạy kiểm tra CLI (`results/next_steps/cli_mode0_check`) và HTTP web thực trên cổng cô lập mà tuyệt đối không chạm vào `results/mic_web/server.pid` (server port `8766` của người dùng) hay `results/e2e/audio/`.
  4. **Lựa chọn `DEFAULT_ORT_POLICY = "auto"` trong code:** Giữ tương thích hành vi baseline trong khi cho phép cấu hình qua `--ort-policy`/`S2S_ORT_POLICY`; tuy nhiên đo đạc ở Step 3b & 3c cho thấy `basic` vượt trội cả `auto` lẫn `disable` mà không gặp optimizer crash/fallback trên macOS arm64.

---

## Bằng chứng test

### 1. Command thực tế + cwd + thời gian chạy
(Tất cả chạy từ cwd `/Users/khoa/study/Onevoice_final_ves/OneVoice_E2E_HybridTTS`, Python `/private/tmp/s2s-original-env/bin/python`, ghim `OMP_NUM_THREADS=4`, chạy tuần tự từng tác vụ suy luận, không chạy song song):

| Bước | Command thực tế | Mode sandbox | Bắt đầu -> Kết thúc (UTC / wall) | Exit code |
|---|---|---|---|---:|
| **T1 contract tests** | `python -m unittest discover -s tests -p test_asr_evaluation_contract.py -v` | Sandboxed | `11.414s` (8 tests) | `0` |
| **Step 3a (VAD ablation)** | `OMP_NUM_THREADS=4 python -u tools/ablate_profiles.py vad > results/next_steps/logs/t3a_vad_ablation.log 2>&1` | Sandboxed | `10:37:14 -> 10:47:04` (`~9m 50s`) | `0` |
| **Step 3b (ORT ablation)** | `OMP_NUM_THREADS=4 python -u tools/ablate_profiles.py ort --fixed-vad legacy > results/next_steps/logs/t3b_ort_ablation.log 2>&1` | Sandboxed | `10:47:49 -> 10:55:00` (`~7m 11s`) | `0` |
| **Step 3c (#1: legacy+basic)** | `OMP_NUM_THREADS=4 python -u tools/evaluate_asr_integration.py --run-id plan01_top1_legacy_basic --manifest results/next_steps/manifests/asr_integration_6dir.json --source-mode 0 --vad-profile legacy --ort-policy basic --tts-format hybrid --warmup 1 > results/next_steps/logs/t3c_top1_legacy_basic.log 2>&1` | Sandboxed | `10:56:39 -> 10:57:31` (`~52s`) | `0` |
| **Step 3c (#2: legacy+disable)** | `OMP_NUM_THREADS=4 python -u tools/evaluate_asr_integration.py --run-id plan01_top2_legacy_disable --manifest results/next_steps/manifests/asr_integration_6dir.json --source-mode 0 --vad-profile legacy --ort-policy disable --tts-format hybrid --warmup 1 > results/next_steps/logs/t3c_top2_legacy_disable.log 2>&1` | Sandboxed | `10:57:35 -> 10:58:36` (`~61s`) | `0` |
| **Step 3c (CLI `run_e2e` mode=0)** | `OMP_NUM_THREADS=4 python -u tools/run_e2e.py en vi results/e2e/audio/en_to_vi_special_input.wav --vad-profile legacy --ort-policy basic --source-mode 0 --out-dir results/next_steps/cli_mode0_check > results/next_steps/logs/t3c_run_e2e_mode0.log 2>&1` | Sandboxed | `10:58:52 -> 10:59:05` (`~13s`) | `0` |
| **Step 3d (HTTP thật mode=1 & mode=0)** | `python -u .../scratch/run_step3d_http.py > results/next_steps/logs/t3d_http_web.log 2>&1` | `BypassSandbox: true` (chỉ bước này, do socket + Apple Metal) | `11:02:42 -> 11:04:27` (`~1m 45s`) | `0` |
| **Rollback dry-run (pre-review)** | `python tools/rollback_baseline.py --expected-manifest results/next_steps/plan01_expected_state_manifest.json > results/next_steps/logs/t5_rollback_dry_run.log 2>&1` | Sandboxed | `< 3s` | `0` |
| **Rollback negative mismatch (pre-review)** | `python tools/rollback_baseline.py --expected-manifest .../tampered_expected.json` | Sandboxed | `< 2s` | `1` (expected abort) |
| **Pre-review unit tests (5 suites)** | `python -m unittest tests.test_hybrid_format tests.tests_identifiers tests.test_asr_integration tests.test_asr_evaluation_contract tests.test_runtime_profiles -v > results/next_steps/logs/t5_final_unit_tests.log 2>&1` | Sandboxed | `21.536s` (38 tests) | `0` |
| **R5 read-only VAD rescore** | `python -c "...entity_scorer.rescore_vad_report('results/next_steps/t3_ablation/vad/vad_ablation.json', ...)"` | Sandboxed | `< 1s` | `0` |
| **R3/R5 real-ONNX smoke** | `python tools/ablate_profiles.py vad --run-id plan01_r3_smoke_vad --split results/next_steps/manifests/t3_r3_smoke_split.json --repeats 1 \| tee results/next_steps/logs/r3_smoke_ablation.log` | Sandboxed | `11:29:21 -> 11:31:04` (`~1m 43s`) | `0` |
| **R2 live HTTP reuse/mismatch/reload** | `python .../scratch/verify_r2_http.py` (port cô lập `8769`, ghi `results/next_steps/logs/r2_http_reuse_check.log`) | `BypassSandbox: true` (socket localhost; không chạm cổng `8766`) | `11:31:59 -> 11:32:54` (`~55s`) | `0` |
| **R1–R5 Rollback dry-run & negative** | `python tools/rollback_baseline.py --expected-manifest results/next_steps/plan01_expected_state_manifest.json \| tee results/next_steps/logs/r5_rollback_dry_run.log` + tamper check (`r5_rollback_negative.log`) | Sandboxed | `< 2s` | `0` / `1` |
| **R1–R5 Final unit regression (6 suites)** | `python -m unittest tests/test_hybrid_format.py tests/tests_identifiers.py tests/test_asr_integration.py tests/test_asr_evaluation_contract.py tests/test_runtime_profiles.py tests/test_entity_scorer.py -v \| tee results/next_steps/logs/r5_final_unit_tests.log` | Sandboxed | `29.681s` (**49 tests**) | `0` |

### 2. Phiên bản dependency / model, cấu hình và thiết bị
- **Thiết bị & OS:** `macOS-27.2.1-arm64-arm-64bit`, CPU `Apple M5` (10 logical CPUs), `OMP_NUM_THREADS=4`.
- **Thư viện Python `3.11.15` (`/private/tmp/s2s-original-env/bin/python`):** `numpy 1.26.4`, `scipy 1.17.1`, `onnxruntime 1.31.1`, `soundfile 0.14.0`, `sentencepiece 0.2.3`, `mlx 0.32.4`, `mlx-lm 0.32.2`.
- **Models (`models/models.cpu.json`, ONNX CPU):**
  - `asr_vi`: `build/zip150/zip150_ctc_static_1500.onnx` (596,925,289 B, `sha256: 7c2dd9bdea0f...`)
  - `asr_xx_frontend`: `build/sensevoice/sv_frontend_v3.onnx` (1,228,928 B, `sha256: 9ba6897edf73...`)
  - `asr_xx_encoder`: `build/sensevoice/sv_enc_cls.onnx` (941,305,187 B, `sha256: 8daa5febf2e7...`)
  - `mt_enc`: `build/nllb/nllb_enc.onnx` (991,368,468 B, `S=64`, `sha256: 7d6482aec6b7...`)
  - `mt_decB`: `build/nllb/nllb_decB.onnx` (1,276,245,977 B, `Tm=96, B=4`, `sha256: 766c59b8fb2b...`) — đang dùng active (`[s2s] MT decoder in use: mt_decB`)
  - `mt_dec`: `build/nllb/nllb_dec.onnx` (1,276,803,214 B, `Tm=96`, `sha256: 3cabf31ad907...`)
  - `tts_vi`: `components/piper_npu/g2p/build/piper_vi_full_g8.onnx` (215,699,842 B, `sha256: 4b31090bff6b...`)
  - `tts_enko`: `build/supertonic/supertonic_full_N96_L150_s5.onnx` (398,278,832 B, `sha256: b3caaf3109fa...`)
  - `tts_zh`: `build/melo/melo_one.onnx` (1,312,973,091 B, `sha256: 307bb3a8ab15...`)
- **Learned Source TN (`source_mode=1`, Web HTTP):** `models/source/qwen3-4b-instruct-mlx4` trên `Apple Metal / MLX 4-bit` (`S2S_NEURAL_TN=1`), định dạng TTS mặc định `tts_format=hybrid`.

### 3. Exit code, số ca pass/fail/skip
- **Final 6 unit test suites hậu sửa review (`results/next_steps/logs/r5_final_unit_tests.log`):** `Exit code 0` — **49 passed, 0 failed, 0 skipped** trong `29.681s` (`test_hybrid_format`: 10, `tests_identifiers`: 5, `test_asr_integration`: 7, `test_asr_evaluation_contract`: 10, `test_runtime_profiles`: 10, `test_entity_scorer`: 7). *(Lưu ý nhỏ không đổi: 2 `ResourceWarning` file mở từ code gốc `tts.py:87` / `names.py:86` trong test mock gán asset dir và 1 `ResourceWarning` `conversation.py:166` `lid_vi.json`.)*
- **R1–R5 contract & isolated HTTP evidence:**
  - `plan01_r3_smoke_vad` (`results/next_steps/logs/r3_smoke_ablation.log`): Exit code `0`, `comparison_mode="comparative"`, `baseline_verified=true`, `drift_detected=false`, `failure_count=0`, `seeded_orders` lưu trước inference trong `run_manifest.json`; typed `entity_eval`: `legacy` **9 correct / 0 wrong / 6 unknown** vs `upstream` **7 correct / 2 wrong / 6 unknown** (mất ZIP `67297` ở cả `native_en_dev_EV0358` lẫn `syn_en_special`; `syn_vi_special_to_en` chưa gắn `expected_entities` giữ nguyên `6 unknown`, `coverage=0.0`).
  - Live isolated-port HTTP check (`results/next_steps/logs/r2_http_reuse_check.log`, port `8769`, user server port `8766` PID `24863` hoàn toàn không bị chạm): Exit code `0` — start (`source_mode=0, tts_format=legacy, pid=28250`) → match reuse (`pid=28250`) → profile mismatch blocked (`exit 2`, `pid=28250` giữ nguyên) → invalid env while healthy blocked (`exit 2`) → idle `--reload` (`pending=0`) đổi cleanly sang `tts_format=hybrid` (`pid=28259`) → real WAV job `f43d7087e45a4a4c9a61024de24b9c67` (`EV-0358`) hoàn tất trong `4.89s` với `asr='734 Main Street, Harper Valley Oregon,67297.'` → shutdown cô lập sạch port `8769`.
  - Read-only R5 rescore trên file lịch sử bất biến `results/next_steps/t3_ablation/vad/vad_ablation.json` (`results/next_steps/r5_vad_rescore.json`): `mean_coverage = 0.3859` (`7/57` entity phát hiện trong `written` có label typed đúng chuẩn trong `t3_ablation_split.json`); cả `legacy` lẫn `upstream` ghi nhận `10 correct / 20 wrong / 55 unknown` trên bảng nhãn cũ bị khuyết/mảnh, xác nhận metric cũ `7/9 vs 5/9` chỉ là lexical proxy.
- **Pre-Plan-01 baseline integrity:** Kiểm tra lại sau R1–R5 xác nhận `results/next_steps/baseline_freeze/freeze_manifest.json` (`verified=True`, 4/4 `evidence_sha256` khớp), `results/asr_integration/{before,after}/summary.json` và `results/next_steps/t3_ablation/{vad,ort}/` giữ nguyên `mtime`/`sha256`, **0 mismatches**.
- **Rollback guardrails (hậu R1–R5):** dry-run **`44/44` files match** exit `0` (`r5_rollback_dry_run.log`); negative tamper manifest bị chặn `aborted_precheck` exit `1` (`r5_rollback_negative.log`).
- *(Các run lịch sử pre-review giữ nguyên vẹn: Step 3a `t3_ablation/vad/vad_ablation.json`, Step 3b `t3_ablation/ort/ort_ablation.json`, Step 3c Top-2 `plan01_top1_legacy_basic` / `plan01_top2_legacy_disable` đều `6/6 ok`, Step 3d `http_web_summary.json` `8767`/`8768`.)*

### 4. Đường dẫn raw log, JSON, patch và audio
- **Raw logs:**
  - Pre-review Plan 01: `results/next_steps/logs/{t2_runtime_profiles_unit,t2_all_unit_tests,t3a_vad_ablation,t3b_ort_ablation,t3c_top1_legacy_basic,t3c_top2_legacy_disable,t3c_run_e2e_mode0,t3d_http_web,t3d_server_mode1,t3d_server_mode0,t5_rollback_dry_run,t5_rollback_negative,t5_final_unit_tests}.log`
  - Post-review R1–R5: `results/next_steps/logs/{r3_smoke_ablation,r2_http_reuse_check,r2_http_probe_server,r5_rollback_dry_run,r5_rollback_negative,r5_final_unit_tests}.log`
- **Ablation / Harness / Rescore JSONs & Output WAVs:**
  - Historical (untouched): `results/next_steps/t3_ablation/{vad/vad_ablation.json,ort/ort_ablation.json}`, `results/next_steps/plan01_top1_legacy_basic/*`, `results/next_steps/plan01_top2_legacy_disable/*`, `results/next_steps/cli_mode0_check/*`, `results/next_steps/http_web_summary.json`, `results/next_steps/http_web_mode{0,1}/*`
  - Post-review R3/R5: `results/next_steps/r5_vad_rescore.json`, `results/next_steps/plan01_r3_smoke_vad/{run_manifest.json,summary.json,vad/vad_ablation.json}`
- **Freeze, Manifest & Patches:** `results/next_steps/baseline_freeze/`, `results/next_steps/manifests/t3_r3_smoke_split.json`, `results/next_steps/patches/{t1,t2,r1_r5}/` (`plan01_tracked.patch`, `file_hashes.json`), `results/next_steps/plan01_expected_state_manifest.json`.

### 5. Kết quả trước / sau & chi tiết định lượng (tách riêng Cold / Warmup / Paired Warm, tách Native vs Synthetic)

#### 5A. Step 3a — VAD Ablation (cố định `ort_policy=auto`, `source_mode=0`, `tts_format=hybrid`)
- Cold init `registry_init_s`: `0.0003s`; `golden_warmup_s`: `7.3357s` (`asr_xx_frontend 0.01s`, `asr_xx_encoder 1.83s`, `mt_enc 0.00s`, `mt_decB 0.00s`, `tts_vi 0.59s`, `asr_vi 0.58s`, `tts_enko 0.23s`, `tts_zh 1.92s`).
- Paired warm runs (`n=3` lặp trên 7 core clips `paired_vad_core_ids`, thứ tự seeded `["upstream", "legacy"]`, seed `20261113`):

| Metric (7 core clips tổng hợp & audit 18 clips) | `legacy` (`150/500/1.5/3.0`) | `upstream` (`120/320/0.32/1.1`) | Chênh lệch (`upstream` vs `legacy`) |
|---|---:|---:|---:|
| **`core_sum_total_s` warm median (`n=3`)** | **`41.0356s`** | `60.8471s` | **`+48.28%` chậm hơn (FAIL gate > +10%)** |
| `core_sum_total_s` min / max / p95 (descriptive) | `40.7760s` / `45.5275s` / `45.0783s` | `59.9727s` / `61.1404s` / `61.1111s` | — |
| Tổng số segments trên toàn bộ 18 clips audit | **`36`** | `77` | `+113.9%` mảnh vụn |
| Lexical proxy EN (`7/9` vs `5/9` trên `EV-0358`, `EV-0157`, `EV-0491`, `EV-1508`, `EV-3459`, `EV-3775`, `3x syn_en`, `is_semantic_gate=False`) | **`7 / 9` pass** | `5 / 9` pass | **Mất ZIP `67297` ở cả native `EV-0358` và `syn_en_special`** |
| Typed `entity_scorer.py` rescore (`results/next_steps/r5_vad_rescore.json`, read-only trên 18 clips cũ) | `2/18` all-labeled-correct (`10c/20w/55u`, `7/57` labeled, `mean_coverage=0.3859`) | `1/18` all-labeled-correct (`10c/20w/55u`, `7/57` labeled, `mean_coverage=0.3859`) | `EV-0358` rơi từ all-labeled-correct xuống `wrong` trên `upstream`; đa số câu chưa gán `written` (`unknown=55`) |
| R3/R5 Typed `entity_eval` smoke ONNX (`plan01_r3_smoke_vad`, `EV0358` + `syn_en_special` + `syn_vi`) | **`9c/0w/6u`** (`EV0358` `2c/0w/0u`, `syn_en_special` `7c/0w/0u`, `syn_vi` `0c/0w/6u`, `mean_coverage=0.6667`) | `7c/2w/6u` (`EV0358` `1c/1w/0u`, `syn_en_special` `6c/1w/0u`, `syn_vi` `0c/0w/6u`, `mean_coverage=0.6667`) | **`upstream` sai `+2` thực thể (`67297` tách ở `EV0358` & `syn_en_special`)** |

- **Khác biệt định tính cốt lõi (bằng chứng lỗi của `upstream` trên cả native audio thật và synthetic audio):**
  - `native_en_dev_EV0358` (**Native Human Speech**, `../en-vi/BTC/EV-0358_in.wav`, `sha256: ba488dd557d33e42...`):
    - `legacy` (1 seg `[0.0, 5.71]`, median warm `1.9205s`): `asr_raw='734 Main Street, Harper Valley Oregon,67297.'` -> `mt='734 Main Street, Harper Valley Oregon, 67297.'` -> `tts=['bảy trăm ba mươi bốn main street, harper valley oregon, sáu bảy hai chín bảy.']` -> **PASS (`734`, `67297`)**, out `sha256: be4895e7aa49...`
    - `upstream` (3 segs `[[0.0, 2.3], [2.44, 3.88], [4.1, 5.68]]`, median warm `4.6692s`): `asr_raw='734 Main Street. | Harper Valley Oregon. | Six,7,29,7.'` -> `mt='Đường chính 734. | Harper Valley Oregon. | 6,7,29,7.'` -> `tts=[..., 'sáu phẩy bảy phẩy hai chín phẩy bảy.']` -> **FAIL (`missing=['67297']`)**, out `sha256: dc6ff54d7a6d...`
  - `syn_en_special` (**Synthetic TTS**, `en_to_vi_special_input.wav`, `sha256: a0d35c02db7a9646...`):
    - `legacy` (3 segs, median `8.0226s`): preserves `'67297'` (**PASS**), out `sha256: 1228a175a6ff...`
    - `upstream` (7 segs, median `13.0914s`): tách giữa `"is 6"` / `"7297"` thành `'And my ZIP code is6. | 729,7.'` -> `tts=[..., 'và mã dip của tôi là sáu.', 'bảy trăm hai mươi chín phẩy bảy.']` -> **FAIL (`missing=['67297']`)**, out `sha256: d25e0884fc29...`
  - **Lỗi pre-existing phát hiện thêm khi audit (`legacy` vẫn fail 2 ca ngoài VAD):**
    - `native_en_holdout_EV3459` (Native EN holdout): SenseVoice đọc `"thirty two years old"` thành `'That are two years old...'` (mất `'32'` ở cả `legacy` & `upstream`).
    - `syn_en_clear` (Synthetic EN): SenseVoice chèn dấu phẩy vào số điện thoại (`'Call 0912,34,5678.'` ở `legacy`, `'Call 09123,45678.'` ở `upstream`), làm trượt substring `'0912345678'` — bằng chứng đầu vào trực tiếp cho Plan 02/03.
- **Kết luận Step 3a (giới hạn phạm vi R5):** **`legacy` được giữ làm mặc định (`DEFAULT_VAD_PROFILE = "legacy"`)** dựa trên gate latency (`upstream` chậm hơn `+48.28%` và chia nhỏ segments `77` vs `36`), lexical proxy (`7/9` vs `5/9` với bằng chứng tách ZIP `67297` ở cả native `EV-0358` và `syn_en_special`), cùng typed `entity_scorer` rescore (`2/18` vs `1/18` all-labeled-correct, `mean_coverage=0.3859` do `unknown=55`) và smoke ONNX `plan01_r3_smoke_vad` (`9c/0w/6u` vs `7c/2w/6u`, `mean_coverage=0.6667`). Với `55` slot `unknown` trên 18 clips cũ (chưa có nhãn `written`), dữ liệu R5 cho thấy `legacy` vượt trội rõ trên nhóm có nhãn và không gán nhãn vượt phạm vi làm "toàn bộ entity gate đạt 100%".

#### 5B. Step 3b — ORT Policy Ablation (cố định `vad_profile=legacy`, `source_mode=0`, `tts_format=hybrid`, 4 core clips × 3 repeats seeded `seed=20261114`)
*(Mỗi repeat tạo session mới và đo riêng `cold_load_and_golden_s` rồi chạy 4 core clips tuần tự với thứ tự đảo ngẫu nhiên theo seed: rep 0 `[basic, disable, auto]`, rep 1 `[auto, basic, disable]`, rep 2 `[basic, auto, disable]`).*

| Policy | Level thực tế trên 8 graphs loaded | `ort_fallback` | `cold_load_and_golden_s` (median / min–max / p95) | `core_sum_total_s` warm (median / min–max / p95) | So sánh warm median vs `auto` | Text equivalence (`asr_raw`, `mt`, `tts_texts`) |
|---|---|---:|---:|---:|---:|---|
| **`basic`** (`#1`) | **8/8 `ORT_ENABLE_BASIC`** | **0/8 (`False`)** | **`3.7588s`** (`3.6609–3.9155`, p95 `3.8998`) | **`40.4094s`** (`29.3326–41.2796`, p95 `41.1926`) | **`-4.26%` (nhanh nhất ở cả 3 reps)** | 100% giống hệt `auto` & `disable` |
| **`disable`** (`#2`) | 8/8 `ORT_DISABLE_ALL` | 0/8 (`False`) | `6.1187s` (`5.9660–6.4460`, p95 `6.4133`) | `41.9225s` (`31.3112–43.8878`, p95 `43.6913`) | `-0.68%` | 100% giống hệt `auto` |
| **`auto`** (`#3`, baseline) | 1/8 `BASIC` (`sv_frontend_v3` 1.2 MB), 7/8 `DISABLE` (>200 MB) | 0/8 (`False`) | `6.2326s` (`6.0517–6.3931`, p95 `6.3770`) | `42.2074s` (`40.9827–42.6833`, p95 `42.6357`) | `0.00%` (ref) | Tham chiếu |

- **Điểm nổi bật về cold-load & ổn định:** Trong ORT 1.31.1 ARM64 trên Apple M5, bật `ORT_ENABLE_BASIC` cho cả các graph lớn (215 MB – 1.31 GB) không gây lỗi tối ưu hóa (`attempts: 1`, `ort_fallback: False` trên toàn bộ 8/8 session) và giúp tăng tốc khâu khởi tạo + golden warmup lên **~40%** (`asr_xx_encoder`: `0.49–0.60s` vs `1.83–1.91s`; `tts_zh`: `0.72–0.86s` vs `1.63–1.77s`), đồng thời thắng `auto` và `disable` trong mọi repeat (`rep=0`: `29.33s vs 31.31s vs 40.98s`; `rep=1`: `40.41s vs 41.92s vs 42.21s`; `rep=2`: `41.28s vs 42.68s vs 43.89s`). Lưu ý `rep=0` chạy ngay khi CPU còn mát nên thấp hơn đáng kể (`~29–31s` so với `~40–42s` ở các lượt sau khi CPU giữ tải liên tục).

#### 5C. Step 3c — Top-2 cấu hình chạy lại S2S 6 chiều (`asr_integration_6dir.json`, `source_mode=0`, `tts_format=hybrid`) vs Before / After cũ
- Hai cấu hình top-2 từ Step 3b được chọn: **Top 1 `legacy + basic`** và **Top 2 `legacy + disable`** (`legacy + auto` đứng #3 nên không chạy lại vòng 6 chiều theo đúng giới hạn 2 cấu hình của `PLAN.md`).

| Hướng S2S (Synthetic fixtures) | Baseline `before` (legacy-like) | Baseline `after` (upstream VAD + auto ORT ẩn) | **Plan 01 Top 2 (`legacy + disable`)** | **Plan 01 Top 1 (`legacy + basic`)** | Chênh lệch Top 1 vs `before` / vs `after` | Segments |
|---|---:|---:|---:|---:|---:|---:|
| `vi -> en` (`vi_to_en_special`) | `11.550s` | `11.539s` | `11.587s` | **`10.964s`** | `-5.1%` / `-5.0%` | `1` |
| `vi -> ko` (`vi_to_ko_special`) | `6.655s` | `7.039s` | `8.314s` | **`7.354s`** | `+10.5%` / `+4.5%` | `1` |
| `vi -> zh` (`vi_to_zh_special`) | `11.139s` | `12.236s` | `12.863s` | **`11.233s`** | `+0.8%` / `-8.2%` | `1` |
| `en -> vi` (`en_to_vi_special`) | `6.212s` | `11.893s` (`7 segs`) | `7.300s` | **`6.056s`** | `-2.5%` / **`-49.1%`** | **`3`** (vs `7` của `after`) |
| `ko -> vi` (`ko_to_vi_special`) | `4.400s` | `8.057s` (`6 segs`) | `6.152s` | **`4.842s`** | `+10.0%` / **`-39.9%`** | **`3`** (vs `6` của `after`) |
| `zh -> vi` (`zh_to_vi_special`) | `4.122s` | `6.887s` (`5 segs`) | `5.995s` | **`4.550s`** | `+10.4%` / **`-33.9%`** | **`2`** (vs `5` của `after`) |
| **Tổng wall 6 chiều (sau warmup)** | **`44.079s`** | `57.650s` | `52.210s` | **`44.998s`** | **`+2.09%`** vs `before` / **`-21.95%`** vs `after` | **`11`** vs `21` của `after` |
| Warmup time | — | — | `5.961s` | **`3.817s`** | **`-36.0%` vs `disable`** | — |

- Toàn bộ 6/6 hướng giữa `basic` và `disable` cho text (`asr`, `mt`, `tts_text`) **khớp 100%**, và `vi->{en,ko,zh}` khớp bitwise SHA256 audio output; CLI `tools/run_e2e.py en vi ... --vad-profile legacy --ort-policy basic --source-mode 0` cho `results/next_steps/cli_mode0_check/audio/en_to_vi.wav` có `sha256: 10b9c5257666ac22...` **trùng bitwise 100%** với `plan01_top1_legacy_basic/en_to_vi_special.wav` và `http_web_mode0/en_http_output.wav`!

#### 5D. Step 3d — Web HTTP thực (`tools/s2s_web.py` trên cổng cô lập `8767`/`8768`, `vad=legacy`, `ort=basic`)
- **`source_mode=1` (`S2S_NEURAL_TN=1`, Apple Metal Qwen3-4B, port `8767`, `api/health` in `0.510s`):**
  - `health`: `normalization=true`, `source_mode="1"`, `vad_profile="legacy"`, `ort_policy="basic"`, `code_fingerprint={"git_head": "43ea359b...", "s2s_edge_sha256": "ee6605f6..."}`.
  - `en -> vi`: `elapsed=15.328s` (`in=20.27s`, `out=16.03s`), `backend="integrated ONNX CPU + learned normalization on Apple Metal"`, `asr_segmentation` Qwen thực thi 3 segments (`1.55s`, `1.04s`, `1.99s`), `content_preserved=true`. **Sửa sạch các lỗi format do rule-mode gây ra:** `"330 pm"` -> `"3:30 tối"`, `"$15.50"` -> `"mười lăm đô la năm mươi xen"`, `"37.5 degrees Celsius"` -> `"ba mươi bảy phẩy năm độ c"`.
  - `vi -> en`: `elapsed=14.726s` (`in=11.96s`, `out=16.84s`), `asr_identifiers` chuyển `"không chín một hai ba bốn năm sáu bảy tám"` -> `"0912345678"` (`tts_text` đọc `"zero nine one two three four five six seven eight"`), nhưng `asr_segmentation` (`3.986s`, 2 attempts) cắt nhầm `"ngày mười / tháng mười một năm hai nghìn không hai sáu"` và MT dịch `"năm nghìn đồng"` -> `"five thousand denarii"`, `"ba bảy năm độ c"` -> `"three to seven years c"` (bằng chứng thực nghiệm để Plan 03 giải quyết).
  - Negative check: payload lỗi nhận HTTP `400` đúng hợp đồng.
- **`source_mode=0` (`S2S_NEURAL_TN=0`, Rule-only non-Metal, port `8768`, `api/health` in `0.507s`):**
  - `health`: `normalization=false`, `source_mode="0"`, `vad_profile="legacy"`, `ort_policy="basic"`.
  - `en -> vi`: `elapsed=9.669s`, `backend="integrated ONNX CPU (rule source preparation; neural_tn=0)"`, `normalization=[]`, output WAV `sha256: 10b9c5257666ac22...` (khớp bitwise CLI `mode=0`).
  - `vi -> en`: `elapsed=12.980s`, output WAV `sha256: f9715ea53524f82f...` (khớp bitwise `plan01_top1_legacy_basic/vi_to_en_special.wav`).
  - Negative check: HTTP `400`.

#### 5E. Báo cáo chuẩn S2S/TTS (input WAV/hash, ASR thô, bản dịch, text vào TTS, segments/chunk, output WAV/hash)
*Lưu ý về `chunk`: JSON trả về của `evaluate_asr_integration.py` và `s2s_web.py` ghi từng `segment` (sau VAD/split) và `tts_text` thực sự truyền vào synthesizer; việc chia sub-chunk nội bộ theo ký tự bên trong engine TTS không được dump thành trường con riêng trong JSON (`segments[].chunks` chưa tồn tại ở Plan 01).*

| Run / Hướng | Input WAV & `sha256` (prefix 16) | ASR thô (`asr_raw`) | Bản dịch (`mt` / `translation`) | Text thực sự vào TTS (`tts_text`) | Segments (VAD / NMT) | Output WAV & `sha256` (prefix 16) |
|---|---|---|---|---|---:|---|
| **`top1_legacy_basic` `vi->en`** | `results/e2e/audio/vi_to_en_special_input.wav` (`6d0aa3c1cc41fe82`) | `Cuộc họp lúc mười bốn giờ bốn ngày. Mười tháng mười một. Năm, hai nghìn, không. Hai, sáu. Giá vé năm nghìn đồng, giảm mười phần trăm. Nhiệt độ ba, bảy, Năm độ C. Gọi số không. Chín, một, hai, ba, bốn, năm, sáu, bảy, tám.` | `The meeting is at fourteen four days. It's the eleventh of November. Five, two thousand, no. Two, six. Tickets for five thousand, a discount of ten per cent. Temperatures three, seven, five degrees C. Call the number. Nine, one, two, three, four, five, six, seven, eight.` | Giống bản dịch (`tts_format=hybrid`, 1 segment) | `1` seg (`44100Hz`, `20.294s`) | `results/next_steps/plan01_top1_legacy_basic/vi_to_en_special.wav` (`f9715ea53524f82f`) |
| **`top1_legacy_basic` `vi->ko`** | `results/e2e/audio/vi_to_en_special_input.wav` (`6d0aa3c1cc41fe82`) | *(nt — `zip150`)* | `회의는 14일 4시입니다. 11월 10일 5천, 2천, 아니 두, 여섯. 5천원, 10% 줄여요 온도는 3, 7, 5°C입니다. 노트북에 전화해 아홉, 한, 두, 세, 네, 다섯, 여섯, 일곱, 여덟.` | `회의는 십사일 사시입니다. 십일월 십일 오천, 이천, 아니 두, 여섯. 오천원, 십 퍼센트 줄여요 온도는 삼, 칠, 오 도입니다. 노트북에 전화해 아홉, 한, 두, 세, 네, 다섯, 여섯, 일곱, 여덟.` | `1` seg (`44100Hz`, `16.869s`) | `results/next_steps/plan01_top1_legacy_basic/vi_to_ko_special.wav` (`c558f3bfe96d...`) |
| **`top1_legacy_basic` `vi->zh`** | `results/e2e/audio/vi_to_en_special_input.wav` (`6d0aa3c1cc41fe82`) | *(nt — `zip150`)* | `会议是四天上午十四点.这是11月十一.五,两千,没有.两个,六个.票价为五千元,降低了十分之一.温度为三,七,五摄氏度.打电话给零号码.九,一,二,三,四,五,六,七,八.` | `会议是四天上午十四点.这是十一月十一.五,两千,没有.两个,六个.票价为五千元,降低了十分之一.温度为三,七,五摄氏度.打电话给零号码.九,一,二,三,四,五,六,七,八.` | `1` seg (`44100Hz`, `15.256s`) | `results/next_steps/plan01_top1_legacy_basic/vi_to_zh_special.wav` (`349ffda498de...`) |
| **`top1_legacy_basic` `en->vi`** (`= mode=0`) | `results/e2e/audio/en_to_vi_special_input.wav` (`a0d35c02db7a9646`) | `[1] The meeting is at 330 pm on November 15, 2026, the ticket costs $15.50 with a 15% discount. | [2] The temperature is 37.5 degrees Celsius. | [3] My flight is VN 123, and my ZIP code is 67297.` | `[1] Cuộc họp là vào lúc 330 giờ tối ngày 15 tháng 11 năm 2026, giá vé là 15 đô la. 50 với giảm giá 15%. | [2] Nhiệt độ là 37. 5 độ C. | [3] Chuyến bay của tôi là VN 123, và mã địa chỉ là 67297.` | `[1] cuộc họp là vào lúc ba trăm ba mươi giờ tối... mười lăm đô la. năm mươi... | [2] nhiệt độ là ba mươi bảy. năm độ c. | [3] chuyến bay của tôi là vơn một hai ba, và mã địa chỉ là sáu bảy hai chín bảy.` | `3` segs (`22050Hz`, `15.378s`) | `results/next_steps/plan01_top1_legacy_basic/en_to_vi_special.wav` (`10b9c5257666ac22`) |
| **`top1_legacy_basic` `ko->vi`** | `results/e2e/audio/vi_to_ko.wav` (`55a6fc20f198b50b`) | `[1] 회의는 14일 4시 입니다 11월 115천 2천 아니 두 섯, | [2] 5000원 10% 줄여요 온도는 3,75로씨 입니다. | [3] 노트북의 전화해 아홉 한2세,넷 다섯 여섯 일곱 여덟.` | `[1] Cuộc họp vào ngày 14 giờ 4 giờ. | [2] Nó giảm 10 phần trăm so với 5000, và nhiệt độ là 3,75 độ C. | [3] Điện thoại của máy tính xách tay chín một hai tuổi, nét năm sáu bảy tám.` | `[1] cuộc họp vào ngày mười bốn giờ bốn giờ. | [2] nó giảm mười phần trăm so với năm nghìn, và nhiệt độ là ba phẩy bảy năm độ c. | [3] điện thoại của máy tính xách tay chín một hai tuổi, nét năm sáu bảy tám.` | `3` segs (`22050Hz`, `10.543s`) | `results/next_steps/plan01_top1_legacy_basic/ko_to_vi_special.wav` (`37ff1e2828d5...`) |
| **`top1_legacy_basic` `zh->vi`** | `results/e2e/audio/vi_to_zh.wav` (`349ffda498def0f1`) | `[1] 会议14天上午14点，这是11月11，我两清没有我两个的6个票价为5000元，降低了10分之1。 | [2] 温度为375，摄氏多，打电话给您号码，就12345678。` | `[1] Cuộc họp ngày 14 giờ 14 giờ sáng, đó là ngày 11 tháng 11, tôi đã quyết định giá 6 vé mà không có hai người của tôi là 5.000 đô la, giảm 1 phần mười. | [2] Nhiệt độ là 375 độ C, gọi cho số của bạn là 12345678.` | `[1] cuộc họp ngày mười bốn giờ mười bốn giờ sáng... năm nghìn đô la, giảm một phần mười. | [2] nhiệt độ là ba trăm bảy mươi lăm độ c, gọi cho số của bạn là một hai ba bốn năm sáu bảy tám.` | `2` segs (`22050Hz`, `14.966s`) | `results/next_steps/plan01_top1_legacy_basic/zh_to_vi_special.wav` (`73ce37a2c166...`) |
| **`http_web_mode1` `en->vi`** (`S2S_NEURAL_TN=1`) | `results/e2e/audio/en_to_vi_special_input.wav` (`a0d35c02db7a9646`) | `The meeting is at 330 pm on November 15, 2026, the ticket costs $15.50 with a 15% discount. The temperature is 37.5 degrees Celsius. My flight is VN 123, and my ZIP code is 67297.` | `Cuộc họp là lúc 3:30 tối vào ngày 15 tháng 11 năm 2026. Giá vé là 15,50 đô la với giảm giá 15%. Nhiệt độ là 37,5 độ C. Chuyến bay của tôi là VN 123, và mã địa chỉ của tôi là 67297.` | `cuộc họp là lúc ba giờ ba mươi phút tối vào ngày mười lăm tháng mười một năm hai nghìn không trăm hai mươi sáu. giá vé là mười lăm đô la năm mươi xen với giảm giá mười lăm phần trăm. nhiệt độ là ba mươi bảy phẩy năm độ c. chuyến bay của tôi là vơn một hai ba, và mã địa chỉ là sáu bảy hai chín bảy.` | `3` segs (spans `[0.33,10.33]`, `[10.63,13.91]`, `[14.19,19.97]`; `22050Hz`, `16.028s`) | `results/next_steps/http_web_mode1/en_http_output.wav` (`eaeeaa4f746a1e6b`) |
| **`http_web_mode1` `vi->en`** (`S2S_NEURAL_TN=1`) | `results/e2e/audio/vi_to_en_special_input.wav` (`6d0aa3c1cc41fe82`) | `cuộc họp lúc mười bốn giờ bốn ngày mười tháng mười một năm hai nghìn không hai sáu giá vé năm nghìn đồng giảm mười phần trăm nhiệt độ ba bảy năm độ c gọi số không chín một hai ba bốn năm sáu bảy tám` | `meeting at fourteen o'clock four days ten November of the year two thousand no two six Tickets for five thousand denarii are reduced by ten per cent. Temperature three to seven years c call 0912345678` | `meeting at fourteen o'clock four days ten November of the year two thousand no two six Tickets for five thousand denarii are reduced by ten per cent. Temperature three to seven years c call zero nine one two three four five six seven eight` | `1` seg (span `[0.0,11.958]`; `44100Hz`, `16.840s`) | `results/next_steps/http_web_mode1/vi_http_output.wav` (`5a7e55d66f449635`) |

### 6. Test chưa chạy và lý do
- **Native human speech holdout cho VI, KO, ZH:** **KHÔNG CHẠY (MISSING)** — kho mã nguồn cục bộ chỉ có audio người thật tiếng Anh tại `../en-vi/BTC/` (`data/asr/*.wav` và `data/clean/{vi,ko,zh}/*.wav` không tồn tại ngoài `.gitkeep`). Do đó tuyệt đối không suy diễn đánh giá holdout VI/KO/ZH từ synthetic fixtures.
- **Đánh giá nghe hiểu (MOS / listening audit) hoặc ASR-roundtrip WER trên file WAV xuất ra:** **CHƯA CHẠY** theo phạm vi Plan 01 (chỉ đo định lượng latency, segment count, text ASR/MT/TTS, thực thể trên audio đầu vào và SHA256 audio đầu ra).
- **Chưa chạy S2S 6 chiều với `source_mode=1` (Metal) và `legacy + auto` (#3):** Theo quy tắc `PLAN.md`, `source_mode=1` được kiểm trên 2 chiều web chính `EN->VI` & `VI->EN` qua HTTP thật ở Step 3d; còn batch 6 chiều Step 3c chỉ chạy Top-2 từ Step 3b (`legacy+basic`, `legacy+disable`).

---

## Còn lại

### Lỗi chưa sửa (thuộc phạm vi Plan 02 & Plan 03, cố ý KHÔNG sửa trong Plan 01)
1. **Lỗi Rule TextNorm & `hybrid_textnorm` khi `source_mode=0` (`en->vi`, `vi->*`):**
   - `en->vi`: `"$15.50"` -> `"15 đô la. 50"` -> `"mười lăm đô la. năm mươi"`; `"37.5 degrees Celsius"` -> `"37. 5 độ C"` -> `"ba mươi bảy. năm độ c"`; `"330 pm"` -> `"ba trăm ba mươi giờ tối"`; `"VN 123"` -> `"vơn một hai ba"`.
   - `vi->*`: không có dấu câu trong `zip150` khi đọc liền ở `mode=0` gây lỗi ngắt câu & nhận dạng giờ `"14h45"` thành `"mười bốn giờ bốn ngày"`.
   - Các lỗi này sẽ giải quyết ở Plan 02 / Plan 03.
2. **Lỗi `source_mode=1` (Qwen MT splitter) khi đoạn VI thiếu dấu câu dài (`vi->en` Step 3d):**
   - Tách nhầm cụm ngày/tháng (`"ngày mười / tháng mười một..."`) làm MT mất ngữ cảnh, đồng thời NLLB dịch `"năm nghìn đồng"` -> `"denarii"` và `"ba bảy năm độ c"` -> `"three to seven years c"`.
3. **Lỗi từ SenseVoice ASR (`syn_en_clear`, `EV-3459`):**
   - `syn_en_clear`: SenseVoice tự chèn dấu phẩy vào dãy số điện thoại (`0912,34,5678`), khiến rule chuẩn hóa nhận nhầm thành số thập phân (`09123,45678` -> `"chín nghìn một trăm hai mươi ba phẩy bốn năm sáu bảy tám"`).
   - `native_en_holdout_EV3459`: SenseVoice nghe `"thirty two years old"` thành `"That are two years old"`.
4. **Hạn chế dữ liệu & cảnh báo nhỏ:**
   - Thiếu hoàn toàn audio người thật cho **VI, KO, ZH** (đã khai báo `missing_native` rõ ràng).
   - File có sẵn `s2s_edge/s2s_edge/conversation.py:157` chưa đóng file handle `lid_vi.json` (gây 1 `ResourceWarning` trong unit test).
   - `DEFAULT_ORT_POLICY` hiện vẫn giữ `"auto"` để bảo toàn baseline; khuyến nghị Codex xem xét chuyển sang `"basic"` làm mặc định sau khi nghiệm thu kết quả Step 3b/3c (`basic` nhanh hơn `auto` ~4.3% warm, ~39.7% cold+warmup, và 0 fallback trên 8/8 graphs).

### Cách rollback
Khôi phục toàn bộ 44 file code/tool/test/manifest thuộc phạm vi Plan 01 về nguyên trạng trước Plan 01 (`results/next_steps/baseline_freeze/source_snapshot` gồm 35 file gốc đóng băng trước Plan 01 + xóa 9 file thêm mới):

```bash
cd /Users/khoa/study/Onevoice_final_ves/OneVoice_E2E_HybridTTS
# 1. Dry-run kiểm tra toàn bộ hash snapshot & file hiện tại (đã verify pass 44/44, exit 0 -> logs/r5_rollback_dry_run.log):
/private/tmp/s2s-original-env/bin/python tools/rollback_baseline.py \
  --expected-manifest results/next_steps/plan01_expected_state_manifest.json

# 2. Thực thi rollback (sau khi đã kiểm tra dry-run):
/private/tmp/s2s-original-env/bin/python tools/rollback_baseline.py \
  --expected-manifest results/next_steps/plan01_expected_state_manifest.json \
  --apply
```

*(Lưu ý: lệnh trên không đụng vào thư mục evidence cũ `results/asr_integration/{before,after}` hay server đang chạy trên cổng `8766`; nếu muốn dọn luôn report này thì xóa thêm `docs/workflow/next-steps/01-measurement/EXECUTION_REPORT.md` và `results/next_steps/` sau khi rollback xong.)*
