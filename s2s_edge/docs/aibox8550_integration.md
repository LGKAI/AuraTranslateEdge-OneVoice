# Nhánh tích hợp `integrate/aibox8550` — chạy trên HSPTEK AIBOX 8550

Nhánh này gộp ba nhánh:

| Nhánh | Đóng góp chính |
|---|---|
| `feat/plan01-measurement` (đã chứa `main`, `contest_VNG`, TTS hybrid format) | Runtime profile thống nhất (`vad_profile`, `ort_policy`, `source_mode`, `tts_format`), chính sách ORT có fallback, TTS hybrid format, bộ test/đánh giá Plan-01, web mic (`tools/s2s_web.py`) |
| `asr-improvement` | Dấu câu/viết hoa/câu hỏi tiếng Việt, tách mệnh đề, VAD tách câu, fallback ORT |
| `asr-speed-accuracy` (chứa `asr-improvement`) | Cửa sổ ASR nhiều kích thước (bucket, `graph_index`), đặc trưng SenseVoice trên host, đầu RNN-T Zipformer, VAD hysteresis + gợi ý ngắt câu, BURST sau lần nạp đầu |

Hai nhóm nhánh có lịch sử git tách rời; thư mục `s2s_edge/` của cả hai bắt nguồn từ `abe8e30` (contest_VNG) nên được gộp 3 chiều với gốc đó.

## Hành vi mặc định sau khi gộp

| Thành phần | Mặc định | Cách đổi |
|---|---|---|
| VAD | `hybrid` (VAD hysteresis của nhánh ASR, gợi ý ngắt câu ≥ 320 ms cho dấu câu tiếng Việt) | `--vad-profile legacy\|upstream\|hybrid` hoặc `S2S_VAD_PROFILE`; trong `hybrid`, `S2S_VAD_MODE=split\|join` |
| `legacy` / `upstream` | Giữ nguyên VAD gốc, bit-exact, để tái lập số đo Plan-01 (đoạn tối đa 12 s) | — |
| ORT (backend CPU) | `auto` | `--ort-policy` / `S2S_ORT_POLICY` |
| Chuẩn hoá nguồn | `source_mode=0` (punct_vi + rule) | `S2S_NEURAL_TN=1` chỉ dùng được trên Mac (Qwen qua MLX). Trên AIBOX tự quay về rule-based và in cảnh báo, không crash |
| TTS format | `hybrid` | `S2S_TTS_FORMAT=legacy`; `neural` trên AIBOX tự quay về `hybrid` |
| CRF nhận dạng mã định danh | Dùng nếu có `python-crfsuite` + `assets/identifier.crfsuite` | Thiếu một trong hai: chỉ dùng ngữ pháp ngữ cảnh |
| ASR | Tương thích ngược: `models.json` cũ (1 cửa sổ, frontend graph SenseVoice, CTC) chạy như trước | Thêm bucket/RNN-T/host features bằng `--asr-config` khi dựng gói (bên dưới) |

## Dựng gói cài đặt (trên máy dev)

```bash
cd s2s_edge
# Gói giống trước (9 context binary đã kiểm chứng trên QCS8550):
python tools/build_package.py --models-dir <models> --wheelhouse <whl_cp312> --wheelhouse310 <whl_cp310> --out dist/s2s_edge_pkg

# Thêm cải tiến ASR khi đã compile trên AI Hub (docs/asr_optimization.md, tools/build_asr_buckets_aihub.py):
python tools/build_package.py --models-dir <models> --wheelhouse310 <whl_cp310> \
    --asr-config <asr.json> --out dist/s2s_edge_pkg
```

`<asr.json>` là đoạn `models.json` chứa các khoá `asr_vi`, `asr_vi@<frames>`, `asr_xx_encoder`, `asr_xx_encoder@<frames>` (`ctx` + `graph_index`), `asr_vi_decoder` / `asr_vi_joiner` (`onnx`) và `"_sv_frontend": "host"`; đường dẫn tương đối theo vị trí file đó. Các tệp được chép vào `models/`, khoá ASR một-cửa-sổ cũ bị thay.

Nếu `models/normalization/identifier.crfsuite` có trong repo (huấn luyện bằng `tools/train_identifier_tagger.py`), gói sẽ chép vào `assets/`.

**Wheelhouse:** `python-crfsuite` nay là tuỳ chọn. Wheelhouse cũ vẫn cài được (`install.sh` cài các gói tuỳ chọn nếu có wheel). Muốn bật CRF trên box, tải thêm wheel aarch64:

```bash
pip download --only-binary=:all: --platform manylinux2014_aarch64 --python-version 310 python-crfsuite -d <whl_cp310>
```

## Chạy trên AIBOX 8550

```bash
bash install.sh && . .venv/bin/activate && . tools/env.sh
bash tools/session0.sh                     # golden check từng graph (kể cả cửa sổ ASR) + smoke test + benchmark
S2S_MIC=plughw:1,0 S2S_SPK=plughw:1,0 bash run_demo.sh vi en --continuous
python -m s2s_edge live --pair vi,ko --models models_qcs8550/models.json --backend npu --vad-profile hybrid
```

Web mic (Plan-01) trên box, mở từ laptop qua SSH tunnel (trình duyệt chỉ cho dùng micro trên `localhost`/https):

```bash
# trên box (repo đầy đủ, không phải gói):
S2S_MODELS=s2s_edge/models_qcs8550/models.json S2S_BACKEND=npu python tools/s2s_web.py --port 8766
# trên laptop:
ssh -L 8766:localhost:8766 <user>@<ip-box>     # rồi mở http://localhost:8766
```

`s2s_edge/s2s_dashboard.py` (trước đây `s2s_edge/s2s_web.py` của nhánh ASR) là dashboard thử nghiệm dùng torch/transformers trên laptop, **không** chạy trên box (gói không có torch).

## Chưa kiểm chứng

- Chưa chạy trên box thật trong lần gộp này: cần `bash tools/session0.sh` để xác nhận golden + smoke test.
- Context binary cho bucket ASR / encoder có `encoder_out` (RNN-T) chưa compile trên AI Hub; `graph_index` là giả định thứ tự link job, `golden.py check` sẽ báo nếu sai.
- 5 test `RealAsrVadRegressionTests` cần model ONNX trong `build/` nên chưa chạy được trên máy gộp; các test đơn vị còn lại đều qua.
- Profile `hybrid` chưa được chạy lại bộ hồi quy ZIP/số điện thoại của Plan-01 (`tests/test_runtime_profiles.py::RealAsrVadRegressionTests`, `tools/ablate_profiles.py`). Nếu thấy mã số bị cắt, dùng `--vad-profile legacy`.
