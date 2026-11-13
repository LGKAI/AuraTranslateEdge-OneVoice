# Hướng dẫn: các file code và cách chạy

> **Ngày đầu tiên tại VNG: đọc `docs/HUONG_DAN_NGAY1.md`** (từng bước: nối máy, đẩy gói, chạy 2 bộ dữ liệu trên web, đọc kết quả, sửa lỗi). Tài liệu này là bản đồ file và chi tiết.

Cập nhật 12/11/2026 (đã kiểm lại toàn bộ gói, xem mục 6). Hệ thống dịch nói offline vi ⇄ en / zh / ko trên HSPTEK AIBOX 8550 (QCS8550, Hexagon v73). Một đầu luôn là tiếng Việt.

## 1. Chạy nhanh nhất ở VNG (3 lệnh)

Điều kiện: gói `s2s_edge_pkg` và hai thư mục dữ liệu đã nằm trên hộp, ví dụ `/root/onevoice/` (xem mục 2).

```bash
adb shell
cd /root/onevoice/s2s_edge_pkg
bash tools/vng_run_all.sh /root/onevoice 30     # kiểm tra, cài, đo bộ mẫu 30 câu/chiều, đo bộ nhiễu. Ca 2: đổi 30 thành 0 (toàn bộ)
```
Lệnh này tự chạy lần lượt `recon.sh` → `install.sh` → `session0.sh` (kiểm tra từng đồ thị với số chuẩn, chọn bộ model chạy được) → `run_btc.sh` → `run_adv.sh`. Mỗi bước ghi log, bước lỗi không làm dừng bước sau. Kết quả nằm ở `logs/vng_<ngày giờ>.tgz`; kéo về bằng `adb pull`.

### Web theo dõi và chạy thử (chạy trên chip, laptop chỉ hiển thị)

```bash
# trên hộp (giữ cửa sổ này mở). Model nạp một lần; dữ liệu đọc từ /root/onevoice
bash tools/box_serve.sh vi,ko          # vi,ko = cặp mặc định cho tab thu âm (đổi được trên web)
# trên laptop
python tools/laptop_web.py             # = adb forward tcp:8770 tcp:8770 + mở http://127.0.0.1:8770
```
Trang web do chính hộp phục vụ qua cáp USB-C, không cần mạng. Inference chạy trên NPU của hộp; đầu trang hiện tên máy, SoC id, chữ **NPU** và bộ model đang dùng để chứng minh điều đó.
- **Tab "Chạy bộ dữ liệu":** chọn bộ BTC (có bản dịch chuẩn) hoặc ADV (bộ nhiễu, chỉ có transcript), chọn chiều (hoặc cả 6), mức ồn (ADV), số clip mỗi chiều (0 = tất cả), bấm "Chạy trên chip". Mỗi clip hiện một dòng: nghe âm thanh vào và câu gốc; ASR ra gì, mất bao lâu, lỗi bao nhiêu; MT ra gì, mất bao lâu, chrF và câu chuẩn (BTC); TTS nghe âm thanh ra, mất bao lâu; tổng thời gian và thời gian NPU. Bảng tổng hợp cập nhật liên tục: lỗi ASR toàn bộ, chrF, thời gian trung bình từng chặng, p50/p90, RTF, chia theo mức ồn với ADV. Nút "Tải CSV" tải toàn bộ kết quả.
- **Phát nối câu (tab thu âm):** ô "Phát nối câu" bật thì hộp trả từng câu ngay khi xong (endpoint `/api/stream`), laptop xếp hàng phát liên tục. Cách làm giống chế độ `live` trên hộp.
- **Tab "Thu âm thử":** bấm thu bằng mic laptop (hoặc gửi file WAV/MP3), chọn chế độ hội thoại (tự nhận ngôn ngữ trong cặp) hoặc chiều cố định. Hiện ASR → MT → TTS, thời gian từng chặng đo trên hộp, thời gian vòng laptop↔hộp, và phát âm thanh dịch.
- Kết quả lưu trên hộp: `logs/webui/<run>.jsonl`, âm thanh ra `logs/webui/<run>/*.wav`, các lượt thu âm `logs/webui/mic/`. Kéo về bằng `adb pull`.
- `S2S_NO_WARMUP=1 bash tools/box_serve.sh` mở trang ngay, model nạp ở clip đầu tiên (mặc định nạp và chạy thử mọi đồ thị trước).
- Dòng lệnh không giao diện vẫn còn: `tools/laptop_ptt.py` (bấm Enter để nói) dùng cùng server.

Có mic/loa USB cắm thẳng vào hộp thì dùng `python -m s2s_edge live --pair vi,ko --continuous` (đặt `S2S_MIC` / `S2S_SPK`, xem `tools/audio_check.sh`).

## 2. Chuẩn bị ở nhà (laptop)

```powershell
pip install numpy soundfile sounddevice
python tools/laptop_check.py --pkg C:\Users\Admin\Downloads\onevoice_deploy\s2s_edge_pkg     # in OK hoặc FIX: adb, driver, mic, loa, checksum
```
- adb: máy LOQ của Khanh đã cài sẵn ở `C:\platform-tools` (đã thêm vào Path; mở terminal mới để có hiệu lực). Máy khác: tải Android platform-tools, giải nén vào `C:\platform-tools`, thêm vào Path, thử `adb version`. Kiểm tra cáp và driver bằng điện thoại Android (`adb devices` phải hiện một thiết bị).
- Thư mục mang đi `C:\Users\Admin\Downloads\onevoice_deploy\`: `s2s_edge_pkg/` (gói, khoảng 8,3 GB), `onevoice_btc_data/` (bộ mẫu BTC, 143 MB), `onevoice_btc_adv/` (bộ nhiễu, 212 MB), `SHA256SUMS` (checksum của gói), `_keep/` (không cần đẩy lên hộp: script export model, notebook Kaggle, tài liệu BTC/HSPTEK, báo cáo, kết quả đo, bản sao git `repo_all_branches.bundle`).
- Đẩy lên hộp: `adb push <thư mục> /root/onevoice/` cho từng thư mục, rồi `adb shell "cd /root/onevoice/s2s_edge_pkg && sha256sum -c ../SHA256SUMS --quiet && echo CHECKSUM_OK"`. Bình thường dùng `python tools\laptop_push.py` thay cho các lệnh `adb push` thủ công. Chỉ có một cáp USB-C ↔ USB-C: không có console serial, LAN hay ổ USB dự phòng (bảng xử lý sự cố đầy đủ ở `HUONG_DAN_NGAY1.md`).
- Bốn quy tắc với máy dùng chung: không đổi mật khẩu, không flash lại OS, không chép đè thư viện hệ thống (`env.sh` tự đặt đường dẫn thư viện QAIRT 2.51 trong gói lên trước, không đụng `/usr/lib`), không sửa cấu hình mạng.

## 3. Bản đồ file

### Gói chạy `s2s_edge/`
| File | Làm gì |
|---|---|
| `__main__.py` | Lệnh `python -m s2s_edge file | live | serve`. `file`: dịch một file WAV, in thời gian từng chặng. `live`: mic/loa trên hộp, phát nối câu. `serve`: web + API trên 127.0.0.1:8770 (laptop xem qua `adb forward`) |
| `webui.py`, `webui.html` | Web theo dõi: chạy cả 2 bộ trên hộp (đóng trình duyệt vẫn chạy), mỗi clip ASR → MT → TTS kèm thời gian, ASR nghe lại để chấm TTS, tự đánh dấu vấn đề, chạy lại từng clip, thu âm, thử câu chữ, tải 1 file zip kết quả |
| `datasets.py` | Đọc bộ BTC (xlsx) và bộ ADV (csv/txt), chấm lỗi ASR và chrF |
| `registry.py` | Đọc `models.json`, nạp đồ thị khi cần, tự hạ cấp MT khi lỗi bộ nhớ (`MtFallback`), warmup |
| `backends.py` | Hai nền chạy: `ORT` (CPU, để phát triển) và `QNN` (NPU qua qai_appbuilder, đặt đường dẫn thư viện, chọn `S2S_PERF`) |
| `audio.py` | Đọc/ghi WAV, VAD theo năng lượng, cắt đoạn nói, ghi mic và phát loa (`arecord`/`aplay`, `S2S_MIC`/`S2S_SPK`) |
| `asr.py` | ASR tiếng Việt (Zipformer CTC) và en/zh/ko (SenseVoice) |
| `punct_vi.py` | Khôi phục dấu câu và chữ hoa cho ASR tiếng Việt (vibert-capu int8, không cần torch) |
| `mt.py` | NLLB-600M: encoder + bộ giải mã từng bước, beam 4, chặn lặp 3-gram |
| `tts.py` | TTS: Piper (vi), MeloTTS (zh), Supertonic (en, ko); tự nhận thứ tự đầu ra của chip |
| `textnorm.py` | Đọc số thành chữ cho cả bốn ngôn ngữ, glossary |
| `names.py` | Tên riêng Latin cho TTS tiếng Việt/Trung/Hàn |
| `conversation.py` | Bộ định tuyến hội thoại (nhận ra tiếng Việt hay tiếng đối tác), tách câu, xử lý theo câu cho phát nối |
| `pipeline.py` | Hướng dịch cố định (`--src --tgt`) |
| `bench.py`, `denoise.py` | Đo tốc độ; khử ồn GTCRN (tắt mặc định vì không cải thiện) |

### Công cụ `tools/` (dùng khi chạy thật)
| File | Làm gì |
|---|---|
| `vng_run_all.sh` | Một lệnh cho cả buổi trên hộp (mục 1) |
| `recon.sh` | Khảo sát hộp: OS, glibc, Python, FastRPC, thư viện QNN, nhiệt độ, âm thanh |
| `session0.sh` | Cài (nếu chưa), kiểm tra từng đồ thị với số chuẩn, thử dịch, đo nhanh; ghi bộ model chạy được vào `logs/models_ok.txt` |
| `env.sh` | Đặt `LD_LIBRARY_PATH`/`ADSP_LIBRARY_PATH` trỏ vào QAIRT 2.51 trong gói; chạy sau khi kích hoạt `.venv` |
| `run_btc.sh`, `eval_btc.py` | Chạy bộ mẫu BTC trên NPU (6 chiều). Chấm: lỗi ASR, chrF/BLEU của MT, chrF của âm thanh ra (ASR nghe lại), thời gian từng chặng |
| `run_adv.sh`, `eval_adv.py` | Chạy bộ nhiễu (sạch, nhà xưởng 15 dB, công trường 10 dB, rất ồn 5 dB). Bộ này không có bản dịch chuẩn nên đo lỗi ASR và độ ổn định của bản dịch so với bản dịch từ chữ đúng |
| `start_box.sh` | Trên hộp, một lệnh: cài (lần đầu), kiểm tra model trên NPU, mở web |
| `box_serve.sh` | Bật web + server trên hộp (start_box.sh gọi lệnh này) |
| `laptop_push.py` | Trên laptop: đẩy gói + 2 bộ dữ liệu lên hộp qua adb, kiểm checksum trên hộp |
| `laptop_web.py` | Trên laptop: adb forward rồi mở trang web của hộp |
| `laptop_pull.py` | Trên laptop: kéo toàn bộ log của hộp về một file zip (khi tải zip trên web không được) |
| `laptop_ptt.py` | Trên laptop: bấm-để-nói bằng dòng lệnh, dùng cùng server |
| `laptop_check.py` | Kiểm tra máy laptop trước khi đi |
| `golden.py`, `infer_check.py` | Số chuẩn cho từng đồ thị và đối chiếu (AI Hub hoặc chip) |
| `audio_check.sh` | Liệt kê và thử mic/loa trên hộp |
| `thermal_soak.py` | Chạy liên tục để xem nhiệt độ và tốc độ có giảm không |
| `build_package.py` | Đóng gói bản mang đi |
| `docker_run.sh` | Chạy gói trong container Ubuntu 22.04 ARM64 khi hộp không đúng OS |
| `compare_runs.py` | So hai lần chạy bằng bootstrap ghép cặp |
| `tts_web.py` | Trang web nghe thử TTS (chạy trên laptop, không phải trên hộp) |

### Công cụ nghiên cứu (không cần khi demo)
`btc_split.py` (chia dev/holdout), `tune_decoding.py` và `rerank_experiments.py` (chỉnh giải mã, chọn lại ứng viên), `eval_mt.py`, `eval_e2e.py`, `eval_vi_punct.py`, `check_punct_vi.py`, `eval_btc_io.py`, `eval_denoise.py`, `export_gtcrn.py`, `lid_features.py`, `lid_robust.py`, `mix_noise.py`, `relink_device.py`, `profile_links.py`, `live_selftest.py`, `md_to_pdf.py`.

### Thư mục khác
| Thư mục | Nội dung |
|---|---|
| `assets/` | SentencePiece, mã ngôn ngữ NLLB, bảng Supertonic, model dấu câu `vi_capu/` (file `.onnx` chỉ có trong gói, không đưa lên git) |
| `golden/` | Số chuẩn cho `golden.py` (không đưa lên git) |
| `eval/` | Danh sách clip, chia dev/holdout; `cpu_ref/` là kết quả CPU để so với chip (chứa câu của BTC, không đưa lên git) |
| `samples/` | Vài file WAV để thử nhanh |
| `docs/` | Báo cáo và tài liệu (xem `README.md`) |
| `models_qcs8550/` | Bộ model chip cho hộp này (9 file `.bin`, QAIRT 2.51); `models_fast.json` bỏ bộ giải mã beam (dùng khi thiếu bộ nhớ) |
| `models/` | Bộ model link cho IQ-9075, chỉ để dự phòng khi `.bin` của QCS8550 bị từ chối |
| `wheelhouse_cp310/`, `wheelhouse/` | Thư viện Python offline cho Ubuntu 22.04 (Python 3.10, dùng cho hộp này) và 24.04 (Python 3.12), kèm pip để tự tạo venv khi hộp thiếu gói `python3-venv` |
| `syslibs/` | `libatomic.so.1` arm64 (Ubuntu 22.04), chỉ dùng khi hộp không có; `env.sh` tự nhận |

## 4. Dữ liệu BTC

- Bộ mẫu: Drive "Vie-Eng / Vie-Kor / Vie-Man dataset", 100 câu (en) và 180 câu (ko, zh) mỗi chiều. Đã tải về `onevoice_btc_data/`.
- Bộ nhiễu (ADV): cùng cấu trúc thư mục nhưng 25 câu mỗi chiều, mỗi câu ghi 4 lần (sạch, nhà xưởng 15 dB, công trường 10 dB, rất ồn 5 dB). Đã tải về `onevoice_btc_adv/`. Ở Vie-Kor và Vie-Man, câu tiếng Việt và câu Hàn/Trung cùng mã **không phải bản dịch của nhau**, mỗi file chỉ là chữ của chính file âm thanh đó. Vì vậy không chấm chrF với bản dịch chuẩn được, chỉ đo lỗi ASR và độ ổn định.
- Không dùng dữ liệu BTC để huấn luyện.

## 5. Biến môi trường

| Biến | Tác dụng |
|---|---|
| `S2S_MODELS` | Đường dẫn `models.json` |
| `S2S_BACKEND` | `npu` (mặc định) hoặc `cpu` |
| `S2S_PERF` | Chế độ hiệu năng NPU: `DEFAULT`, `HIGH_PERFORMANCE`, `BURST` |
| `S2S_MT_MAXB` | Giới hạn beam lớn nhất của MT (hạ khi thiếu bộ nhớ) |
| `S2S_VI_PUNCT` | `model` (mặc định), `1` (luật), `0` (tắt) |
| `S2S_MIC`, `S2S_SPK` | Thiết bị ALSA cho mic/loa trên hộp |
| `S2S_SYSTEM_QNN=1` | Dùng thư viện QAIRT của hệ thống thay vì bản trong gói |
| `S2S_PORT` | Cổng của `box_serve.sh` (mặc định 8770) |

## 6. Đã kiểm và chưa kiểm

**Đã kiểm (12/11):**
- Mọi file model mà `models.json` trỏ tới đều có (cả `models_qcs8550` và `models`). Checksum của gói khớp sau khi chép.
- Cài offline trong container Ubuntu 22.04 ARM64 không mạng, có sẵn `python3` nhưng không có `python3-venv`: `install.sh` chạy hết, import được numpy, scipy, soundfile, sentencepiece, kaldi-native-fbank, onnxruntime, jiwer, sacrebleu, openpyxl và qai_appbuilder. Trước khi sửa, có 4 lỗi chắc chắn làm hỏng việc cài trên hộp: wheelhouse thiếu `pyyaml` (qai_appbuilder bắt buộc), thiếu `sounddevice` và `tabulate`, `numpy` của bản Python 3.12 quá mới cho qai_appbuilder, và thư viện hệ thống `libatomic.so.1` không có trong Ubuntu tối giản.
- Trong cùng container: khôi phục dấu câu tiếng Việt chạy đúng trên CPU ARM, `recon.sh` chạy hết, đọc bộ ADV đủ 100 clip mỗi chiều, `session0.sh` chạy hết các bước và đóng gói log. Các bước NPU báo lỗi là đúng vì container không có chip; thư viện duy nhất còn thiếu là `libcdsprpc.so` (FastRPC của chính hộp).
- Web (`serve`): trong container ARM64 server mở được, trả đúng trang, danh sách clip (BTC 920, ADV 600), âm thanh vào; chạy clip báo lỗi rõ ràng vì không có NPU. Giao diện đã thử đầy đủ trên laptop với pipeline giả (chạy BTC, ADV theo mức ồn, tổng hợp, CSV, gửi âm thanh ở tab thu âm).

**Chưa kiểm:**
- NPU trên hộp thật (`session0.sh` ở VNG sẽ xác nhận). Đã kiểm trên chip ảo AI Hub.
- `adb forward`, web và mic thật với hộp (chưa có hộp để cắm). Kết quả thật của web chỉ có khi chạy trên hộp.
- `--backend cpu` **không còn dùng được**: các file ONNX chạy CPU đã bị xóa khi dọn Downloads (gói chỉ có file `.bin` cho NPU). Bảng xử lý sự cố trong `plan_os_hardware.pdf` có dòng "tạm chạy --backend cpu"; dòng đó không áp dụng được nữa.
- Thời gian chạy của phần CPU ARM trên hộp (dấu câu, tách câu).
