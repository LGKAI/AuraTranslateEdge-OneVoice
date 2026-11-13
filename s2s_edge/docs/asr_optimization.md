# Tối ưu ASR trong s2s_edge: tốc độ và độ chính xác

Ngày: 12/11/2026. Phạm vi: khâu ASR của pipeline `s2s_edge` (Zipformer-150M cho tiếng Việt, SenseVoice-Small cho en/zh/ko), từ lúc thu âm đến lúc có văn bản đưa vào MT. Không đổi MT, TTS.

Quy ước nhãn như `improvement_playbook.md`: **[Đo]** = số do ta đo (máy, dữ liệu ghi rõ); **[Đọc]** = đã đọc nguồn gốc; **[Tóm tắt]** = chỉ thấy qua tóm tắt; **[Giả thuyết]** = chưa đo.

## 0. Tóm tắt

| Thay đổi | Đo được (CPU M1, ONNX fp32) | Trạng thái |
|---|---|---|
| Cửa sổ tĩnh nhiều kích thước + đặc trưng SenseVoice trên host | Câu 3 s: Zipformer 243 → **60 ms**, SenseVoice 905 → **109 ms**. 10 clip BTC (en): ASR trung bình 1196 → **437 ms**, transcript giữ nguyên | code + script export xong; chưa compile AI Hub |
| VAD hysteresis + gộp ngắt ngắn + giới hạn 14 s | WER FLEURS-en 7,27 → **6,88 %**, FLEURS-vi 5,30 → **4,89 %** (250 câu mỗi tiếng) | xong |
| Nhánh RNN-T cho Zipformer | FLEURS-vi: WER 4,83 → **4,40 %** (−9 % tương đối), +17 ms host/câu | xong; cần graph có `encoder_out` |
| Cả pipeline ASR tiếng Việt mới (VAD + RNN-T + chuẩn hoá mới) | FLEURS-vi: WER 5,30 → **4,53 %** (−15 % tương đối) | xong |
| BURST đặt sau khi nạp model | chưa đo (cần máy thật) | xong |
| Sửa "at530 AM" | đúng trên clip BTC EV-3775 | xong |
| Chuẩn hoá RMS phần có tiếng | WER như cách cũ trên FLEURS vi/en ở 0 / −25 / −40 dB (mục 3) | code xong |
| Ngưỡng bắt đầu thu của `listen()` | sửa lỗi micro nhạy thấp không kích hoạt (chưa đo trên máy) | code xong |

## 1. Hiện trạng và nguyên nhân lỗi (đã kiểm chứng)

Bộ 10 clip BTC en→vi do Hieu chạy (`btc_half1_20261112_1610`) được chạy lại trên CPU bằng chính code của nhánh và tái hiện đúng các lỗi. Nguyên nhân:

| Triệu chứng | Nguyên nhân tìm được |
|---|---|
| ASR luôn ~0,92 s dù câu dài 2,3 s hay 9 s | Đồ thị tĩnh: Zipformer luôn xử lý cửa sổ 15 s (1500 frame), SenseVoice luôn 29 s (464000 mẫu / 500 frame LFR). Câu ngắn trả tiền cho phần đệm. |
| Mất từ cuối: "followers" → "follower.", "tuesday" → "to." | `audio.segment()` đặt ngưỡng `floor + 0,12·(ceil − floor)` ≈ **18 dB dưới đỉnh cục bộ**; âm cuối yếu (/s/, /z/, phụ âm tắc) thấp hơn nguyên âm 20–30 dB nên bị coi là im lặng; chỉ đệm 150 ms. `audio_in_s` trong JSON của Hieu khớp đúng từng mẫu với `segment()` (vd. EV-4294: 4,34 s trên 5,63 s). |
| Câu bị cắt đôi trước MT: "...we could do is. Maybe if..." (EV-0383) | Hai đoạn cách nhau ~0,6 s đều dài hơn 3 s nên luật gộp (`short_s`) không gộp. |
| "at530 AM" | SenseVoice ITN cần 2 token không có âm thanh riêng ("▁" đứng riêng và ":"); CTC greedy ở 60 ms/frame bỏ đúng 2 token đó (tokenizer: `at 5:30 AM` → `at ▁ 5 : 3 0 ▁AM`). |
| EV-2748 rỗng | Clip hỏng ở nguồn: 77,5 % mẫu bằng 0 tuyệt đối, 128 mức biên độ, trung vị biên độ khác 0 là 9/32767 (≈ 4–5 bit) có noise gate. Không có gain/dither/ngôn ngữ nào cứu được (đã thử). Đây là lỗi thu âm, không phải pipeline. |
| Micro nhạy thấp không kích hoạt `live` | `listen()` đòi RMS > 0,01 (−40 dBFS) tuyệt đối; README ghi micro thật có RMS tiếng nói ~5·10⁻⁴. |
| Thời gian trên máy chậm 3–5 lần profile AI Hub | **[Đọc]** `qai_appbuilder` docs/guide_en.md mục 3.9: phải gọi `SetPerfProfileGlobal()` *sau* khi nạp model đầu tiên; các bản đến bản 2.51.40 đầu tiên bỏ qua lời gọi trước khi nạp. Gói ghim đúng 2.51.40 và gọi trong `_qnn_init` (trước khi nạp) ⇒ rất có thể HTP chưa từng chạy BURST. **[Giả thuyết]** cần đo trên máy. |

## 2. Các thay đổi

### 2.1 Cửa sổ tĩnh nhiều kích thước (bucket) cho cả hai ASR

* `AsrVi` / `AsrXx` nhận danh sách `(cửa sổ, graph)` và chạy cửa sổ **nhỏ nhất đủ chứa câu** (`asr.pick`). Một graph = hành vi cũ.
* `models.json`: khoá gốc là cửa sổ lớn nhất (`asr_vi` = 1500 frame, `asr_xx_encoder` = 500 frame LFR), thêm `asr_vi@<frame>` / `asr_xx_encoder@<frame LFR>`. Ví dụ đầy đủ: `models.dev_asr_buckets.json`.
* Phần đệm bị mask trong encoder (x_lens / speech_lengths) nên transcript **giống hệt** giữa các cửa sổ: Zipformer 25/25 clip, argmax từng frame trùng 100 %; SenseVoice 33/33 câu (`build_zip150_buckets.py --check-wavs`, `export_sensevoice_buckets.py --check-wavs`). Giá trị đệm 0 hay log(1e-10) cho cùng WER (đo trên 250 câu FLEURS-vi) ⇒ giữ 0.
* Trên chip: link tất cả cửa sổ của một model vào **một context binary** (AI Hub link job nhiều model; trọng số trùng tên và nội dung chỉ lưu một lần) và chọn graph bằng `Inference(graphIndex=i)`: khoá `"graph_index"` trong models.json, các khoá cùng `"ctx"` dùng chung một QNNContext (`backends.QnnGraph`).
* Đề xuất cửa sổ: Zipformer 300/500/800/1200/1500 frame (3/5/8/12/15 s); SenseVoice 50/100/200/500 frame LFR (3/6/12/30 s).

### 2.2 SenseVoice: tính đặc trưng trên host (bỏ graph frontend)

* `asr.sv_features` tái tạo đúng `WavFrontend` của FunASR lúc huấn luyện: kaldi fbank trên mẫu nhân 32768 (25/10 ms, hamming, snip_edges), LFR 7/6 với `ceil(T/6)` frame và frame cuối đệm bằng frame fbank cuối, CMVN toàn cục `assets/sensevoice_am.mvn`.
* Graph frontend fp16 đang chạy dùng chuẩn hoá theo frame để không tràn fp16, không trừ DC, và **bỏ frame LFR cuối chưa đủ** (`floor((T−4)/6)+1` thay vì `ceil(T/6)`, mất tới ~60–100 ms cuối câu). Host fbank không có các sai lệch này.
* Chi phí host: ~9 ms cho 10 s âm thanh trên M1 **[Đo]**; bỏ được graph frontend 58 ms trên chip + truyền 464000 mẫu.
* Bật bằng `"_sv_frontend": "host"` trong models.json (hoặc `S2S_SV_FRONTEND=host`); mặc định vẫn là graph nếu models.json có `asr_xx_frontend`.

### 2.3 Zipformer tiếng Việt: giải mã bằng nhánh RNN-T

* Model `hynt/ZipFormer-150M-CR-CTC-RNNT-6000h` được huấn luyện với CR-CTC + CTC + RNN-T; nhóm chỉ dùng nhánh CTC. **[Đọc]** Bài CR-CTC (arXiv 2410.05101) và kết quả icefall: nhánh transducer của model huấn luyện chung tốt hơn nhánh CTC ~10 % tương đối; các WER hynt công bố nhiều khả năng là của RNN-T.
* Graph encoder xuất thêm `encoder_out` (T′, 512) (`build_zip150_buckets.py --with-encoder-out`); host chạy decoder (ONNX 5 MB, onnxruntime CPU, cache theo ngữ cảnh) + joiner (`tanh(enc + dec) @ W + b`, numpy, chấm theo khối 8 frame). Greedy, tối đa 1 ký hiệu/frame như icefall/sherpa-onnx.
* Bật khi models.json có `asr_vi_decoder` + `asr_vi_joiner`; `S2S_VI_DECODE=ctc` để tắt. Thống kê LID vẫn tính từ logits CTC như trước.

### 2.4 VAD / phân đoạn (`audio.segment`)

* **Hysteresis**: đoạn *bắt đầu* ở ngưỡng cũ, nhưng biên được kéo dài qua các frame còn trên ngưỡng nhả (~30 dB dưới đỉnh, tối đa 400 ms) rồi mới đệm 250 ms đầu / 350 ms cuối (không vượt quá giữa khoảng lặng tới đoạn bên cạnh).
* **Gộp** hai đoạn cách nhau < 0,8 s (ngắt trong câu) nếu tổng ≤ giới hạn; giữ luật gộp đoạn ngắn cũ. Chế độ hội thoại **không** dùng luật gộp này (`join_gap_s=0`): hai lượt nói của hai người cách nhau 0,6–1 s phải là hai đoạn để LID định tuyến riêng; giới hạn đoạn ở đây = min(14 s, cửa sổ Zipformer, cửa sổ SenseVoice).
* Giới hạn đoạn 12 s → **14 s** (`S2S_SEG_MAX_S`): Zipformer nhận 14,5 s, SenseVoice 28,5 s; câu 11,5–14 s bị cắt đôi ở 12 s mất chính xác.

### 2.5 Mức âm

* `normalize_level`: đưa **RMS phần có tiếng** (các frame trong 35 dB của frame to nhất) về −24 dBFS, gain tối đa +35 dB; tiếng to hơn đích giữ nguyên tới −14 dBFS, trên đó đưa về −14 dBFS (ánh xạ liên tục). Giới hạn đỉnh dùng phân vị 99,9 % của |x| (một tiếng click không chặn gain), vài mẫu vượt ±1 bị cắt. Cả hai ASR nhạy với mức âm: SenseVoice dùng CMVN toàn cục cố định, Zipformer không có CMVN.
* `listen()`: ngưỡng bắt đầu tuyệt đối 0,01 → 2·10⁻⁴ (`S2S_LISTEN_MIN_RMS`), ngưỡng tương đối `4·floor` giữ nguyên.

### 2.6 HTP perf profile

`backends._apply_perf()` gọi `SetPerfProfileGlobal(BURST)` một lần **sau** khi QNNContext đầu tiên được nạp (thay cho lời gọi trước khi nạp). Ảnh hưởng mọi graph (ASR, MT, TTS). `S2S_PERF` giữ nguyên ý nghĩa.

### 2.7 Sửa ITN giờ tiếng Anh

`asr.fix_itn`: tách chữ dính số sau giới từ chỉ thời gian ("at530" → "at 530") và đổi số 3–4 chữ số trước AM/PM thành h:mm ("530 AM" → "5:30 AM"); không đụng "1530 PM", "530 amps", "10 AM".

## 3. Kết quả đo

Máy đo: MacBook M1 Pro, onnxruntime CPU, graph fp32 (xu hướng tương đối, không phải số trên chip).

**Độ trễ một lần gọi encoder theo cửa sổ** (trung vị 5–7 lần, CPU đa luồng):

| Zipformer (frame / giây) | 300 / 3 s | 500 / 5 s | 800 / 8 s | 1200 / 12 s | 1500 / 15 s |
|---|---|---|---|---|---|
| ms | 60 | 95 | 142 | 193 | 243 |

| SenseVoice (frame LFR / giây) | 50 / 3 s | 100 / 6 s | 200 / 12 s | 500 / 30 s |
|---|---|---|---|---|
| ms | 109 | 180 | 345 | 905 |

Gần tuyến tính theo độ dài (sàn ~20 ms), khớp ước tính FLOPs (Zipformer ≈ tuyến tính; SenseVoice 70 khối SAN-M). Đặc trưng SenseVoice trên host: 4 ms cho 5 s âm thanh (graph frontend 29 s trên CPU: 20 ms; trên chip 58 ms).

**Tốc độ ASR, 10 clip BTC en (Hieu), VAD mới:** cũ (frontend graph + cửa sổ 29 s) trung bình 1196 ms; mới (host features + cửa sổ 50/100/200/500) **437 ms**. Cửa sổ dùng: 50×2, 100×8, 200×1.

**10 clip BTC en (Hieu), toàn bộ đường ASR mới:** WER 24,0 → **21,8 %**. EV-4294 "follower." → "followers."; EV-3775 "Gett me … at530 AM." → "Get me … at 5:30 AM."; EV-0383 không còn bị cắt đôi câu. EV-2748 (clip hỏng) và EV-5269 (câu 10 từ trong 2,4 s) vẫn sai.

**LID vi / không-vi của chế độ hội thoại** (100 clip FLEURS mỗi tiếng, tỷ lệ đúng vi / en): cũ (chuẩn hoá cũ + CTC) 0,99 / 1,0 (nguyên clip), 0,96 / 1,0 (3 s), 0,75 / 1,0 (1,5 s); mới (chuẩn hoá mới + RNN-T) 1,0 / 1,0, 0,96 / 1,0, 0,79 / 1,0 ⇒ không suy giảm.

**Phân đoạn (250 câu ≤ 14 s mỗi tiếng, chuẩn hoá cũ, CTC):**

| | FLEURS-en WER | sai từ cuối | FLEURS-vi WER | sai từ cuối |
|---|---|---|---|---|
| VAD cũ (12 s) | 7,27 % | 10,4 % | 5,30 % | 6,8 % |
| VAD mới, 12 s | 7,05 % | 9,6 % | 4,95 % | 6,4 % |
| VAD mới, 14 s | **6,88 %** | 9,6 % | **4,89 %** | 6,4 % |
| Nguyên clip (trần) | 6,88 % | 8,0 % | 4,83 % | 6,8 % |

Với clip en ≤ 11,5 s: VAD mới 6,43 % so với cũ 6,72 % (nguyên clip 6,58 %).

**Giải mã Zipformer, FLEURS-vi (250 câu, nguyên clip):** CTC greedy 4,81 %; **RNN-T greedy 4,40 %** (17 ms host); RNN-T beam 4: 4,32 % (120 ms, không đáng); đệm 0 hay log(1e-10): như nhau. Qua code production (`eval_asr.py`, `asr_vi_decoder/joiner` trong models.json, ngữ cảnh khởi đầu [-1, blank] và bỏ `<unk>` như icefall): 4,40 %. Pipeline mới đầy đủ (VAD mới 14 s + RNN-T + chuẩn hoá mới): **4,53 %** so với 5,30 % của pipeline cũ.

**Mức âm** (120 câu mỗi tiếng, nguyên clip, hạ gain rồi lượng tử lại 16 bit như micro nhỏ):

| WER | −40 dB | −25 dB | 0 dB |
|---|---|---|---|
| vi, chuẩn hoá cũ / mới | 6,64 / 6,61 % | 5,18 / 5,18 % | 5,15 / 5,15 % |
| en, chuẩn hoá cũ / mới | 63,2 / 62,5 % | 10,9 / 10,8 % | 6,98 / 7,02 % |

* Chuẩn hoá mới **không hơn, không kém** cách cũ trên FLEURS: cả hai đều nâng phần lớn clip (FLEURS-en rất nhỏ: peak trung vị −42 dBFS, 74 % clip có peak < 0,2). Khác biệt chỉ xuất hiện với bản ghi có peak > 0,2 nhưng tiếng nói nhỏ, nơi cách cũ không làm gì; chưa có dữ liệu đo trường hợp này.
* Chỗ hỏng ở −40 dB (en) **không phải do chuẩn hoá**: cùng clip hạ −40 dB ở dạng float (không lượng tử) vẫn cho 6,1 % như 0 dB; sau khi lượng tử 16 bit, 82 % mẫu thành 0 tuyệt đối và tiếng nói mất thông tin (giống clip BTC EV-2748). Dither không cứu được (75 → 69 %). Hướng xử lý là phần cứng: tăng gain micro / ALSA capture trên AIBOX, không phải phần mềm ASR.
* Đệm thêm 300–500 ms nhiễu −70 dBFS quanh clip (khuyến nghị cho SenseVoice của sherpa-onnx) không đổi WER 10 clip BTC (21,8 % cả 3 mức) vì VAD mới đã đệm bằng âm thanh thật ⇒ không thêm vào pipeline.

## 4. Triển khai lên AIBOX 8550

1. Xuất ONNX các cửa sổ (máy dev có torch/funasr):
   ```bash
   python src/step4_quantization/asr/zipformer/step4_hardware/build_zip150_buckets.py --model-dir <thư mục hynt> --out-dir out/zip --frames 300,500,800,1200,1500 --with-encoder-out --check-wavs <wav tiếng Việt>
   python src/step4_quantization/asr/sensevoice/export/export_sensevoice_buckets.py --out-dir out/sv --frames 50,100,200,500 --check-wavs <wav tiếng Anh>
   ```
2. Compile + link trên AI Hub (một context binary cho mỗi model; xem `--dry-run` trước):
   ```bash
   python s2s_edge/tools/build_asr_buckets_aihub.py --model zip --onnx-dir out/zip --frames 300,500,800,1200,1500 --out models_qcs8550
   python s2s_edge/tools/build_asr_buckets_aihub.py --model sv --onnx-dir out/sv --frames 50,100,200,500 --precision fp16 --out models_qcs8550
   ```
   Trước khi chạy: chép tuỳ chọn compile fp16 thật của job Zipformer đang dùng (`hub.get_job('jpv226qjg').options`) vào `--fp16-options` (`--quantize_full_type` đã bị AI Hub đánh dấu deprecated). Độ chính xác SenseVoice fp16 so với w8a16 **chưa đo**: build cả hai, so bằng `golden.py check` + `eval_asr.py` trước khi chọn.
   Script in ra các khoá models.json (`ctx` + `graph_index`). `graph_index` = thứ tự trong link job là **giả định** (AI Hub không ghi rõ thứ tự): kiểm tra trên máy, cửa sổ sai sẽ FAIL ở `golden.py check`. Chép thêm `decoder-epoch-11-avg-2.onnx`, `joiner-epoch-11-avg-2.onnx` vào gói (chỉ cần onnxruntime; nếu không đọc được, ASR tự quay về CTC và in cảnh báo) và khai báo `asr_vi_decoder` / `asr_vi_joiner` (`onnx`), đặt `"_sv_frontend": "host"`.
3. Trên máy: `python tools/test_asr_host.py`, `python tools/golden.py make` (máy dev) rồi `check --backend npu` (máy thật): graph Zipformer có `encoder_out` được chấm bằng token RNN-T greedy NPU so với CPU.
4. Đo lại: `python tools/eval_asr.py --models models/models.json --backend npu --lang en --manifest <BTC> --seg vad` (và `tools/eval_btc.py` toàn trình). So `asr_ms` với bản cũ; kiểm tra log `[s2s] HTP perf profile: BURST (set after the first context load)`.

**Chưa kiểm trên chip** (không có quyền AI Hub/máy thật ở đây): compile/link nhiều graph, `graphIndex` của bản `qai_appbuilder` đang dùng, độ chính xác fp16/w8a16 của các cửa sổ SenseVoice, thời gian thật của RNN-T trên CPU ARM.

## 5. Biến môi trường mới

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `S2S_SV_FRONTEND` | theo models.json | `host` / `graph` |
| `S2S_VI_DECODE` | `rnnt` | `ctc` để dùng nhánh CTC |
| `S2S_SEG_MAX_S` | 14 | đoạn dài nhất gửi ASR (s) |
| `S2S_LISTEN_MIN_RMS` | 2e-4 | ngưỡng tuyệt đối bắt đầu thu của `listen()` |

## 5b. Review

Một vòng review độc lập (2 reviewer: code runtime, công cụ/script) đã chạy; các lỗi đã sửa: phụ thuộc `onnx` trên máy (nay chỉ cần onnxruntime, có fallback CTC), ngữ cảnh khởi đầu RNN-T và `<unk>`, giới hạn đoạn + không gộp lượt nói trong chế độ hội thoại, ánh xạ mức âm không liên tục và click chặn gain, đoạn cắt chỉ có nhiễu, warmup thiếu cửa sổ, `golden.py` với cửa sổ khác 1500 và thứ tự output, node Constant `-inf` không được kẹp trong export SenseVoice, tên graph khi link trên AI Hub. Còn lại (chưa làm): `tools/lid_features.py` chưa hỗ trợ chế độ host; huấn luyện lại LID trên đặc trưng mới (đã kiểm tra: không suy giảm).

## 6. Đã nghiên cứu, chưa làm (xếp theo lợi ích / chi phí)

1. Đo trên chip: BURST, `graphIndex`, fp16 vs w8a16 cho SenseVoice (w8a16 không chắc nhanh hơn trên v73).
2. LM n-gram âm tiết tiếng Việt / KenLM tiếng Anh khi giải mã (5–15 % tương đối theo tài liệu), mở đường cho hotword (CTC-WS, arXiv 2406.07096). SenseVoice cần xuất top-k thay vì byte.
3. VAD mô hình Silero v6 (ONNX, < 1 ms/khối) để xác nhận tiếng nói trong môi trường ồn.
4. Rút ngắn thời gian chờ im lặng cuối câu của `listen()` (0,9 s) — đánh đổi với cắt giữa câu.

## 7. Nguồn

CR-CTC arXiv 2410.05101; icefall RESULTS.md (librispeech zipformer); qai_appbuilder docs/guide_en.md §3.3, §3.9; AI Hub link jobs & submit_link_job docs; QNN HTP backend (weight sharing); FunASR wav_frontend.py; Silero VAD utils_vad.py; CTC-WS arXiv 2406.07096.
