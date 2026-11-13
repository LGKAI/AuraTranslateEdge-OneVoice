# Tổng hợp kỹ thuật trong pipeline AuraTranslate Edge (bản gộp, 13/11/2026)

Hệ thống phiên dịch giọng nói offline vi ↔ en / zh / ko trên **HSPTEK AIBOX 8550** (QCS8550, NPU Hexagon v73, QAIRT 2.51). Tài liệu này gom mọi kỹ thuật đang có trong nhánh `integrate/aibox8550-full`, theo từng tầng của pipeline. Với mỗi kỹ thuật ghi: làm gì, file nào, số đo, và đo ở đâu.

**Quy ước "đo ở đâu":** **[chip]** = đo trên hộp thật hoặc trên QCS8550 của AI Hub; **[laptop]** = đo trên CPU với cùng thuật toán; **[test]** = chỉ có unit test, chưa có số chất lượng. Số đo lấy từ các báo cáo gốc liệt kê ở mục 9. Chỗ nào chưa đo thì ghi rõ là chưa đo.

```
Mic / WAV 16 kHz → VAD (host) → ASR trên NPU (Zipformer vi | SenseVoice en/zh/ko) → lọc rác ASR → dấu câu + chữ hoa tiếng Việt
 → MT trên NPU (NLLB-3.3B hoặc 600M theo chiều, dịch từng câu) → lọc đầu ra MT → phiên âm tên, kính ngữ tiếng Hàn
 → chuẩn hoá chữ trước TTS (hybrid) → TTS trên NPU (Piper vi | MeloTTS zh | Supertonic en/ko) → loa / web
```

## 1. Nền tảng chạy trên NPU

| Kỹ thuật | Làm gì | File | Kết quả |
|---|---|---|---|
| Context binary QAIRT 2.51 | 9 đồ thị (cộng 4 đồ thị NLLB-3.3B) được compile và link trên AI Hub cho "QCS8550 (Proxy)", khớp runtime trong wheel `qai_appbuilder` 2.51.40 | `models_qcs8550/*.bin`, `backends.py` | cả 9 đồ thị nạp được và chạy đúng trên hộp: golden 8/8 PASS **[chip]** |
| Dùng thư viện QAIRT trong wheel thay vì của hộp | hộp cài sẵn QAIRT 2.41 trong `/usr/lib`; `tools/env.sh` đặt `LD_LIBRARY_PATH` / `ADSP_LIBRARY_PATH` để bản 2.51 trong wheel được dùng trước | `tools/env.sh`, `backends._qnn_init` | sửa lỗi stub 2.51 gặp skel 2.41 **[chip]** |
| Chạy bằng user `system` | trên image này root không mở được `/dev/adsprpc-smd`; `start.sh` và `start_box.sh` tự chạy lại bằng `runuser -u system` | `start.sh`, `tools/start_box.sh` | **[chip]** (đường đi đã kiểm trên hộp 12/11; `start.sh` kiểm thêm trong container arm64) |
| Sửa đầu ra số nguyên | chip trả số nguyên nhỏ (độ dài, token id) dưới dạng float32 mang mẫu bit int32; đọc kiểu thật từ file golden rồi đổi lại | `backends.fix_int_output`, `registry.graph` | hết ASR rỗng và audio dài 0 **[chip]** |
| Lượng tử hoá đầu vào SenseVoice | encoder w8a16 nhận fbank kiểu UFIXED_16: host lượng tử hoá trước với scale/offset ghi trong `models.json` (`in_quant`) | `backends.QnnGraph.run` | encoder hết "nghe thấy nhiễu" **[chip]** |
| Thứ tự đầu ra MeloTTS | chip trả (độ dài, audio) thay vì (audio, độ dài): chọn đầu ra theo kích thước và kiểu dữ liệu | `tts.split_outputs` | hết câm ở đầu ra tiếng Trung **[chip]** |
| HTP perf profile BURST sau lần nạp đầu | `SetPerfProfileGlobal` phải gọi sau context đầu tiên, nếu gọi trước có thể bị bỏ qua | `backends._apply_perf` | chưa đo lại trên hộp |
| Giữ onnxruntime thật | `qai_appbuilder` thay module `onnxruntime`; các model CPU (dấu câu, GTCRN, RNN-T decoder) dùng bản thật qua `cpu_ort()` | `s2s_edge/__init__.py` | **[chip]** (dấu câu chạy trên hộp 12/11) |
| Kiểm golden | mỗi đồ thị chạy trên đầu vào cố định, so với đầu ra CPU (cosine, argmax); có đối chứng âm | `tools/golden.py`, `tools/session0.sh` | 8/8 PASS **[chip]** |

## 2. VAD và mức âm

| Kỹ thuật | Làm gì | Số đo |
|---|---|---|
| VAD năng lượng tương đối mức thu (mặc định cũ `legacy`) | ngưỡng theo mức của chính bản ghi, chạy được với mic nhỏ (RMS 5e-4) | — |
| **VAD hysteresis (`hybrid`, mặc định mới)** | bắt đầu ở ngưỡng cũ, kéo biên qua các frame còn trên ngưỡng nhả (~30 dB dưới đỉnh, ≤ 400 ms), đệm 250 ms đầu / 350 ms cuối, gộp các ngắt ngắn, đoạn tối đa 14 s | WER FLEURS-en 7,27 → **6,88 %**, FLEURS-vi 5,30 → **4,89 %** (250 câu mỗi tiếng) **[laptop]** |
| Gợi ý ngắt câu từ VAD | khoảng lặng ≥ 320 ms trong đoạn được chuyển thành vị trí dấu câu cho tiếng Việt | trong số đo ASR tiếng Việt ở trên |
| Chuẩn hoá mức âm | đưa RMS phần có tiếng về −24 dBFS (gain tối đa +35 dB) | WER như cách cũ ở 0 / −25 / −40 dB **[laptop]** |

Đổi về VAD cũ bằng `S2S_VAD_PROFILE=legacy` (VAD `hybrid` chưa chạy trên hộp).

## 3. ASR

| Kỹ thuật | Làm gì | File | Số đo |
|---|---|---|---|
| Zipformer-150M CTC fp16 (vi) | model ASR tiếng Việt trên NPU; bản w8a16 kém nên giữ fp16 | `asr.AsrVi` | WER 7,5 / 9,5 / 6,2 %, 174 ms **[chip]** |
| SenseVoice-Small w8a16 (en/zh/ko) | một model cho ba ngôn ngữ | `asr.AsrXx` | ADV CPU: en WER 7,7 → 17,6 % ở 5 dB **[laptop]** |
| **Cửa sổ tĩnh nhiều kích thước** | đồ thị cho nhiều độ dài (Zipformer 100–1500 frame, SenseVoice 50–500); câu ngắn dùng cửa sổ nhỏ; transcript giống hệt | `registry.windows`, `asr.pick` | câu 3 s: Zipformer 243 → **60 ms**, SenseVoice 905 → **109 ms**; 10 clip BTC en: ASR 1196 → **437 ms** **[laptop, ước tính theo profile]**. Cần compile thêm đồ thị trên AI Hub (`--asr-config`); gói hiện chạy 1 cửa sổ như cũ |
| **Đặc trưng SenseVoice trên host** | tái tạo đúng `WavFrontend` của FunASR (fbank, LFR 7/6, CMVN), bỏ đồ thị frontend fp16 (bị mất frame LFR cuối) | `asr.sv_features` | ~9 ms cho 10 s âm thanh; bỏ được 58 ms trên chip **[laptop]**; bật bằng `"_sv_frontend": "host"` |
| **Nhánh RNN-T của Zipformer** | model được huấn luyện CR-CTC + RNN-T; giải mã greedy bằng decoder ONNX trên CPU và joiner bằng numpy | `asr.Transducer` | FLEURS-vi WER 4,81 → **4,40 %**, +17 ms **[laptop]**; cần đồ thị có `encoder_out` (chưa compile) |
| Sửa ITN giờ tiếng Anh | "at530 AM" → "at 5:30 AM" | `asr.py` | 10 clip BTC en: WER 24,0 → **21,8 %** (cả đường ASR mới) **[laptop]** |
| Nhận dạng ngôn ngữ (chế độ hội thoại) | hồi quy logistic trên độ tự tin của Zipformer (vi / không vi), rồi chữ viết của SenseVoice (Hangul / Hán / Latin) | `conversation.LanguageID` | 10-fold 100 % trên 120 clip sạch; sai khi câu 1,5 s ở 5 dB **[laptop]** |
| **Lọc rác ASR** (`clean_asr`) | bỏ nhãn lạc "Q. "/"H. "/"问：", "Um." đầu câu, thán từ đứng thành câu riêng ở đầu/cuối khi nhiễu ("嗯。", "그.", "没。" cuối câu), chuỗi lặp ≥ 4 lần; giữ "Có.", "好。", "네.", "-5 độ", "Q1", "H.264" | `asr_clean.py` | 22/1.256 câu ASR thật bị đổi, không câu nào cắt nhầm; chrF 21 câu đó 68,5 → 72,7 (p ≈ 0,33, chưa có ý nghĩa thống kê) **[laptop, chữ ASR chip + ADV CPU]** |

## 4. Chuẩn bị chữ nguồn

| Kỹ thuật | Làm gì | Số đo |
|---|---|---|
| **Dấu câu + chữ hoa tiếng Việt** (vibert-capu, ONNX int8 111 MiB) | Zipformer ra chữ thường không dấu câu; model thêm `, . ? :`, viết hoa, đổi số đọc thành chữ số | chrF MT trên chữ ASR thật (30 clip): vi→ko 25,6 → **34,1**, vi→zh 31,0 → **36,3** **[laptop]** |
| Luật "?" cuối câu (dự phòng) | dùng khi thiếu model (`S2S_VI_PUNCT=1`) | — |
| Glossary `_asr_fix` | sửa cách ASR viết sai một thuật ngữ | — |
| Chuẩn hoá nguồn bằng LLM (`source_mode=1`) | Qwen qua MLX, **chỉ chạy trên Mac**; trên hộp tự quay về cách trên và in cảnh báo | — |

## 5. MT

| Kỹ thuật | Làm gì | File | Số đo |
|---|---|---|---|
| NLLB-600M tách encoder + 1 bước decoder | encoder 64 token tĩnh + đồ thị một bước giải mã với KV cache tĩnh; vòng lặp chạy trên host (bỏ đồ thị trải 16 bước nặng 8,8 GB) | `mt.NllbSplit` | enc 7,4 ms, greedy 16 ms/bước, beam 31 ms/bước **[chip]**; chất lượng = HF fp32 |
| Beam search 4 trên host | chặn lặp 3-gram, chặn chạy vô hạn (≤ 2,2 × nguồn + 14 token) | `mt.NllbBeam` | vi→en +2,4 chrF so với greedy **[laptop]** |
| `MtFallback` | thiếu bộ nhớ / lỗi NPU / đầu ra sai bố cục thì giải phóng đồ thị, hạ beam → greedy, dịch lại câu đó | `mt.MtFallback` | đã thử với lỗi giả lập **[test]** |
| **NLLB-200 3.3B trên NPU** | tách 4 đồ thị w8a16 (encA/encB/decA/decB, beam 4), embedding chạy trên host | `mt33.py`, `models_qcs8550/nllb33_*.bin` | profile QCS8550: enc 31 ms, 107 ms/bước, không lỗi bộ nhớ **[chip-proxy]**; chất lượng trên hộp thật **chưa đo** |
| **Route theo chiều** | 3.3B cho vi↔zh, vi↔ko; 600M cho en↔vi (`mt_route`) | `mt.MtRouter`, `models.json` | 3.3B so với 600M (BTC, chữ ASR chip, đã tách câu như pipeline): vi→zh **+6,5** (KTC +2,5..+10,6), zh→vi **+3,4**, ko→vi **+3,3**, vi→ko −2,5 (không chắc chắn; giữ 3.3B vì sửa lỗi đồng âm, rubric 10/10 so với 6/10), en→vi +0,6 / vi→en +2,5 (không chắc chắn → giữ 600M vì nhanh hơn) **[laptop, 3.3B int8]** |
| Lùi về 600M có báo động | 3.3B không nạp được thì in `!! MT DEGRADED`, dải đỏ trên web; `S2S_MT_STRICT=1` để báo lỗi thật | `mt.MtRouter` | **[test]** |
| Tách câu giữ nguyên số | lớp MT dịch từng câu; dấu `. ! ? ;` chỉ là hết câu khi theo sau là khoảng trắng; không tách sau "Mr.", "a.m.", "TP." | `mt.split_sentences` | sửa lỗi cắt "15.000 đồng", "36.5", "1,000", "7:30" **[test]** |
| Câu quá dài | > 62 token hoặc giải mã chưa kết thúc: cắt theo câu → mệnh đề → chia đôi, đệ quy; đếm `truncations` | `mt.Nllb._tr` | **[test]**; luôn cắt theo dấu phẩy thì kém hơn (48,2 → 44,0) nên không dùng |
| Một đường MT cho mọi chỗ gọi | CLI, stream, web hộp, web demo, eval cùng gọi `translate_text_tagged` | `conversation.py` | — |
| **Lọc đầu ra MT** (`clean_mt`) | bỏ ký hiệu khối "❚ ‖", "问:" chép từ nguồn, gạch / dấu câu đầu câu, "。" trong câu Việt/Anh; chỉ toàn dấu câu thì không đọc | `asr_clean.py` | 2/658 câu đổi mỗi model, chrF tổng không đổi **[laptop]** |
| Phiên âm tên Latin trong zh/ko | MT dịch riêng tên (en → đích), có cache; glossary ghi đè | `names.py` | — |
| **Kính ngữ tiếng Hàn** | đổi đuôi banmal sang -요 ở từ cuối câu (어때 → 어때요, 갈까 → 갈까요, 하지 마 → 하지 마세요…); giữ -니다/-니까 | `ko_polite.py` | 5/5 câu đổi gần đáp án hơn; 167/180 câu vi→ko vốn đúng **[laptop]** |

## 6. TTS

| Kỹ thuật | Làm gì | File | Số đo |
|---|---|---|---|
| Piper (vi), MeloTTS (zh), Supertonic (en/ko) trên NPU | giữ độ chính xác cao cho vocoder / bước ODE (không INT8 thuần) | `tts.py` | chạy được trên hộp **[chip]** |
| **Chuẩn hoá chữ trước TTS kiểu hybrid** (mặc định) | parser có kiểu tạo span (giờ, ngày, tiền, %, đơn vị, số âm/thập phân, mã, ZIP, điện thoại), sinh lời đọc theo từng ngôn ngữ: giờ Hàn dùng số thuần Hàn, năm Trung đọc từng số, mã/điện thoại đọc từng ký tự và giữ số 0 đầu; CRF (tuỳ chọn) chỉ phân loại phần mơ hồ | `tts_format.py`, `hybrid_textnorm.py`, `identifiers.py`, `format_locales.py` | 55/55 ví dụ có lời đọc kỳ vọng, 1.600 ca sinh tự động, 49/49 bản dịch en-vi tạo WAV; 0,23 ms/câu **[laptop]**. Sửa "5:30" → "năm giờ ba mươi phút", "08:00" → "tám giờ", ZIP đọc từng số, KO "3:30" → "세 시 삼십 분", ZH "2027" → "二零二七" |
| Chia đoạn đúng giới hạn đồ thị | không cắt giữa từ Latin/Hangul; giới hạn Supertonic tính sau NFKD và token ngôn ngữ | `tts.py` | hết cắt âm cuối |
| Đọc từ ngoài bảng của Piper | từ Latin được viết lại bằng âm tiết Việt (CMUdict hoặc luật chữ cái), glossary văn phòng 34 thuật ngữ | `names.latin_to_vi`, `glossary_enterprise.json` | "file" hết thành "phai lơ" (mới kiểm ở mức văn bản) |
| Phát gối đầu theo câu (live) | phát câu 1 trong lúc dịch câu 2 | `conversation.translate_stream` | thời điểm bắt đầu nghe giảm khoảng một nửa **[laptop]** |

## 7. Giao diện và vận hành

| Thành phần | Việc |
|---|---|
| `bash start.sh demo / box / check / live` | một lệnh: tự chuyển root → `system`, cài offline lần đầu, kiểm NPU, chạy web hoặc live |
| Web demo (`web_demo/`) | mic, tải file, **sửa chữ nhận dạng rồi dịch lại** (`POST /api/text`), hiện thời gian từng bước, engine đang chạy (3.3B / 600M / DEGRADED) |
| Web điều khiển trên hộp (`s2s_edge/webui.py`) | chạy BTC (920) / ADV (600) theo lô, gắn thẻ lỗi, chỉ số TTS bằng ASR nghe lại, xuất zip (`results.csv`, `report.html`, `out/*.wav`) |
| Cài offline | `install.sh` + wheelhouse cp310/cp312 + `qai_appbuilder`; venv không cần `ensurepip`; gói tuỳ chọn cài khi có wheel |
| Công cụ laptop | `laptop_push.py` (adb + checksum), `laptop_web.py [--demo]`, `junk_report.py` (tìm rác mới sau mỗi lần chạy chip), `sync_package.py` |

## 8. Đã thử và loại (có số)

| Thử | Kết quả |
|---|---|
| NLLB-1.3B | 920 câu: không hơn 600M (43,7 so với 44,0; en→vi −2,7) |
| Dịch bắc cầu qua tiếng Anh | kém dịch thẳng 3–10 chrF |
| Xếp hạng lại n-best (noisy channel) | dev có lợi nhưng holdout không (43,39 → 43,66 / 43,35) |
| GTCRN khử nhiễu trước ASR | không cải thiện |
| Luôn cắt câu dài theo dấu phẩy | chrF 48,2 → 44,0 (tốt hơn 16 / kém hơn 36) |
| Gộp thán từ lặp ("그래, 그래.") | đáp án BTC cũng lặp; vi→ko 26,74 → 26,69 |
| Dấu câu toàn khổ tiếng Trung | chrF không đổi |
| RNN-T beam 4 | 4,32 % so với greedy 4,40 %, nhưng chậm hơn 7 lần |

## 9. Chưa kiểm trên hộp thật và nguồn số liệu

**Chưa kiểm trên hộp thật:**
- Chất lượng và thời gian của NLLB-3.3B w8a16 trên NPU; RAM khi nạp cùng ASR/TTS.
- VAD `hybrid` và chuẩn hoá TTS hybrid trên hộp.
- Cửa sổ ASR nhiều kích thước và RNN-T (cần compile thêm đồ thị trên AI Hub).
- Rác ASR do ASR w8a16 trên chip sinh ra (chạy `junk_report.py` sau lần chạy đầu).
- Mic thật trên web demo.

**Nguồn số liệu chi tiết:** `asr_optimization.md` (ASR/VAD), `results_hybrid_format/README.md` và `hybrid_format_plan.md` (TTS hybrid), `quality_improvements_summary.md` (dấu câu, fallback, TTS), `mt33_routing.md` và `THAY_DOI_VA_CACH_CHAY.md` (MT 3.3B, route, lọc rác, tách câu, câu dài), `chip_results.md` (số đo trên chip), `aibox8550_integration.md` (gộp nhánh), `HUONG_DAN_CHAY_BOX.md` (cách chạy).
