# MeloTTS tiếng Trung trên NPU Dragonwing IQ-9075

Đưa MeloTTS-ZH (TTS tiếng Trung, 44.1 kHz) lên NPU Hexagon của **Dragonwing IQ-9075 EVK** qua Qualcomm AI Hub.
Kết quả mới nhất: **chữ (byte UTF-8) → âm thanh 44.1 kHz chạy trong MỘT graph trên NPU**, kể cả phần chữ → phoneme
(thay bằng bảng tra tĩnh, không còn chạy jieba/pypinyin trên CPU). Chấm bằng ASR: **CER 6.4% trên 40 câu, bằng bản CPU (6.0%)**.

## Bản mới: một graph, 100% NPU (`melo_one`)

| | Giá trị |
|---|---|
| Đầu vào | byte UTF-8 của câu, `int32 [1,112]` (tối đa 24 chữ Hán) |
| Đầu ra | audio 44.1 kHz `[1,196608]` + số frame |
| Layer trên NPU | **3451/3451** |
| Thời gian chip | **720 ms** (luôn tính đủ 384 frame ≈ 4.5 s audio) |
| Nạp lần đầu | **0.9 s** sau khi link thành context binary (bản DLC chưa link: 514 s) |
| CER 8 câu cũ / 32 mệnh đề FLEURS / tổng 40 | **2.6% / 7.4% / 6.4%** |
| Cùng 40 câu trên CPU (front bảng tĩnh / front gốc MeloTTS) | 6.0% / 6.4% |

Trong graph có: front bảng tĩnh → BERT 22 lớp (embedding bằng one-hot × bảng) → mở rộng theo số phoneme → VITS → vocoder cửa sổ.
Audio nghe thử: `results/wav_one_graph_npu/`. Job: `docs/ket_qua_chip.md`.

### Front bảng tĩnh (thay cho jieba + pypinyin + biến điệu chạy trên CPU)

Chưng cất từ chính MeloTTS: chạy front gốc trên khoảng 490 nghìn câu tiếng Trung (OPUS-100), ghi lại cách đọc từng chữ theo ngữ cảnh.
- 7335 token (7321 chữ Hán, 7 dấu câu, 5 đặc biệt), mỗi chữ một cách đọc mặc định.
- 69 nghìn mục ghi đè theo ngữ cảnh (chữ trước/sau, 2 chữ trước/sau, thanh của chữ bên cạnh), áp theo thứ tự ưu tiên.
- Trong graph: so khớp byte UTF-8 → nén thành dãy chữ → tra mặc định + ghi đè → đặt phoneme/thanh bằng cumsum và so sánh. Không có gather động, không vòng lặp.
- File: `melo_front.py`, bảng `tables/front_tables.pkl` (2.8 MB), dựng lại bằng `g2p_get_opus.py` → `g2p_corpus.py` → `front_tables.py`.

Độ khớp với front gốc (84 mệnh đề FLEURS test, CPU): id phoneme 83/84 câu, thanh điệu 73/84, ma trận mở rộng 84/84, token BERT 84/84.
Trên chip, đầu ra front khớp bản torch 40/40 câu.

**Điểm yếu (nói thẳng):**
- Biến điệu thanh 3 phụ thuộc ranh giới từ (vd. 手写) nên bảng đoán sai thanh ở khoảng 13% câu; ảnh hưởng CER nhỏ (6.0% so với 6.4%).
- **Chữ số, chữ Latin, ký tự ngoài 7321 chữ bị bỏ qua** (front gốc đổi số bằng cn2an, bảng chưa làm). Câu có số phải viết bằng chữ Hán.
- Tối đa 24 chữ Hán / 96 phoneme / 4.5 s mỗi lần; câu dài phải cắt trước.

### Lỗi đã gặp khi làm bản này
- **Id token > 2048 không chính xác trong fp16**: tính `hi*128+lo` (tới 7334) rồi so sánh làm BERT trên chip lệch (cosine 0.74, CER 14.8%).
  Sửa bằng so sánh `hi` và `lo` riêng (mỗi số < 2048) → cosine 0.99997.
- **Cùng một VITS, compile cũ (`g5`, tháng trước) cho kết quả sai trên câu dài hơn**: chuỗi graph A → VITS `g5` cho CER 13.0%.
  Tách từng tầng: `enc_p` chạy riêng trên chip khớp CPU tuyệt đối (cosine 1.0 cả 6 lớp), nhưng trong graph lớn `m_p` chỉ còn cosine 0.6.
  Graph gộp compile mới (`melo_one`) không bị lỗi này (CER 6.4%). Nguyên nhân gốc trong trình biên dịch chưa rõ — phải luôn chấm lại chất lượng trên chip sau mỗi lần compile.
- **Link job** (`submit_link_job`) dựng context binary ở máy chủ → nạp lần đầu giảm từ 432–514 s xuống 0.7–0.9 s, thời gian chạy không đổi.

## Bản cũ: 2 graph (BERT, rồi VITS)

Chuỗi **chữ → âm thanh chạy bằng 2 graph trên NPU** (BERT, rồi VITS), chấm bằng ASR cho **CER 4.6%** trên 8 câu (front chạy trên CPU).

## Kết quả (đo thật trên chip)

Chấm bằng SenseVoice fp32 nhận dạng lại audio, so với chữ đầu vào (8 câu tiếng Trung ngắn).

| Cấu hình | CER | Ghi chú |
|---|---|---|
| CPU fp32, model gốc MeloTTS | 2.6% | `results/cpu_baseline_fp32.json` |
| CPU, bản tĩnh (duration xác định, nhiễu cố định) | 4.2% | tương đương bản tĩnh dùng trên chip |
| **NPU: VITS (đặc trưng BERT từ CPU)** | **4.6%** | 2191/2191 layer NPU, **702 ms** |
| **NPU: BERT → (máy chủ mở rộng) → VITS** | **4.6%** | chuỗi 2 graph hoàn toàn trên NPU |

Hai graph trên chip:

| Graph | Layer trên NPU | Thời gian chip | Nạp model lần đầu |
|---|---|---|---|
| BERT (22 lớp, 32 token) | 690/690 | 10.7 ms | 16 s |
| VITS (encoder, duration, flow, vocoder) | 2191/2191 | 702 ms | 432 s |

Đặc trưng BERT chạy trên NPU khớp CPU với cosine ≥ 0.99993. Audio nghe thử: `results/wav_chain2_npu/` (44.1 kHz).
Job ID từng lần chạy: `docs/ket_qua_chip.md`.

## Luồng dữ liệu (bản cũ 2 graph)

```
chữ ──(máy chủ: chuẩn hoá, pinyin, id token)──► bert_ids [1,32], phones/tones [1,96]
bert_ids ──► [NPU graph 1: BERT 22 lớp] ──► h [32,1024]
h ──(máy chủ: nhân h^T với ma trận 0/1 E theo số phoneme mỗi chữ)──► bert [1,1024,96]
phones, tones, bert ──► [NPU graph 2: encoder → duration → căn chỉnh → flow → vocoder cửa sổ] ──► wav 44.1 kHz
```

Kích thước tĩnh: tối đa **96 phoneme** (khoảng 22 chữ) và **384 frame** (≈ 4.5 giây), BERT tối đa 32 token.
Câu dài hơn phải chia ở máy chủ. Chip luôn tính đủ 384 frame nên thời gian chip gần như không đổi.

## Điều chỉnh so với MeloTTS gốc

- Dùng bộ dự đoán độ dài **xác định** (`sdp_ratio=0`) thay cho `0.2`, nhiễu **cố định**. Làm CER trên CPU tăng từ 2.6% lên 4.2%.
- Độ dài và căn chỉnh viết bằng `cumsum` + so sánh (không dùng `NonZero`/`Range`).
- Vocoder chạy theo **cửa sổ 64 frame (chồng 12 frame mỗi bên)** vì chạy cả 384 frame một lần làm chip hết bộ nhớ.
  Kết quả cửa sổ giống bản đầy đủ (cosine 1.00000 trên CPU).

## Những thứ không chạy được (nói thẳng)

| Thử | Kết quả |
|---|---|
| Vocoder đủ 384 frame một lần | chip báo hết bộ nhớ (`QNN_COMMON_ERROR_MEM_ALLOC`) |
| **Gộp BERT + VITS thành một graph** | chip báo lỗi chung (`QNN_COMMON_ERROR_GENERAL`) khi chạy, nên dùng 2 graph |
| Đưa phép mở rộng `h^T @ E` vào graph VITS | chạy được nhưng **CER 18%** (tệ hơn 4.6%); phép nhân riêng lẻ trên chip lại đúng tuyệt đối, nên lỗi do cách graph được biên dịch khi gộp. Vì vậy phép này làm ở máy chủ |
| **Lượng tử hoá W8A16 và W16A16** | quantize và compile thành công, nhưng chip **không nạp được** (`MODEL_GRAPH_ERROR`) nên không chạy được |

Vì vậy bản triển khai là **fp16** (không lượng tử). Model chỉ có 52M tham số, khoảng 104 MB fp16, nên lượng tử không cần để giảm dung lượng.
Chưa tìm ra op nào gây `MODEL_GRAPH_ERROR` ở bản lượng tử; để thử tiếp có thể lượng tử riêng vocoder.

## Đo trên mẫu lớn hơn (40 mệnh đề FLEURS, CPU) — chỉnh nhiễu để tăng chất lượng

8 câu quá ít để kết luận, nên đo thêm 40 mệnh đề (`step10_variants.py`, `results/cpu_variants_40clauses.json`):

| Cấu hình | CER trung bình | Số câu có lỗi |
|---|---|---|
| Model gốc (`sdp_ratio=0.2`, nhiễu 0.6 ngẫu nhiên) | 6.8% | 18/40 |
| Bản tĩnh, nhiễu 0.6, seed 0 (**đang dùng trên chip**) | 9.1% | 21/40 |
| Bản tĩnh, nhiễu 0.6, seed 1 | 7.5% | 20/40 |
| Bản tĩnh, nhiễu 0.667, seed 0 | 9.2% | 21/40 |
| **Bản tĩnh, nhiễu 0.3, seed 0** | **6.6%** | 20/40 |

Kết luận: lỗi phát âm phân tán ở từ khó chứ không có lỗi hệ thống, và hạ nhiễu xuống 0.3 giúp bản tĩnh bằng bản gốc.
Chưa chạy lại trên chip với nhiễu 0.3 (graph trên chip hiện dùng 0.6); đổi `noise_scale` trong `melo_static.py` rồi xuất lại.

## Giới hạn

- Mới thử 8 câu ngắn, một giọng (ZH), không có đánh giá nghe chủ quan (MOS).
- Bản cũ 2 graph: máy chủ còn làm chuẩn hoá chữ, pinyin/id, ma trận E, phép `h^T @ E`. **Bản `melo_one` đã đưa hết vào graph**; máy chủ chỉ đổi chữ ra byte UTF-8 và cắt audio theo số frame.
- AI Hub chỉ chạy từng graph, không chạy mã tuỳ ý. Mỗi job thật mất khoảng 10–45 phút vì nạp model và hàng đợi.
- Số frame trên chip lệch khoảng 2–7% so với CPU do làm tròn fp16 ở bộ dự đoán độ dài (nhịp đọc hơi nhanh hơn).

## Chạy lại

Yêu cầu: Python 3.11, `pip install -r requirements.txt`, `qai-hub` đã cấu hình, và `git clone https://github.com/myshell-ai/MeloTTS` vào thư mục này (tên `MeloTTS`).
Model và BERT (`hfl/chinese-roberta-wwm-ext-large`, khoảng 1.3 GB) tự tải về lần đầu. Chạy từ thư mục `melotts_npu/` (file sinh ra ở `work/`).
Chấm điểm cần SenseVoice combined ONNX (biến `SV_ONNX`, xem nhánh `sensevoice`).

```bash
python step0_cpu_baseline.py                        # mốc CPU + CER
python step2_cpu_static_parity.py                   # bản tĩnh khớp bản gốc trên CPU
python step3_export.py 96 384 5                     # xuất VITS tĩnh (cửa sổ, nhóm 5) -> work/melo_N96_Y384_g5.onnx
python hub_melo.py g5 work/melo_N96_Y384_g5.onnx 96 384     # compile + inference + profile trên chip, CER
python step6_bert_only.py                           # xuất BERT 22 lớp (32 token) -> work/bert_L32.onnx
python hub_generic.py bert work/bert_L32.onnx work/bert_inputs.npz
python step9_chain_hostexpand.py                    # chuỗi BERT(NPU) -> máy chủ -> VITS(NPU), CER

# Bản 1 graph, 100% NPU
python step12_front_parity.py                       # front bảng tĩnh khớp front gốc (CPU)
python step13_front_cer.py                          # CER: front gốc vs front bảng (CPU)
python step14_export_A.py                           # front + BERT + mở rộng -> work/graphA.onnx
python step3_export.py 96 384 5                     # VITS -> work/melo_N96_Y384_g5.onnx
python step16_merge_one_graph.py                    # ghép -> work/melo_one.onnx (1.31 GB)
python hub_generic.py melo_one work/melo_one.onnx work/melo_one_inputs.npz   # compile + chạy thử
python step17_one_graph_linked.py                   # link context binary, 40 câu, CER, profile
```

## Các file

| File | Việc |
|---|---|
| `melo_static.py` | VITS tĩnh một khối + vocoder cửa sổ |
| `melo_full.py` | BERT tĩnh, ghép BERT+VITS, và `host_front` (chữ → id, ma trận E) |
| `melo_front.py` | Front bảng tĩnh (byte UTF-8 → phoneme, thanh, ma trận E, token BERT) |
| `melo_graphA.py` | Graph A = front + BERT (embedding one-hot) + mở rộng |
| `g2p_*.py`, `front_tables.py` | Chưng cất front MeloTTS thành bảng tra |
| `step12`–`step17`, `probe_*.py` | Kiểm tra front, ghép 1 graph, link, tách tầng tìm lỗi |
| `datasets_zh.py` | Dữ liệu hiệu chuẩn lượng tử từ FLEURS tiếng Trung |
| `step0`–`step9`, `hub_*.py` | Kiểm tra CPU, xuất ONNX, chạy trên chip, chấm điểm |
| `logs/chip/` | Log chạy trên chip (đã lọc) |
| `docs/` | Nhật ký job, giải thích |
