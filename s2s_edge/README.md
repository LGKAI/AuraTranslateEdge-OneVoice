# s2s_edge — Hệ thống Phiên dịch Giọng nói hai chiều Offline (vi ⇄ en / zh / ko) cho HSPTEK AIBOX 8550 (QCS8550)

Nguyên mẫu Phase 3, Cuộc thi OneVoice AI Challenge 2026. Nhận âm thanh đầu vào → Dịch và phát âm thanh đầu ra, **toàn bộ quá trình suy luận thực thi trực tiếp trên thiết bị biên (On-device), 100% Offline, không phụ thuộc kết nối mạng**.
Hỗ trợ 6 chiều dịch: vi→en, vi→zh, vi→ko, en→vi, zh→vi, ko→vi (một đầu luôn là Tiếng Việt).

> **Hiện trạng thực tế:** Mã nguồn pipeline chạy toàn trình (End-to-End) hỗ trợ cả 2 backend: Backend NPU tăng tốc phần cứng (`--backend npu`) và Backend tham chiếu CPU (`--backend cpu` qua ONNX Runtime). Cả 9 đồ thị mô hình đã được biên dịch và link thành QNN Context Binary cho NPU Hexagon v73 trên thiết bị **HSPTEK AIBOX 8550 (SoC Qualcomm QCS8550)** qua Qualcomm AI Hub với QAIRT 2.51. Hệ thống đã được nạp và đo kiểm thực tế trên thiết bị phần cứng AIBOX 8550 tại cơ sở VNG Campus.
>
> **Chạy trên hộp AIBOX 8550 (gói đã gộp, web điều khiển, MT 3.3B, lọc rác ASR/MT): xem `../HUONG_DAN_CHAY_BOX.md`** ở thư mục gốc repo. Sơ đồ file và lệnh chi tiết: `docs/HUONG_DAN.md`, các thay đổi 13/11: `docs/THAY_DOI_VA_CACH_CHAY.md`.

---

## 1. Kiến trúc Pipeline

```
Micro / WAV ──► VAD năng lượng + Phân đoạn (Host, numpy) ──► ASR ──► MT (NLLB-200-600M) ──► Chuẩn hoá số sang chữ (Host) ──► TTS ──► Loa / WAV
                                                               │           │                                                   │
        vi : Zipformer-150M CTC (1 graph, fp16)                │           Đồ thị Encoder + Đồ thị bước Decoder               vi : Piper vais1000 (1 graph, byte text → audio)
        en/zh/ko : SenseVoice-Small (frontend + encoder)       │           (Vòng lặp giải mã trên Host, KV cache dạng tensor) zh : MeloTTS-ZH (1 graph: front bảng tĩnh + BERT)
                                                                                                                               en/ko : Supertonic-3 (1 graph: 5 bước ODE + vocoder)
```

Các đồ thị chạy trực tiếp trên Hexagon NPU (`--backend npu`) hoặc CPU ONNX Runtime (`--backend cpu` dùng cho môi trường phát triển / kiểm thử). Host CPU chỉ đảm nhận các phần việc tính toán nhẹ: I/O âm thanh, VAD năng lượng, fbank, CTC collapse, SentencePiece tokenizer, vòng lặp giải mã NLLB, chuẩn hoá số thành chữ và ghép nối file WAV.

| Thành phần | Mô hình | Đồ thị (Graph) | Ghi chú kỹ thuật |
|---|---|---|---|
| **ASR vi** | Zipformer-150M CR-CTC-RNNT-6000h, CTC head | `asr_vi` | Cửa sổ 15 s, FP16 (Đo kiểm trên chip qua AI Hub: WER 7.5% sạch / 9.5% 5dB / 6.2% 0dB) |
| **ASR en/zh/ko** | SenseVoice-Small | `asr_xx_frontend`, `asr_xx_encoder` | Cửa sổ 29 s, Encoder nén W8A16 |
| **MT** | NLLB-200-distilled-600M, từ vựng 139k token | `mt_enc` (nguồn 64 token), `mt_decB` (Decoder 4-beam) hoặc `mt_dec` (Greedy) | Vòng lặp giải mã trên host; Beam search tăng +0.5…2.4 chrF so với Greedy, tốc độ 31 ms (Beam) so với 16 ms (Greedy) mỗi bước trên chip |
| **TTS vi** | Piper vi_VN-vais1000-medium | `tts_vi` | $\le 26$ từ / chunk; các từ ngoài từ điển được tự động đếm (`n_oov`) |
| **TTS zh** | MeloTTS-ZH (+ bảng tra G2P chưng cất) | `tts_zh` | $\le 20$ ký tự / chunk |
| **TTS en/ko** | Supertonic-3 | `tts_enko` | $\le 82$ ký tự / chunk |

---

## 2. Khởi động nhanh (Môi trường phát triển trên Laptop / CPU tham chiếu)

```bash
pip install -r requirements.txt
# Lưu ý: backend CPU cần các file ONNX export, KHÔNG có trong gói cài đặt (gói chỉ có context binary cho NPU)
cp models.dev.json models/models.json          # Cấu hình đường dẫn tới các file ONNX cục bộ
python -m s2s_edge file --src vi --tgt en --in samples/vi_0.wav --out out.wav --backend cpu --models models/models.json
python -m s2s_edge live --src vi --tgt zh --backend cpu --models models/models.json     # Nhấn Enter, nói vào mic và nghe bản dịch
```

Hoặc chạy giao diện Web Dashboard cục bộ trực quan:
```bash
python s2s_dashboard.py --port 8765
# Mở trình duyệt tại: http://localhost:8765
```

---

### ASR: cửa sổ nhiều kích thước, RNN-T, đặc trưng SenseVoice trên host

Mỗi ASR có thể có nhiều đồ thị tĩnh (khoá `asr_vi@<frame>`, `asr_xx_encoder@<frame LFR>` trong models.json); pipeline chạy cửa sổ nhỏ nhất vừa câu nói, nên câu 3 s không phải trả giá cửa sổ 15 s / 29 s. Zipformer tiếng Việt giải mã bằng nhánh RNN-T khi có `asr_vi_decoder` + `asr_vi_joiner`; SenseVoice tính đặc trưng trên host khi `"_sv_frontend": "host"`. Mẫu cấu hình: `models.dev_asr_buckets.json`. Chi tiết, số đo và cách build/link trên AI Hub: [`docs/asr_optimization.md`](docs/asr_optimization.md); đánh giá riêng ASR: `python tools/eval_asr.py`, kiểm tra nhanh không cần model: `python tools/test_asr_host.py`.

---

## 3. Chế độ Hội thoại Đa ngữ

```bash
python -m s2s_edge live --pair vi,ko --continuous --models models/models.json   # Việt <-> Hàn, tự động nhận diện ngôn ngữ nói, không cần bấm phím
python -m s2s_edge live --auto --partner en --models models/models.json         # Việt <-> bất kỳ ngôn ngữ nào được nói (en / zh / ko)
```
Chi tiết về xử lý tên riêng (tên Latinh mà TTS tiếng Việt / Trung / Hàn chưa hỗ trợ), bảng thuật ngữ và giới hạn câu nói trộn ngôn ngữ: xem [`docs/conversation.md`](docs/conversation.md).

---

## 4. Chạy trên Thiết bị Phần cứng Thật (HSPTEK AIBOX 8550)

Bản tích hợp (Plan-01 + ASR speed/accuracy): profile mặc định, dựng gói với `--asr-config`, fallback trên box → [`docs/aibox8550_integration.md`](docs/aibox8550_integration.md).

Khởi động phiên kiểm tra toàn diện đầu tiên trên thiết bị:
```bash
bash tools/session0.sh      # Khảo sát môi trường + cài đặt + kiểm tra golden từng graph + smoke test + benchmark nhanh (kết quả đóng gói dạng .tgz)
```

Chạy từng lệnh độc lập:
```bash
bash tools/recon.sh                                                        # Khảo sát phần cứng & OS -> recon_<host>_<date>.txt
python tools/golden.py check --models models/models.json --backend npu      # So sánh từng graph NPU đối chiếu với CPU tham chiếu
python -m s2s_edge live --src vi --tgt en --models models/models.json --backend npu
python -m s2s_edge.bench --src vi --tgt en --manifest eval/manifest_vi.json --runs 3 --logs logs/     # Đo đạc minh chứng benchmark thực tế
```

Tệp cấu hình `models/models.json` ánh xạ tên các thành phần tới đường dẫn tệp (`ctx` = QNN context binary `.bin` biên dịch từ AI Hub link job, `onnx` = tệp ONNX tham chiếu CPU).

---

## 5. Danh mục Tệp tin & Thư mục

| Đường dẫn | Mục đích sử dụng |
|---|---|
| `s2s_edge/` | Gói thư viện chính: `audio` (I/O, VAD), `asr`, `mt`, `tts`, `textnorm`, `backends` (ONNX Runtime, QNN), `registry`, `pipeline`, `bench`, `__main__` (CLI) |
| `s2s_dashboard.py` | Web Dashboard trực quan chạy localhost: mic thu âm, dịch văn bản, nghe TTS, đo latency |
| `docs/VNG_SETUP.md` | Hướng dẫn kết nối và cấu hình thực địa trên phần cứng AIBOX 8550 tại VNG |
| `tools/recon.sh` | Script tự động khảo sát môi trường (OS, SoC, QNN libs, thiết bị âm thanh, mạng) |
| `tools/session0.sh` | Chuỗi tự động chạy đầu tiên trên thiết bị, đóng gói log thành `.tgz` |
| `tools/golden.py` | Tạo và kiểm tra các cặp vector kiểm thử đầu vào - đầu ra (golden test) cho từng graph |
| `tools/eval_e2e.py` | Đánh giá toàn trình audio → audio → ASR nghe lại đối chiếu với chuẩn FLORES (ASR-chrF / BLEU) cho 6 chiều |
| `tools/lid_features.py` | Trích xuất đặc trưng phục vụ nhận diện ngôn ngữ tiếng Việt / Ngoại ngữ |
| `tools/eval_mt.py` | Đánh giá chất lượng dịch máy FLORES-200 (chrF / BLEU) cho cả 6 chiều dịch |
| `assets/` | Chứa SentencePiece model, NLLB language ids, bảng mã unicode Supertonic + voice styles |
| `models.dev*.json` | File cấu hình mẫu cho môi trường phát triển (đường dẫn ONNX CPU) |

---

## 6. Các giới hạn đã biết & Giải pháp khắc phục

* **TTS Tiếng Trung:** Xử lý bằng bảng tra tĩnh: các chữ số được đọc thông qua `textnorm`, chữ cái Latinh và ký tự hiếm ngoài bảng bị lược bỏ.
* **TTS Tiếng Việt (Piper):** Lược bỏ các từ ngoài bảng âm tiết (được ghi nhận trong `n_oov`).
* **MT NLLB:** Đoạn câu nguồn tối đa $\le 30$ SentencePiece token; câu dài hơn được tự động ngắt theo dấu câu hoặc điểm giữa câu để dịch phân đoạn.
* **Độ dài đoạn âm thanh:** Một phân đoạn tiếng nói tối đa 14 s (`S2S_SEG_MAX_S`, vừa cửa sổ Zipformer 14,5 s); âm thanh dài hơn được cắt ở khung tĩnh lặng nhất.
* **VAD (Voice Activity Detector):** Sử dụng thuật toán năng lượng thích ứng theo mức tín hiệu thu âm (hoạt động tốt với cả micro độ nhạy thấp), có ngưỡng trễ (hysteresis) để giữ âm cuối yếu và gộp các ngắt ngắn trong câu; không dùng mô hình nơ-ron VAD riêng để tiết kiệm tài nguyên NPU.
* **Nói chồng âm & Pha trộn ngôn ngữ:** Trường hợp nhiều người nói cùng lúc hoặc câu tiếng Việt chèn từ tiếng Anh đàm thoại chưa được hỗ trợ hoàn toàn (xem chi tiết tại [`docs/conversation.md`](docs/conversation.md)).
* **Khả năng tương thích phần cứng:** Toàn bộ model được tối ưu hóa trực tiếp cho thiết bị **HSPTEK AIBOX 8550** (SoC Qualcomm QCS8550, kiến trúc NPU Hexagon v73, QAIRT 2.51). Không tương thích trực tiếp với các dòng chip cũ như QCS6490 / RB3 Gen 2 (Hexagon v68).
