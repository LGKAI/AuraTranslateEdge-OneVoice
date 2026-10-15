# NLLB-600M (dịch máy) trên NPU Dragonwing IQ-9075 — không còn vòng lặp từng token

Dịch máy kiểu NLLB sinh từng token một (vòng lặp). Gọi chip qua AI Hub cho mỗi token thì rất chậm
(mỗi lần gọi mất vài chục phút). Thư mục này đưa **cả encoder và nhiều bước giải mã vào một graph duy nhất**,
nên mỗi câu chỉ cần **một lần gọi**: vào là các id token của câu nguồn, ra là các id token của bản dịch.

## Kết quả trên chip (đo thật, Dragonwing IQ-9075 EVK, qua Qualcomm AI Hub)

Mỗi dòng là một graph có số bước giải mã khác nhau. "Giống CPU" là so 8 câu dịch với bản chạy trên CPU.

| Số bước giải mã (T) | Kết quả | Layer trên NPU | Thời gian chip | Nạp model lần đầu |
|---|---|---|---|---|
| 2 | **8/8 giống CPU** | 1309/1309 (100% HTP) | 23.8 ms | 62 s |
| 8 | **8/8 giống CPU** | 4087/4087 (100% HTP) | 100.1 ms | 385 s |
| **16** | **8/8 giống CPU (100%)** | **7617/7617 (100% HTP)** | **215.87 ms** | 865.9 s |
| 24 | **không chạy được**: quá thời gian tối đa của AI Hub (4072 s) | — | — | — |

Vậy với một graph, **T=16 là mức tối ưu lớn nhất đã chạy thành công trên chip thật Dragonwing IQ-9075 EVK**.
- **Compile Job chính thức** (`leogiakhanh1609@gmail.com`): [`jgd62w9lp`](https://workbench.aihub.qualcomm.com/jobs/jgd62w9lp/) (**SUCCESS**, 100% NPU offload).
- **Profile Job chính thức**: [`jp1o6ly25`](https://workbench.aihub.qualcomm.com/jobs/jp1o6ly25/) (**215.87 ms**, RAM đỉnh 6.59 MB).
- **Inference Job chính thức**: [`jgd629eep`](https://workbench.aihub.qualcomm.com/jobs/jgd629eep/) (**8/8 câu trùng khớp 100% từng token**).
- Chi tiết kết quả JSON đã lưu tại: `results/hub_silicon_test_results_T16.json` (và nhật ký đo kiểm tại `docs/ket_qua_chip.md`).

8 câu thử nghiệm (bao phủ 6 hướng dịch: Việt ↔ Anh, Anh → Hàn, Hàn → Anh, Trung ↔ Anh, Hàn → Việt) cho kết quả trên silicon thật giống 100% từng token so với PyTorch CPU, ví dụ:
- "Hôm nay trời đẹp quá, chúng ta cùng đi dạo nhé." → "It's a beautiful day, let's go for a walk."
- "Tôi muốn đặt một bàn cho hai người vào tối nay." → "I want to set a table for you two tonight."

## Cách làm (đơn giản)

- **Encoder**: cố định 32 token (đệm bằng token pad).
- **Giải mã**: viết lặp lại T bước ngay trong graph, bộ nhớ KV (k, v của các token đã sinh) giữ trong graph. Mỗi bước
  chọn token có điểm cao nhất (`argmax`) rồi dùng làm đầu vào bước sau. Bước đầu bị ép sinh token mã ngôn ngữ đích.
- **Cắt bảng từ vựng** từ 256k xuống 139k token (giữ trọn bốn chữ viết: Latin, Việt, Hàn, Hán) để file ONNX nhỏ hơn 2 GB
  (AI Hub không nhận file lớn hơn). Bảng nhúng lưu fp16. Việc đổi id cũ ↔ id mới nằm trong graph.
- Đầu ra là mảng T id; máy chủ cắt tại token kết thúc (EOS = 2) rồi giải mã chữ.

## Giới hạn (nói thẳng)

- Câu nguồn tối đa 32 token; bản dịch tối đa T−1 token (với T=16 là 15). Câu dài hơn phải chia câu.
- Mới thử 8 câu ngắn đối chiếu với CPU; **chưa đo BLEU** trên tập lớn và chưa nối với đầu ra ASR thật.
- Bản fp16 (không lượng tử), nặng khoảng 1.98 GB. Cắt từ vựng xuống 42k làm hỏng chất lượng (xem `docs/cach_hoat_dong.md`).
- Máy chủ vẫn làm: tách token (SentencePiece), cắt tại EOS, giải mã chữ. AI Hub chỉ chạy từng model, không chạy mã tuỳ ý.
- Encoder cũ của dự án từng lệch nhiều trên chip (cosine 0.2–0.8 khi lượng tử); bản này viết lại bằng fp16 và khớp CPU trên 8 câu thử.

## Chạy lại

Yêu cầu: Python 3.11 / 3.12, `pip install -r requirements.txt`, `qai-hub` đã cấu hình. Chạy từ thư mục `src/step4_quantization/mt/nllb/` (các file sinh ra nằm ở `work/`, đã được đưa vào `.gitignore` tổng).

```bash
cd src/step4_quantization/mt/nllb

# 1. Tải model, dịch 8 câu trên CPU -> work/ref.json
python step0_cpu_ref.py

# 2. Kiểm tra graph tĩnh (T=24, trên CPU) khớp HF generate: 8/8
python step1_cpu_static.py

# 3. Cắt từ vựng 139k: 48/48 câu FLEURS giống HF
python step3b_parity_allscripts.py

# 4. Xuất ONNX T=16 (~1.98 GB, 12.710 nodes) -> work/nllb_S32_T16.onnx
python step4_export.py 16

# 5. Compile + profile + inference trên chip Dragonwing, đối chiếu kết quả
python hub_nllb.py T16 work/nllb_S32_T16.onnx 16
```

Với `hub_nllb.py`, đặt biến môi trường `REUSE=1` (PowerShell: `$env:REUSE="1"`) để dùng lại bản compile đã có mà không cần compile lại.

## Chạy MT Demo Dashboard (`mt_demo_server.py`)

Tương tự như ASR Demo Dashboard (`asr_demo_server.py`), dự án cung cấp máy chủ web demo trực quan cho mô hình dịch máy **NLLB-600M** tích hợp NPU Qualcomm và mô phỏng CPU:

```bash
cd src/step4_quantization/mt

# Khởi chạy server demo (mặc định cổng 8430):
python mt_demo_server.py --port 8430

# Hoặc dùng uvicorn trực tiếp:
uvicorn mt_demo_server:app --host 127.0.0.1 --port 8430

# Tự động tạo public link HTTPS qua Cloudflare Tunnel (chia sẻ ra mạng ngoài):
python mt_demo_server.py --port 8430 --share
```

Truy cập: `http://localhost:8430`

**Các tính năng trên Dashboard:**
1. **6 Hướng dịch đa ngữ:** Việt ↔ Anh, Anh ↔ Trung, Anh ↔ Hàn, Hàn ↔ Việt (kèm nút đổi chiều nhanh `⇄`).
2. **Hai chế độ thực thi (Dual Engines):**
   - 🟢 **NPU Qualcomm Dragonwing (Cloud Hub):** Chạy trực tiếp trên chip NPU vật lý (`mqy498l5q`, 7.617 layers Hexagon HTP v73, đo latency 215.87 ms).
   - ⚡ **Local CPU (ONNX Runtime):** Chạy tức thì trong ~30ms trên file `nllb/work/nllb_S32_T16.onnx` để thử nghiệm câu dịch nhanh mà không phải chờ hàng đợi AI Hub.
3. **8 Câu mẫu đối chiếu FLEURS:** Click 1 chạm để tự động điền câu và chọn cặp ngôn ngữ chuẩn.
4. **Timeline trạng thái thời gian thực:** Hiển thị từng giai đoạn (Tokenize → Upload → Provisioning → Running on NPU → Detokenize).
5. **Chi tiết Token IDs trên silicon:** Xem mảng raw token IDs và các subword tokens do chip sinh ra.

## Các file

| File | Việc |
|---|---|
| `mt_demo_server.py` | Dashboard Web demo dịch máy đa ngữ NPU / CPU |
| `nllb_static.py` | Mô hình tĩnh: encoder + T bước giải mã trong một graph |
| `vocab_restrict.py` | Chọn tập token giữ lại và tách token theo tập đó |
| `step0_cpu_ref.py` | Bản dịch tham chiếu trên CPU |
| `step1_cpu_static.py` | Kiểm tra graph tĩnh khớp HF trên CPU |
| `step2_prune.py`, `step3_prune_parity.py` | Thử cắt từ vựng còn 42k (không đạt, giữ để đối chiếu) |
| `step3b_parity_allscripts.py` | Cắt từ vựng còn 139k: khớp HF 48/48 |
| `step4_export.py` | Xuất ONNX với T bước |
| `hub_nllb.py` | Đưa lên chip và so kết quả |
| `logs/chip/` | Log chạy trên chip (đã lọc), `results/` kết quả |
| `docs/` | Giải thích chi tiết và nhật ký job |
