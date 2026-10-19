# Supertonic 3 TTS trên NPU Dragonwing IQ-9075 — một graph, một lần inference

Đưa toàn bộ chuỗi **chữ → âm thanh** của Supertonic 3 (tiếng Anh, tiếng Hàn) lên NPU Hexagon của
**Dragonwing IQ-9075 EVK** qua Qualcomm AI Hub, dưới dạng **một graph ONNX duy nhất, một lần gọi inference**.
Đo bằng phần cứng thật, có job ID và log đầy đủ.

> Mọi số liệu dưới đây là kết quả thật của tác giả trên tài khoản AI Hub của họ. Mẫu đánh giá nhỏ
> (6 câu mỗi ngôn ngữ), xem mục *Giới hạn*.

## 1. Kết quả

Đánh giá vòng khứ hồi: tổng hợp 12 câu → nhận dạng bằng SenseVoice fp32 (CPU) → so với chữ đầu vào.

| Cấu hình | Tiếng Anh WER | Tiếng Hàn CER | Ghi chú |
|---|---|---|---|
| CPU fp32, 4 model rời (tham chiếu) | 0.0% | 0.0% (mỗi câu một hạt nhiễu) | `results/roundtrip_cpu_N96_L150.json` |
| CPU fp32, graph gộp (hạt nhiễu cố định) | 0.0% | 5.4% | khớp chuỗi rời: cosine 1.000002 |
| **NPU, graph gộp, đã sửa lỗi** (`full_v3`) | **0.0%** | **1.5%** | `results/eval_full_v3.json` |
| NPU, graph gộp, chưa sửa lỗi (`full_fp16`) | 95.1% | 89.5% | giọng nhiễu, xem `docs/bug_reducesum.md` |

WER tiếng Hàn không dùng được vì SenseVoice chèn khoảng trắng giữa các âm tiết, nên chỉ nhìn CER.

Profile trên chip (`submit_profile_job`, thời gian chip, không tính hàng đợi và nạp model):

| | |
|---|---|
| Layer chạy trên NPU | **4476 / 4476** |
| Thời gian chip mỗi câu | **748.3 ms** (cố định, luôn tính đủ 150 frame) |
| Bộ nhớ đỉnh | 10 MB |
| Job (compile / profile / inference) | `jprxn9nep` / `jpe6l2l0g` / `jgzz4w46g` |

Audio nghe thử: `results/wav_npu_full_v3/` (12 file, 44.1 kHz). Toàn bộ job ID: `docs/deploy_log.md`.

## 2. Graph gộp

```
cp  int32 [1,96]  (mã Unicode của chữ đã gắn thẻ ngôn ngữ, 0 = đệm)
   │  Gather(bảng unicode_indexer 65.536 mục) → text_ids ; text_mask = (cp>0)
   ├─ Duration predictor ───────────────► duration
   │        └─ độ dài latent = ceil(duration/1.05 × 44100/3072), kẹp [1,150]  → latent_mask
   └─ Text encoder ──► text_emb
                        │  nhiễu hằng số (seed 0) × latent_mask
                        ▼
              5 bước Euler ODE (Vector estimator; trọng số dùng chung, node nhân bản 5 lần)
                        ▼
                     Vocoder ──► × mặt nạ theo độ dài ──► wav [1, 460800] (44.1 kHz)
 đầu vào phụ: style_dp [1,8,16], style_ttl [1,50,256] (vector giọng, giọng M1)
 đầu ra     : wav, lat_len  (số mẫu hợp lệ = lat_len × 3072)
```

Kích thước tĩnh: văn bản tối đa 96 ký tự sau khi gắn thẻ (khoảng 87 ký tự chữ), latent tối đa 150 frame
(≈ 10.4 giây). Máy chủ chỉ còn: chuẩn hoá chữ + gắn thẻ ngôn ngữ + đổi sang mã Unicode (do thư viện `supertonic`),
cắt audio theo `lat_len`, ghi wav.

## 3. Lỗi đã tìm ra (quan trọng)

Lần chạy đầu của graph gộp trên chip **chạy 100% trên NPU nhưng giọng nhiễu**. Nguyên nhân sau nhiều vòng bisect:
**`ReduceSum(axes=[1,2])` của mặt nạ latent trả về 1 thay vì 62 trên chip**. Độ dài này dùng để tính góc rotary của
query trong 4 khối attention chéo của vector estimator, nên góc sai ~62 lần, attention sai chỗ, các bước ODE lệch dần.
Cách sửa: thay `ReduceSum` bằng `Reshape + MatMul` với vector toàn 1 (tương đương toán học, khớp CPU).
Chi tiết, bằng chứng và các giả thuyết đã loại trừ: **`docs/bug_reducesum.md`**.

## 4. Cấu trúc thư mục

```
supertonic_npu/
├─ scripts/        chuỗi dựng + chạy chính (xem mục 5)
├─ diagnostics/    script chẩn đoán dùng để tìm lỗi (bisect, thí nghiệm giả thuyết) — không cần để chạy
├─ data/           sentences.json (12 câu đánh giá)
├─ results/        eval_*.json, roundtrip CPU, wav NPU
├─ logs/chip/      log chạy trên chip đã lọc thanh tiến trình + jobs.json
├─ logs/cpu/       log kiểm tra trên CPU
└─ docs/           bug_reducesum.md, deploy_log.md, friend_repo_review.md
```

## 5. Cách chạy lại

Yêu cầu: Python 3.11, `pip install -r requirements.txt`, `qai-hub` đã cấu hình (`qai-hub configure --api_token ...`),
mô hình SenseVoice combined ONNX để chấm điểm (đường dẫn qua biến `SV_ONNX`, mặc định
`...\AuraTranslateEdge-OneVoice\outputs\sensevoice-e2e-onnx\model_sv_combined_single_inline.onnx` — xem nhánh `sensevoice`).
Gói `supertonic` tự tải model về `~/.cache/supertonic3` lần đầu. File sinh ra nằm ở `work/` (đã `.gitignore`).
Chạy từ thư mục `supertonic_npu/`:

```bash
# 1) Ép 4 model gốc về kích thước tĩnh (N=96 ký tự, L=150 frame) + fold hằng
python scripts/staticize.py 96 150
python scripts/verify_static.py 96 150          # so với model động (CPU)

# 2) Mẫu tham chiếu, sửa ReduceSum cho HTP, ghép thành một graph (398 MB)
python diagnostics/ve_variants.py prep
python scripts/rot_fix.py make R
python scripts/fuse.py 96 150 5
mkdir -p work/wav_cpu
python scripts/verify_fused.py work/supertonic_full_N96_L150_s5.onnx 96 150   # CPU + ASR

# 3) Lên chip: compile fp16 -> profile + một job inference cho 12 câu
python scripts/hub_run.py full_v3 work/supertonic_full_N96_L150_s5.onnx 96 150
python scripts/eval_hw.py full_v3               # ASR, so với CPU, đọc profile
```

`hub_run.py` mất khoảng 25 phút (chủ yếu là nạp model lần đầu ~740 s và hàng đợi). AI Hub trả đầu ra tên
`output_N`; script luôn sắp theo số N (thứ tự dict không đáng tin).

## 6. Giới hạn

- Mẫu đánh giá nhỏ: 6 câu Anh + 6 câu Hàn, câu ngắn, một giọng (M1), một hạt nhiễu cố định.
  Tiếng Hàn khá nhạy với hạt nhiễu (CER 5.4% trên CPU với seed 0).
- Độ dài cố định: tối đa 96 ký tự văn bản và 10.4 giây audio mỗi lần. Câu dài phải chia ở máy chủ.
- Chip luôn tính đủ 150 frame nên 748 ms không giảm với câu ngắn.
- Chỉ thử tiếng Anh và tiếng Hàn (model hỗ trợ nhiều ngôn ngữ hơn).
- Lỗi `ReduceSum` được xác định bằng thực nghiệm trên chip; tôi chưa biết nó nằm ở bộ chuyển đổi QAIRT hay kernel HTP.
- Máy chủ vẫn làm: chuẩn hoá chữ, đổi mã Unicode, cắt audio, ghi wav. AI Hub chỉ chạy từng model, không chạy được
  mã tuỳ ý trên CPU của board.

## 7. Bài học khi deploy lên AI Hub / HTP

- `download_output_data()` trả `output_0..N`; luôn sắp theo chỉ số, đừng tin thứ tự dict.
- Một op có thể đúng khi compile riêng nhưng sai khi nằm trong graph lớn: **kiểm chứng trên chip, đừng chỉ dựa vào CPU
  hoặc mô phỏng fp16**.
- Cách bisect hiệu quả: cắt mô hình con tiền tố bằng `onnx.utils.extract_model` (xoá `graph.value_info` sau khi cắt),
  chạy từng cái trên chip, so cosine với CPU, rồi giải ngược đại lượng mà chip đã dùng.
- Cosine của **một bước** ở đầu quá trình lặp (ODE) không đủ để phát hiện lỗi; đo ở mọi bước.
- Chạy lại cùng một model đã compile ở ngày khác có thể cho kết quả khác (đã gặp với Zipformer); luôn kiểm tra lại.

Đánh giá repo TTS của đồng đội (mà phần này dựa trên): `docs/friend_repo_review.md`.
