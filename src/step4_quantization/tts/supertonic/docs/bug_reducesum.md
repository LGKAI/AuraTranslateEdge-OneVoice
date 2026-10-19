# Lỗi `ReduceSum` trên HTP làm hỏng vector estimator

## Triệu chứng

Graph gộp (chữ → wav) compile và chạy trên Dragonwing IQ-9075 với **4476/4476 layer trên NPU**, độ dài audio khớp CPU
(62/62, 43/43, ...), nhưng giọng nhiễu: round-trip ASR cho WER tiếng Anh 95%, CER tiếng Hàn 89% (`results/eval_full_fp16.json`).

Đưa latent cuối do NPU tạo ra vào vocoder fp32 trên CPU vẫn ra nhiễu (WER Anh 93%) ⇒ vocoder trên chip không sai,
lỗi nằm ở 5 bước ODE (vector estimator). Đo vector estimator riêng trên chip bằng đầu vào thật, cosine **một bước** so với CPU:

| Bước ODE | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| Trước khi sửa | 0.998 | 0.978 | 0.956 | 0.948 | 0.941 |
| Sau khi sửa | 1.0 | 1.0 | 0.99999 | 0.99999 | 0.99997 |

(`logs/chip/ve_fp16.log`, `logs/chip/ve_rotR_L150.log`.)

## Vị trí lỗi trong graph

Khối attention chéo `/vector_estimator/vector_field/main_blocks.3/attn` (bảng sin/cos của nó được dùng chung cho các
khối 9, 15, 21 vì trình tối ưu gộp biểu thức giống nhau):

```
latent_mask [1,1,150]
 → Tile (lặp [2,1,1])                 → [2,1,150]
 → ReduceSum(axes=[1,2])              → [2]      CPU: 62    chip: 1   ← LỖI
 → Reshape [2,1,1]
 → Div: vị_trí[0..149] / độ_dài       (góc chuẩn hoá của rotary)
 → Mul tần số → Sin, Cos
 → quay query (Slice, Mul, Sub, Add, Concat)  → điểm số attention chéo
```

Rotary dùng `vị_trí / độ_dài_latent_thật`. Chip tính độ dài bằng 1 thay vì 62, nên góc quay của query lớn hơn đúng ~62
lần. Key (độ dài lấy từ mặt nạ text) vẫn đúng, query và key lệch nhau, attention nhìn sai vị trí, và sai số tích luỹ qua
5 bước ODE.

## Bằng chứng

1. Cắt graph tại các điểm trung gian, chạy từng cái trên chip (`diagnostics/ve_cut.py`, `ve_cut2.py`), so cosine với CPU
   (chỉ trên các frame hợp lệ): đầu vào query *trước* khi quay đúng tuyệt đối (cosine 1.00000), **sau** khi quay sai
   (0.21 ở bước 0, 0.36 ở bước 4); góc quay trên chip riêng lẻ (Div, Mul, Cos, Sin) đúng khi chạy từng cái.
2. Từ kết quả của chip và hai nửa query đúng, giải ngược phép quay mà chip đã áp dụng: khớp với **độ dài = 1**
   (cosine 0.9965); độ dài = 62 (đúng) chỉ cho 0.21. Đó là một phép quay thuần (c²+s² ≈ 1) nhưng sai góc.
3. Graph nhiều đầu ra (`diagnostics/ve_multi_run.py`, `logs/chip/ve_multi.log`) trả cùng lúc `ReduceSum`, `Reshape`, `Div`,
   `Mul`, `Cos`, `Concat_3`:

   | Tensor | CPU | NPU |
   |---|---|---|
   | ReduceSum | 62 | **1** |
   | Div (giá trị lớn nhất) | 2.40 | **149.0** |
   | Mul (giá trị lớn nhất, góc) | 24.0 | **1490** |

4. Sau khi thay `ReduceSum` bằng `Reshape + MatMul` với vector toàn 1 (`scripts/rot_fix.py make R`), cosine từng bước
   ODE về 1.0 đến 0.99997, và graph gộp cho WER Anh 0.0% / CER Hàn 1.5%.

Cùng phép `ReduceSum` đó chạy **riêng một mình** trên chip lại cho 62 (đúng). Lỗi chỉ xuất hiện trong graph lớn, gợi ý do
trình biên dịch tối ưu/gộp phép toán; chưa xác định được lỗi ở bộ chuyển đổi QAIRT hay ở kernel HTP.

## Cách sửa

```python
# ReduceSum(x[B,1,L], axes=[1,2]) -> [B]   thay bằng:
y = MatMul(Reshape(x, [B, L]), ones[L, 1])   # [B,1]
y = Reshape(y, [B])
```

Tương đương toán học (kiểm trên CPU: cosine 0.99999994 với graph gốc). Áp dụng cho cả hai `ReduceSum` trong khối
(độ dài latent `[2,1,150]` và độ dài text `[2,1,96]`).

## Giả thuyết đã thử và loại trừ (bằng đo trên chip)

Mỗi mục đều có log trong `logs/chip/` và job ID trong `docs/deploy_log.md`.

| Giả thuyết | Cách thử | Kết quả |
|---|---|---|
| Dải giá trị fp16 / LayerNorm tràn | đo biên độ tensor trên CPU (`diagnostics/ve_ranges.py`) | biên độ nhỏ (LayerNorm vào ≤ 13) — loại |
| Hàm Erf (GeLU) | thay Erf bằng tanh | cosine y hệt — loại |
| Nhánh thời gian `sin/cos(1000·t)` | tính sẵn embedding thời gian (`diagnostics/ve_tin.py`) | cosine y hệt — loại |
| `-inf` trong mặt nạ attention | thay bằng −30000 | cosine y hệt — loại |
| Lượng tử W8A16 / W16A16 | quantize + compile (cần thêm bias 0 cho Conv) | **tệ hơn** (bước 4: 0.76) — loại |
| Độ dài latent không phải bội của 32 | L = 96, 120, 128, 160 | cùng lỗi — loại |
| Kích thước tensor ≥ 65536 phần tử | L = 96, 120 (dưới ngưỡng) | cùng lỗi — loại |
| Sin/Cos sai ở góc lớn | làm tròn fp16 trên CPU; bảng cos/sin tính sẵn | không đổi — loại |
| Mul broadcast rank-4 × rank-3 | graph tổng hợp (`diagnostics/mul_synth.py`) | đúng — loại |
| Slice → Mul bị gộp sai | thay Slice bằng MatMul chọn cột | cosine y hệt — loại |

Mô phỏng fp16 trên CPU (`diagnostics/fp16_sim.py`) cho cosine ~1.0 ở mọi tầng, **không phát hiện được lỗi này**: chỉ chạy
thật trên chip mới thấy.
