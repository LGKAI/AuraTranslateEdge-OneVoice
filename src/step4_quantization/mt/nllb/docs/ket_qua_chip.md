# Nhật ký chạy trên chip (AI Hub, Dragonwing IQ-9075 EVK)

Job xem tại `https://workbench.aihub.qualcomm.com/jobs/<ID>/` (tài khoản AI Hub của tác giả). Log đã lọc bỏ thanh tiến trình, bản gốc đã lọc nằm ở `logs/chip/`.
Cấu hình compile: `--target_runtime qnn_dlc --truncate_64bit_tensors --truncate_64bit_io` (fp16, không lượng tử).
Mỗi graph nhận `src_ids` (int32 [1,32]) và `tgt_lang` (int32 [1]), trả `out_ids` (int32 [1,T]).

| T (bước giải mã) | Compile | Profile | Inference | Kết quả |
|---|---|---|---|---|
| 2 | `jgk699mn5` | `jgk698xn5` | `j5q4mvqog` | 8/8 giống CPU; 23.8 ms; 1309/1309 layer NPU; nạp 62 s (`logs/chip/hub_T2.log`) |
| 8 | `jp0ovv7np` | `jp3ow6jnp` | `jgod482k5` | 8/8 giống CPU; 100.1 ms; 4087/4087 layer NPU; nạp 385 s (`hub_T8.log`) |
| 16 | `jg9oxdj8g` | `jgd6z486p` | `jp4e94m2g` | 8/8 giống CPU; 275.6 ms; 7617/7617 layer NPU; nạp 861 s (`hub_T16b.log`) |
| 24 | `j5q4mvreg` | `j57o7nkng` (lỗi) | `jpx0dr38p` (lỗi) | compile thành công; inference **FAILED: exceeded the maximum execution time** sau 4072 s (`hub_T24b.log`) |

Ghi chú:
- Thời gian chip là thời gian ổn định đo bằng profile, không gồm hàng đợi và nạp model. Mỗi job thật mất khoảng 10–70 phút
  vì nạp model lần đầu và hàng đợi.
- Số layer tăng khoảng 400 mỗi bước giải mã. T=16 (7617 layer) chạy được, T=24 (~11000 layer) thì không chạy được trong
  giới hạn thời gian của AI Hub. Chưa thử T=17..23 nên mức tối đa chính xác nằm trong khoảng đó.
- Job T=16 lần đầu bị đứt phiên giữa chừng, nên inference và profile được gửi lại bằng cách dùng lại bản đã compile (`REUSE=1`).
- Kết quả thô từ chip: `results/hw_T2.npy`, `hw_T8.npy`, `hw_T16.npy` (8 câu × T id).
