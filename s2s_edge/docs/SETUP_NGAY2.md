
## 7. NLLB-200 3.3B trên hộp (chuẩn bị tối 12/11, đích là 3.3B; chi tiết: `docs/mt33_routing.md`)

Có trong gói: 4 file `models_qcs8550/nllb33_*_w8a16.bin` (3,1 GB), `assets/nllb33_host.npz`, `s2s_edge/mt33.py`, bộ định tuyến theo chiều `MtRouter` (trong `s2s_edge/mt.py`), cấu hình `models_qcs8550/models_nllb33.json` (khóa `mt_route`), `tools/mt33_check.py`, `tools/mt_rubric.py` + `eval/mt_rubric.json`.

**Route:** 3.3B cho vi-zh, zh-vi, vi-ko, ko-vi (chiều mà 600M sai); 600M cho en-vi, vi-en. Đổi trong `mt_route`.
**Cảnh báo:** nếu 3.3B không chạy được, pipeline lùi về 600M **kèm khung đỏ `!! MT DEGRADED`**, dải đỏ trên web, thẻ "MT LÙI VỀ 600M" ở từng clip; `S2S_MT_STRICT=1` thì báo lỗi thật. Hai script kiểm tra thoát với `FAIL: NLLB-3.3B IS NOT IN USE`. **Mọi lần chạy phải xác nhận dòng `3.3B in use: True` trước khi tin số liệu.**

Số đo AI Hub (chip ảo QCS8550, w8a16, beam 4): encA 12,6 ms, encB 18,5 ms, decA 50,4 + decB 56,9 = 107 ms/bước giải mã; không lỗi bộ nhớ. Chất lượng mô phỏng lượng tử hóa: float 41,82, w8 từng kênh 41,74 (chrF trung bình, 20 câu/chiều).

Các bước (sau `laptop_push.py`, thêm khoảng 4 GB):
1. Rubric qua đúng pipeline: lệnh trong `docs/mt33_routing.md` (`mt_rubric.py`). Phải thấy `3.3B in use: True`, rồi từng câu `[mt33]`/`[mt600]` và số câu đạt (laptop dự đoán 54/66).
2. `mt33_check.py` (chrF trên 120 câu ASR của chip, thời gian MT mỗi câu, RAM).
3. Chạy web: `cd /tmp/onevoice/s2s_edge_pkg && MODELS=models_qcs8550/models_nllb33.json SKIP_CHECK=1 bash tools/start_box.sh`; kiểm dải đỏ không xuất hiện, rồi chạy BTC như ngày 1 (`laptop_run_halves.py`, `laptop_stream.py <STAMP> <thư mục>`).
4. Quay về 600M hoàn toàn: `S2S_MT=600m` hoặc `models_qcs8550/models.json`.
