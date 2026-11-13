# MT: NLLB-3.3B cho các chiều mà 600M sai, 600M cho phần còn lại

**Đích là NLLB-200 3.3B.** Lùi về 600M chỉ là lưới an toàn để pipeline không chết, và không bao giờ im lặng.

## Route (trong `models_qcs8550/models_nllb33.json`, khóa `mt_route`)

| Chiều | Model | Cơ sở |
|---|---|---|
| vi→zh | **3.3B** | BTC 90 câu: chrF +7,2 (khoảng tin cậy 95 % +3,0..+11,6). Rubric: 15/22 so với 10/22 của 600M |
| zh→vi | **3.3B** | chrF +3,3 (+0,7..+5,8). Rubric 6/6 so với 5/6 |
| ko→vi | **3.3B** | chrF +3,1 (+0,4..+6,1). Rubric 8/8 so với 5/8 |
| vi→ko | **3.3B** | chrF −1,9 (khoảng tin cậy −6,5..+2,5, không kết luận được) nhưng sửa 3/3 lỗi đồng âm đã báo cáo (cảm, rảnh rỗi, dịch vụ phòng). Rubric 10/10 so với 6/10 (kèm luật kính ngữ cho cả hai) |
| en→vi | 600M | 3.3B không hơn: chrF −1,3 (−3,7..+0,9); có câu bị bỏ mất nửa ý. Rubric 11/14 (3.3B) so với 10/14 (600M): không khác biệt có ý nghĩa |
| vi→en | 600M | chrF +2,4 (−1,2..+6,1) chưa chắc chắn, và 3.3B cũng bỏ chữ ("You wait for me." thiếu "một chút"). Rubric 5/6 ngang nhau |

Đổi một chiều: sửa một dòng trong `mt_route` (không cần build lại). Chiều không có trong `mt_route` dùng 600M.
Số chrF lấy từ chữ ASR thật của chip trên BTC (`outputOnevoice2/mt_compare`); rubric là `eval/mt_rubric.json` (66 câu mới cùng kiểu lỗi, không trùng câu báo cáo; `outputOnevoice2/mt_rubric`). 3.3B trong các phép đo này là bản int8 chạy trên laptop; bản w8a16 trên chip chưa đo chất lượng.

## Khi 3.3B không chạy được thì sao (cố ý ồn ào)
- Lúc khởi động server (nạp 4 đồ thị + dịch thử một câu) hoặc giữa chừng (lỗi bộ nhớ, lỗi NPU): in khung `!! MT DEGRADED: NLLB-3.3B is NOT in use ...` kèm lý do, `degraded=True`.
- Web: dải đỏ đầu trang "MT LÙI VỀ 600M" (lấy từ `/api/info`), mỗi clip bị ảnh hưởng có cột `mt_model=mt600` khác `mt_wanted=mt33` và thẻ vấn đề **"MT LÙI VỀ 600M"**; cột này nằm trong `results.csv` của file zip.
- `S2S_MT_STRICT=1`: không lùi, báo lỗi thật (server không khởi động / clip lỗi). Dùng khi muốn chắc chắn đang đo 3.3B.
- `tools/mt33_check.py` và `tools/mt_rubric.py` thoát với mã 2 và in `FAIL: NLLB-3.3B IS NOT IN USE` thay vì in số của 600M.
- Chiều đã route về 600M (en→vi, vi→en) không bao giờ đụng 3.3B nên không báo lỗi khi 3.3B hỏng.

## Kiểm trên hộp (chạy được ngay sau `laptop_push.py`)
```
runuser -u system -- bash -c 'cd /tmp/onevoice/s2s_edge_pkg && . .venv/bin/activate && . tools/env.sh 2>/dev/null && python tools/mt_rubric.py 2>&1 | grep -a "^##" | tee /tmp/mt_rubric.txt'
```
In từng câu: model đã dịch ([mt33] / [mt600]), PASS/FAIL, thời gian; cuối cùng số câu đạt theo chiều. Sau đó `tools/mt33_check.py` (chrF trên 120 câu ASR của chip) và chạy web bằng `MODELS=models_qcs8550/models_nllb33.json SKIP_CHECK=1 bash tools/start_box.sh`.

## Lọc rác trước MT và sau MT (`s2s_edge/asr_clean.py`, tắt cả hai bằng `S2S_ASR_CLEAN=0`)
**Lớp 1, chữ ASR trước MT (`clean_asr`)**: chỉ cắt thứ không phải lời nói, mỗi luật đều có mẫu thật:
- nhãn lạc đầu câu `Q. `, `H. `, `问：` (chip BTC: VM-1966, VM-5073); chỉ phụ âm đơn + "." + khoảng trắng, nên "A,", "Q1", "H.264", "K.O." còn nguyên
- từ ngập ngừng đầu câu `Um. `, `Ừm, ` (chip BTC: VK-2808)
- từ ngập ngừng đứng thành **câu riêng** ở đầu/cuối khi nhiễu: `嗯。`, `그.`, cuối câu `没。` (ADV có nhiễu: 19 clip zh/ko); có dấu phẩy thì là lời nói ("嗯，我知道了", "음, 알겠어요") và được giữ
- chuỗi lặp ≥ 4 lần (giữ 2); chữ rỗng / chỉ dấu câu / một chữ cái ASCII → không gọi MT, TTS
**Lớp 2, chữ MT trước TTS (`clean_mt`)**: ký hiệu khối "❚ ‖", nhãn "问:" chép từ nguồn, gạch đầu dòng "- " / dấu câu ở đầu, "。" trong câu tiếng Việt/Anh; chỉ toàn dấu câu → không đọc. Không đụng chữ.
**Không làm** (đã đo, không có lợi): gộp thán từ bị lặp "그래, 그래." — chính đáp án BTC cũng lặp ("아니, 아니."), chrF vi-ko 26,74 → 26,69.
Chữ gốc luôn còn trong cột `asr_raw`; cột `asr_cleaned` ghi luật nào đã chạy; thẻ vấn đề `asr_cleaned` trên web.
Test: `tests/test_asr_clean.py`, `tests/test_mt_clean.py`, `tests/test_asr_clean_pipeline.py` (ca giữ/ca đổi/đánh đổi đã biết, 20.000 chuỗi ngẫu nhiên mỗi lớp, transcript gốc của BTC + ADV không bị cắt gì ngoài "um" đầu câu, cả đường thường lẫn stream).
**Sau mỗi lần chạy trên chip**: `python tools/junk_report.py <results.csv trong zip> --out junk_report.md`: (1) mọi chỗ đã cắt để kiểm có cắt nhầm không, (2) mảnh ngắn đầu/cuối không có trong transcript gốc = ứng viên rác mới, (3) rác còn trong MT. Chỉ thêm luật khi có mẫu thật.
