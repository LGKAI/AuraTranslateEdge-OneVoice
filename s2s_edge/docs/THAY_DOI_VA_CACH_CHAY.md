# Những gì đã đổi (13/11) và cách chạy

Ba thay đổi vào luồng chính: **(1) MT mặc định là NLLB-3.3B theo route**, **(2) lọc rác ASR/MT**, **(3) sửa lỗi tách câu cắt số + kính ngữ tiếng Hàn rộng hơn** (mục 6), **(4) câu dài** (mục 7). Chưa chạy trên chip nên mọi số liệu dưới đây đo trên laptop.

## 1. Đã đổi gì

| Hạng mục | Trước | Bây giờ |
|---|---|---|
| `models_qcs8550/models.json` (mặc định mọi script) | NLLB-600M mọi chiều | **NLLB-3.3B** cho vi↔zh, vi↔ko; 600M cho en→vi, vi→en (khóa `mt_route`) |
| Bộ cũ | `models.json` | giữ nguyên nội dung ở `models_qcs8550/models_600m.json` (lưới an toàn thủ công) |
| `models_nllb33.json` | bản thử | giống hệt `models.json`, giữ để các lệnh cũ không gãy |
| Chữ ASR trước MT | chỉ bỏ câu < 2 ký tự | `clean_asr`: bỏ nhãn lạc "Q. ", "H. ", "问：", "Um. " đầu câu, thán từ đứng thành câu riêng ("嗯。", "그.", cuối câu "没。") khi nhiễu, chuỗi lặp ≥ 4 lần |
| Chữ MT trước TTS | đọc nguyên | `clean_mt`: bỏ "❚ ‖", "问:" chép từ nguồn, "- "/dấu câu đầu câu, "。" trong câu Việt/Anh; chỉ toàn dấu câu thì không đọc |
| Web / `results.csv` | | thêm cột `asr_raw`, `asr_cleaned`, thẻ vấn đề "ASR đã lọc rác" |

`session0.sh`, `run_btc.sh`, `run_adv.sh`, `box_serve.sh`, `run_demo.sh`, `start_box.sh` đều đọc `models.json` (hoặc `logs/models_ok.txt`) nên tự dùng 3.3B, không cần `MODELS=...` nữa. Kiểm golden của `session0.sh` chỉ xét các đồ thị 600M/ASR/TTS; bốn đồ thị 3.3B được kiểm riêng ở bước 3 bên dưới.

Nếu 3.3B không nạp được (thiếu bộ nhớ, lỗi NPU): pipeline lùi về 600M **và báo to** (khung `!! MT DEGRADED`, dải đỏ trên web, cột `mt_model` ≠ `mt_wanted`). Chi tiết: `docs/mt33_routing.md`.

## 2. Công tắc

| Mục đích | Cách |
|---|---|
| Quay lại 600M toàn bộ | `S2S_MT=600m` hoặc `MODELS=models_qcs8550/models_600m.json` |
| Cấm lùi âm thầm (đang muốn đo 3.3B) | `S2S_MT_STRICT=1` |
| Tắt lọc rác | `S2S_ASR_CLEAN=0` |
| Tắt tách câu ở tầng pipeline (lớp MT vẫn tách) | `S2S_MT_SPLIT=0` |
| Đổi route một chiều | sửa một dòng trong `mt_route` của `models.json` |

## 3. Cách chạy trên hộp (theo thứ tự)

```bash
# laptop: đẩy gói lên hộp
python tools/laptop_push.py

# hộp, tài khoản system: kiểm 3.3B có chạy trên NPU không và chất lượng
runuser -u system -- bash -c 'cd /tmp/onevoice/s2s_edge_pkg && . .venv/bin/activate && . tools/env.sh 2>/dev/null && python tools/mt33_check.py 2>&1 | grep -a "^##" | tee /tmp/mt33_check.txt'
runuser -u system -- bash -c 'cd /tmp/onevoice/s2s_edge_pkg && . .venv/bin/activate && . tools/env.sh 2>/dev/null && python tools/mt_rubric.py 2>&1 | grep -a "^##" | tee /tmp/mt_rubric.txt'

# chạy web (mặc định đã là 3.3B)
SKIP_CHECK=1 bash tools/start_box.sh
```

Hai lệnh kiểm thoát mã 2 và in `FAIL: NLLB-3.3B IS NOT IN USE` nếu 3.3B không chạy: khi đó dừng và đọc lỗi, đừng chạy BTC/ADV.

Sau đó chạy BTC và ADV như thường (`tools/run_btc.sh`, `tools/run_adv.sh`, hoặc trên web), rồi kiểm bộ lọc:

```bash
python tools/junk_report.py <results.csv trong zip> --out junk_report.md
```

Báo cáo có 3 phần: mọi chỗ đã cắt (xem có cắt nhầm lời thật không), mảnh ngắn đầu/cuối câu không có trong transcript gốc (ứng viên rác mới), rác còn sót trong MT. Chỉ thêm luật khi có mẫu thật.

## 4. Đã kiểm gì, chưa kiểm gì

Đã kiểm (laptop):
- Test: `python tests/test_asr_clean.py`, `test_mt_clean.py`, `test_asr_clean_pipeline.py` đều qua (ca giữ/ca đổi, 20.000 chuỗi ngẫu nhiên mỗi lớp, cả đường thường lẫn stream, công tắc tắt).
- Lọc rác trên 1.256 câu ASR thật: 22 câu bị đổi, đọc từng câu thấy đều là rác. Transcript gốc không bị cắt (trừ "um" đầu câu).
- Ảnh hưởng tới bản dịch trên 21 câu đó: chrF 68,5 → 72,7, nhưng mẫu nhỏ, không có ý nghĩa thống kê (p ≈ 0,33). Phần chắc chắn là âm thanh không còn đọc "问:", "❚".
- Route 3.3B: BTC 90 câu vi→zh chrF +7,2; rubric tổng 3.3B/route hơn 600M (số trong `docs/mt33_routing.md`). 3.3B đo là bản int8 laptop.

**Chưa kiểm:**
- 3.3B w8a16 trên NPU: chất lượng, thời gian mỗi câu, bộ nhớ khi nạp cùng ASR/TTS.
- ASR w8a16 trên chip có thể sinh rác khác bản CPU, nên luật lọc có thể thiếu hoặc thừa.
- Ca cố ý đánh đổi: tự hỏi rồi tự đáp "没。" ở cuối câu bị cắt; "Hmm, ..." đầu câu bị bỏ "Hmm". Ghi ở `KNOWN_TRADEOFFS` trong `tests/test_asr_clean.py`.

## 5. Bằng chứng

`Downloads\outputOnevoice2\asr_clean\` (README, `mt_effect.csv`, `junk_report.md`) và `Downloads\outputOnevoice2\mt_rubric\`.

## 6. Tách câu, số, kính ngữ tiếng Hàn (13/11, ĐÃ ĐÍNH CHÍNH)

**Đính chính:** bản trước của mục này nói đường thường trên chip dịch cả đoạn một lần và việc tách câu cho chrF +5,3. **Điều đó sai.** Lớp MT trên chip (`Nllb.__call__` trong `mt.py`) vốn đã tách câu từ trước: chip 600M đạt chrF 58,3 trên các đoạn en→vi nhiều câu, ngang mức "tách câu" đo trên laptop. Mức +5,3 chỉ là so với cách laptop dịch cả đoạn, không phải cải thiện trên chip.

**Lỗi thật đã sửa:** bộ tách câu cũ của lớp MT cắt sau mọi dấu "." kể cả khi không có khoảng trắng phía sau:
- "Giá là 15.000 đồng." thành "15." và "000 đồng.";
- "36.5 độ" thành "36." và "5 độ";
- "7 a.m." thành "a." và "m.".
Bộ tách mệnh đề dùng khi câu quá dài cũng cắt "1,000" và "7:30".

Bây giờ:
- Dấu `. ! ? ;` ASCII chỉ là hết câu khi theo sau là khoảng trắng. Dấu `, :` chỉ là ranh giới mệnh đề khi theo sau là khoảng trắng.
- Không cắt sau "Mr.", "Dr.", "a.m.", "TP.".
- Pipeline và lớp MT dùng chung **một** hàm `split_sentences` (nằm trong `mt.py`), nên tách lại lần nữa không đổi gì. Câu cụt như "Vâng." ghép vào câu bên cạnh.

Trên BTC chỉ 1 câu có số kiểu này nên không có số đo chrF. Bằng chứng là test `tests/test_mt_long.py::test_numbers_not_split`: chạy với code cũ thì test thất bại, đúng như mô tả ở trên. `Engine.translate_text` vẫn là đường MT chung cho mọi chỗ gọi (đường thường, stream, gõ chữ trên web, `eval_btc.py`, `eval_adv.py`).

**Kính ngữ tiếng Hàn** (`ko_polite.py`): thêm "-ㄹ까 → -ㄹ까요", "어때 → 어때요", "멈춰 → 멈춰요", "(안) 돼 → 돼요", "(하지) 마 → 마세요". Đuôi "-니까/-습니까" giữ nguyên. Trên 180 câu vi→ko của BTC có 5 câu bị đổi, cả 5 đều gần đáp án hơn (chrF tổng 25,20 → 25,37). Trước đó 167/180 câu đã đúng kính ngữ.

**Đã thử, không làm:** đổi dấu câu ASCII sang toàn khổ trong tiếng Trung (",." → "，。"). chrF không đổi (41,98 → 41,98) vì phép đo đã chuẩn hóa dấu câu, nên không có bằng chứng là có lợi.

**Thời gian:** không đổi so với trước, vì lớp MT trên chip vốn đã tách câu.

**Lỗi MT còn lại** (rubric 54/66 theo route, đều là lỗi từ vựng, chưa sửa vì chưa có cách sửa chung mà không học vẹt bộ test):
- Địa danh Việt: "Đà Nẵng" → "大南" (đúng là 岘港).
- Từ lóng/thành ngữ: "nước ngọt" → "甜水", "buôn dưa lê" → "卖梨", "mới tới" (= 才到) → "新巴士".
- en→vi (600M): "boil" → "nấu" (đúng là "luộc"), "alarm" → "báo động" (đúng là "báo thức"), "minutes" (biên bản) → "hồ sơ".
- Nhiều câu điểm thấp là do ASR nghe sai ("taxi" → "tax", "bắp rang" → "bắ rang"), không phải lỗi MT.

Test: `tests/test_mt_text.py`: tách câu ở cả hai đường, zh nối không khoảng trắng, câu đơn chỉ gọi MT một lần, công tắc tắt, từ viết tắt, kính ngữ (có kiểm lọc hai lần ra cùng kết quả).

## 7. Một câu quá dài thì sao

Đồ thị MT trên chip có giới hạn cứng: đầu vào tối đa 62 token. Đầu ra dừng ở 95 token hoặc ở 2,2 × số token nguồn + 14 (chặn trường hợp lặp vô tận). Code xử lý có sẵn trong `Nllb._tr`, nay đã có test (`tests/test_mt_long.py`):

1. Câu vừa giới hạn và dịch xong (có token kết thúc): dùng luôn.
2. **Câu quá 62 token, hoặc bộ giải mã hết bước mà chưa kết thúc**: cắt theo câu, rồi theo mệnh đề ("," ":" kèm khoảng trắng), rồi chia đôi theo từ, đệ quy đến 6 tầng. Dịch từng mảnh rồi nối lại. Test kiểm không mất chữ nào, đúng thứ tự, không cắt giữa số.
3. Không cắt được nữa (một từ duy nhất): chấp nhận bản cắt cụt và **đếm** vào `mt.truncations` (có trong `bench.py`, `eval_mt.py`), không giấu.

Trên BTC, câu dài nhất sau khi tách là 38 token. Không câu nào chạm giới hạn 62, nên đường cắt ở bước 2 chỉ được kiểm bằng test giả lập, chưa gặp trên dữ liệu thật. Người nói một câu rất dài không ngắt nghỉ (ASR tiếng Việt cần vibert-capu thêm dấu câu) có thể chạm giới hạn.

**Trường hợp chưa xử lý: model tự dừng sớm, bỏ ý dù câu vừa giới hạn.** Ví dụ "Tôi xin được giới thiệu cho các bạn trước chút, vị này là giáo sư…" bị bỏ vế đầu. Đã thử hai cách và **không đưa vào luồng chính**:
- Luôn cắt câu dài (≥ 20 token) theo dấu phẩy: chrF **48,2 → 44,0**, tốt hơn 16 / kém hơn 36 trên 56 câu. Mảnh rời mất ngữ cảnh và sinh ra chữ bịa, ví dụ zh→vi thêm "tôi nhìn thấy một bức ảnh của một người đàn ông".
- Chỉ cắt khi bản dịch nguyên câu ngắn bất thường (< 0,8 lần bản cắt): chỉ bắt được 7/56 câu (5 tốt hơn / 2 kém hơn, +0,5 chrF trung bình). Quá ít để kết luận, và phải dịch hai lần.

Chi tiết: `Downloads\outputOnevoice2\mt_long\long_whole_vs_clause.csv`.

**Đo lại route cho công bằng.** Cột 3.3B trong `mt_compare` được dịch cả đoạn, còn 600M trên chip thì tách câu. Đo lại 3.3B có tách câu như pipeline, so với 600M chip (BTC, khoảng tin cậy 95 % bootstrap):

| Chiều | 600M chip | 3.3B (cũ, cả đoạn) | 3.3B (tách câu) | chênh | Route |
|---|---|---|---|---|---|
| en→vi | 56,18 | 54,87 | 56,77 | +0,6 (−1,2..+2,3) | giữ 600M: không khác biệt, 600M nhanh hơn |
| vi→en | 51,68 | 54,03 | 54,15 | +2,5 (−0,9..+6,1) | giữ 600M: chưa chắc chắn |
| vi→zh | 34,76 | 41,98 | 41,24 | **+6,5 (+2,5..+10,6)** | 3.3B |
| zh→vi | 50,57 | 53,90 | 53,98 | **+3,4 (+0,7..+6,1)** | 3.3B |
| ko→vi | 37,11 | 40,25 | 40,39 | **+3,3 (+0,4..+6,1)** | 3.3B |
| vi→ko | 26,74 | 24,85 | 24,20 | −2,5 (−6,8..+1,7) | 3.3B, vì sửa lỗi đồng âm và rubric 10/10 (xem `mt33_routing.md`) |

Route không đổi. Riêng en→vi trước đây ghi 3.3B "−1,3" là do cách đo; đo đúng thì hai model ngang nhau. Nếu 3.3B trên chip đủ nhanh, có thể thử route vi→en sang 3.3B khi có số thời gian thật. Chi tiết: `Downloads\outputOnevoice2\mt_split\route_fair.csv`.
