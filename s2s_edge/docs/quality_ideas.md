# 10 phương án nâng chất lượng tiếp theo

Ngày soạn: 11/11/2026. Bổ sung cho `improvement_playbook.pdf` và `device_plan.pdf`. Không lặp lại những gì đã làm: khôi phục dấu câu và chữ hoa (vibert-capu), luồng B (NLLB-1.3B, beam 4 → 2 → greedy có cơ chế tự hạ cấp), glossary văn phòng.

Nhãn: **[Đo]** số của nhóm; **[Đã đọc]** tài liệu đã đọc đúng đoạn (xem `improvement_playbook.pdf`); **[Chưa đọc]** ý tưởng kỹ thuật, chưa có tài liệu hay số đo; **[Ước]** ước lượng.

## Chất lượng đang mất ở đâu

Từ log 30 clip mỗi chiều chạy trên CPU (`logs/btc_cpu30*.json`, đo trước khi dùng vibert-capu). Đơn vị chrF.

| Chiều | MT từ câu chuẩn | MT từ chữ ASR | "heard" (ASR nghe lại audio đầu ra) | Mất ở ASR → MT | Mất ở TTS |
|---|---|---|---|---|---|
| vi→en | 56,4 | 53,8 | 53,9 | −2,6 | 0 |
| en→vi | 58,3 | 53,0 | 50,3 | −5,3 | **−2,6** |
| vi→ko | 33,0 | 24,1 | 21,6 | **−8,9** | **−2,5** |
| ko→vi | 39,8 | 35,1 | 34,4 | −4,7 | −0,7 |
| vi→zh | 37,7 | 32,4 | 31,5 | −5,2 | −1,0 |
| zh→vi | 50,6 | 50,3 | 50,3 | −0,3 | 0 |

Ba kết luận:

1. **Chỗ mất lớn nhất là giữa ASR và MT** (5 đến 9 điểm). vibert-capu đã lấy lại được một phần (vi→ko từ 25,6 lên 34,1 trên chữ ASR thật). Phần còn lại là lỗi ASR và kiểu câu nói khác kiểu câu viết.
2. **TTS làm mất 2 đến 3 điểm** ở tiếng Việt (Piper) và tiếng Hàn (Supertonic). Một phần có thể do chính ASR dùng để chấm, chứ không hẳn do TTS.
3. **MT gốc yếu nhất ở tiếng Hàn và tiếng Trung.**

## Bảng ưu tiên

| # | Phương án | Đánh vào | Lợi ích kỳ vọng | Chi phí | Cần chip? | Ưu tiên |
|---|---|---|---|---|---|---|
| 1 | Chỉnh tham số giải mã trên bộ dev | MT | +0,5 đến 1 **[Ước]** | 2 đến 3 giờ (laptop) | Không | **Cao** |
| 2 | Bộ lọc đầu ra hỏng và tự dịch lại | MT | tránh câu hỏng nặng | 2 giờ | Không | **Cao** |
| 3 | Chuẩn bị riêng cho kịch bản demo 30 phút | ASR, MT, TTS | lớn với demo | nửa ngày | Không | **Cao** |
| 4 | Giảm mất mát ở TTS | TTS | +1 đến 2 ở vi, ko **[Ước]** | nửa ngày | Một phần | Cao |
| 5 | Chống ồn: đo và chọn cách xử lý | ASR | chưa biết | 1 ngày | Một phần | Cao (field test ồn) |
| 6 | Tách câu trước khi dịch | MT | 0 đến +1 **[Ước]** | 1 giờ | Không | Trung bình |
| 7 | Chọn lại bằng dịch ngược (round-trip) | MT | chưa biết **[Chưa đọc]** | 1 giờ thử, rồi đo độ trễ | Không | Trung bình |
| 8 | Định tuyến ngôn ngữ chắc hơn trong chế độ hội thoại | ASR | tránh dịch sai chiều | nửa ngày | Không | Trung bình (demo) |
| 9 | Thử ASR tiếng Anh khác SenseVoice | ASR | chưa biết | 1 ngày | Có | Trung bình |
| 10 | Fine-tune nhẹ NLLB (LoRA) cho hội thoại và đầu vào ASR | MT | lớn nhất nhưng chưa đo | 2 đến 3 ngày | Biên dịch lại | Sau 16/11 |

---

## 1. Chỉnh tham số giải mã MT trên bộ dev

**Vấn đề.** Tham số giải mã đang đặt theo mặc định: `length_penalty=1.0`, chặn lặp 3-gram, giới hạn độ dài `2,2 × nguồn + 14`. Beam ảnh hưởng rất mạnh: bỏ beam ở 600M làm vi→ko tụt từ 32,1 xuống 22,7 **[Đo]**. Nên các tham số khác của beam cũng đáng chỉnh.

**Cách làm.** Quét `length_penalty` (0,6 đến 1,4), `no_repeat_ngram` (0, 3, 4), giới hạn độ dài, trên **bộ dev** (không đụng holdout), riêng cho từng chiều. Dùng NLLB trên laptop, mô phỏng đúng vòng beam của `mt.py`. Tham số nằm trong mã máy chủ nên **không cần biên dịch lại**; chỉ cần đọc từ `models.json` theo từng chiều.

**Đo.** chrF trên bộ dev, kèm `compare_runs.py` (bootstrap) giữa cấu hình mới và mặc định.

**Rủi ro.** Chỉnh quá khớp với dev. Chỉ nhận thay đổi có khoảng tin cậy dương, và xác nhận một lần trên holdout.

## 2. Bộ lọc đầu ra hỏng và tự dịch lại

**Vấn đề.** Đôi khi MT ra câu hỏng hẳn:
- lặp vô hạn (đã thấy "제곱하기 제곱하기…" khi chạy NLLB không có chặn lặp **[Đo]**);
- dịch rỗng, hoặc chép nguyên câu nguồn;
- sai hệ chữ (câu đích tiếng Hàn mà ra chữ Latin);
- quá ngắn hoặc quá dài so với câu nguồn.

Mỗi câu hỏng mất trọn điểm câu đó, và rất dễ bị giám khảo nhận ra khi nghe.

**Cách làm.** Kiểm sau MT, chỉ trên CPU, tốn vài micro giây:
- tỉ lệ độ dài đích/nguồn;
- tỉ lệ ký tự đúng hệ chữ của ngôn ngữ đích;
- n-gram lặp;
- trùng khít với câu nguồn.

Nếu câu không qua, dịch lại bằng cách khác: đổi beam ↔ greedy, hoặc tách câu theo mệnh đề (đã có `_split`). Chọn bản đạt kiểm tra. Ghi log mỗi lần kích hoạt.

**Đo.** Đếm số câu bị bắt trên toàn bộ 920 câu BTC và đọc tay từng câu. Thời gian thêm chỉ tốn ở câu bị bắt.

## 3. Chuẩn bị riêng cho kịch bản demo 30 phút

**Vấn đề.** Brief yêu cầu mỗi đội tự chuẩn bị **một kịch bản hội thoại khoảng 30 phút** trong bối cảnh đã đăng ký ở vòng 2, thử ở VNG Campus có ồn nhẹ (`phase3_brief.pdf`, mục 4). Đây là phần giám khảo trực tiếp nhìn và nghe.

**Cách làm.**
1. Viết kịch bản trước.
2. Cho từng câu chạy qua pipeline: đọc thật bằng micro, hoặc dùng TTS để tạo âm thanh.
3. Ghi lại mọi lỗi:
   - ASR nghe sai → thêm vào `_asr_fix`;
   - TTS đọc sai từ Latin → thêm cách đọc vào glossary;
   - tên riêng sang tiếng Trung, tiếng Hàn sai → thêm vào glossary;
   - MT dịch sai thuật ngữ → thêm cặp thay thế.
4. Lặp đến khi cả kịch bản sạch lỗi.

Danh sách thuật ngữ lấy thẳng từ kịch bản. Đây là bài toán "danh sách đóng", nên các kỹ thuật ưu tiên từ vựng mới phát huy được.

**Chi phí.** Nửa ngày, không cần mã mới (dùng `glossary_enterprise.json` và `live_selftest.py`).

## 4. Giảm mất mát ở TTS

**Vấn đề.** "heard" thấp hơn chữ dịch 2,5 điểm ở vi→ko (Supertonic tiếng Hàn) và 2,6 điểm ở en→vi (Piper tiếng Việt) **[Đo]**.

**Cách làm, theo thứ tự rẻ đến tốn:**
1. **Tìm câu hỏng:** so từng câu `mt_e2e` với `heard` trong CSV của `eval_btc.py`, lọc các câu chênh nhiều để biết lỗi do đâu (số, từ ngoại lai, câu dài bị cắt cửa sổ, đọc quá nhanh).
2. **Chuẩn hóa văn bản trước TTS:** số, đơn vị, viết tắt, ký hiệu (%, $, giờ).
3. **Số bước Supertonic:** bài báo cho thấy chất lượng gần như phẳng từ 8 bước trở lên, sụp ở 4 bước (arXiv 2503.23108, Bảng 3) **[Đã đọc]**. Cần kiểm số bước của đồ thị đang biên dịch.
4. **Tốc độ nói và khoảng ngắt** giữa các cửa sổ.

**Lưu ý.** "heard" dùng ASR của chính mình để chấm, nên một phần chênh lệch có thể do ASR chấm sai. Nên nghe tay khoảng 20 câu chênh nhiều nhất trước khi sửa.

## 5. Chống ồn: đo và chọn cách xử lý

**Vấn đề.** Khu thử có ồn nhẹ (quạt, người qua lại). Pipeline chưa được đo trong điều kiện ồn.

**Đồ nghề đã sẵn:** 3 bản BTC có ồn (`onevoice_btc_noisy`), `mix_noise.py --kind file` để trộn tiếng phòng thu thật, giai đoạn A9 của runbook.

**Cách làm.**
1. Đo mức tụt chất lượng theo SNR (20, 10, 5 dB).
2. Thử lần lượt:
   - lọc ồn GTCRN cộng **trộn lại một phần tín hiệu gốc** (observation adding; Iwamoto và cộng sự, Mục 3.2 và 4.3 **[Đã đọc]**: cách này cải thiện ASR mà không cần huấn luyện);
   - lọc thông cao (bỏ tiếng ù của quạt dưới khoảng 100 Hz);
   - chỉnh ngưỡng VAD.
3. Chỉ giữ cách có cải thiện có ý nghĩa thống kê.

**Kỳ vọng thực tế.** Lọc ồn rời rạc thường không giúp ASR, và GTCRN đã từng không giúp **[Đo]**. Bước quan trọng nhất là **biết pipeline tụt bao nhiêu** để chuẩn bị (vị trí micro, micro định hướng, khoảng cách nói).

## 6. Tách câu trước khi dịch

**Vấn đề.** `mt.py` chỉ tách câu khi đầu vào vượt 64 token. Một lượt nói gồm 2 hoặc 3 câu sẽ được dịch thành một khối, trong khi NLLB học chủ yếu trên từng câu. Đã thấy trường hợp hai câu dính nhau bị dịch sai nghĩa ("tôi rất hài lòng bao nhiêu tiền") **[Đo]**.

**Cách làm.** Sau khi vibert-capu đã đặt dấu chấm và dấu hỏi, luôn tách tại ranh giới câu rồi dịch từng câu (cơ chế `_split` đã có). Thử trên laptop, khoảng 1 giờ.

**Rủi ro.** Mất ngữ cảnh giữa các câu (đại từ, chủ ngữ bị lược). Chỉ giữ nếu chrF trên dev tăng.

## 7. Chọn lại bằng dịch ngược (round-trip)

**Vấn đề.** Lỗi sai nghĩa từ của NLLB, ví dụ "máy lạnh" → 냉장고 (tủ lạnh) **[Đo]**.

**Ý tưởng [Chưa đọc].** Lấy các ứng viên của beam (2 đến 4 câu), dịch ngược từng câu về ngôn ngữ nguồn bằng greedy, rồi chọn ứng viên có bản dịch ngược giống câu nguồn nhất (chrF). Bản dịch "tủ lạnh" khi dịch ngược sẽ khác "máy lạnh" và bị trừ điểm.

**Chi phí và rủi ro.** Mỗi ứng viên tốn thêm một lượt encoder và giải mã greedy: khoảng +0,3 đến 0,6 s mỗi ứng viên với 600M trên chip **[Ước]**. Có thể chỉ áp dụng cho câu ngắn. Tôi chưa tìm và đọc tài liệu nào đo cách này, nên phải thử trên laptop trước khi tin.

## 8. Định tuyến ngôn ngữ chắc hơn trong chế độ hội thoại

**Vấn đề.** Chế độ hội thoại (`--pair vi,ko`) tự đoán người vừa nói là tiếng Việt hay tiếng kia. Trên dữ liệu sạch đoán đúng 100 %, nhưng sai khi câu ngắn (1,5 s) hoặc ồn (5 dB) **[Đo]**. Đoán sai thì cả lượt dịch hỏng. Đây là lỗi dễ thấy nhất khi demo.

**Cách làm.**
1. Kết hợp nhãn ngôn ngữ của SenseVoice với độ tự tin của Zipformer.
2. Câu quá ngắn hoặc độ tự tin thấp thì giữ chiều của lượt trước.
3. Thêm lựa chọn **hai nút bấm**, mỗi người một nút, để người dùng tự chọn chiều. Cách này chắc chắn đúng và hợp với mục giao diện, trải nghiệm.

**Đo.** Bộ hội thoại trộn có sẵn (`samples/mixed_conversation.wav`) cộng các bản có ồn.

## 9. Thử ASR tiếng Anh khác SenseVoice

**Vấn đề.** WER tiếng Anh thật khoảng 11,5 % sau khi bỏ từ đệm và lỗi dữ liệu **[Đo]**. Đây là ASR yếu nhất, gây mất 5,3 điểm ở en→vi.

**Cách làm.** So một ASR tiếng Anh khác (ví dụ một bản Whisper cỡ nhỏ) với SenseVoice trên đúng các clip en→vi, đo trên laptop trước. Nếu hơn rõ, mới tìm hoặc biên dịch bản cho QCS8550. Tôi **chưa kiểm** AI Hub có sẵn bản nào cho QCS8550 hay không.

**Rủi ro.** Thêm một mô hình ASR nữa, nặng hơn và chậm hơn. Mô hình có bước giải mã tự hồi quy (như Whisper) khó chạy tĩnh trên NPU hơn mô hình CTC.

## 10. Fine-tune nhẹ NLLB (LoRA) cho hội thoại và đầu vào ASR

**Vấn đề.** Đây là chỗ mất lớn nhất (bảng đầu tài liệu). NLLB học trên văn bản viết, còn đầu vào thật là lời nói có lỗi ASR, kiểu khẩu ngữ du lịch và văn phòng.

**Tài liệu [Đã đọc]:**
- Di Gangi và cộng sự (2019), Bảng 3: fine-tune trên đầu vào ASR có lỗi tăng 24,5 lên 25,9 BLEU, và nên học đồng thời dữ liệu sạch lẫn dữ liệu nhiễu.
- Bapna và cộng sự (2019), Bảng 1: adapter đạt ngang fine-tune toàn bộ khi thích ứng miền.
- LoRA (tóm tắt bài): gộp vào trọng số được nên không tăng độ trễ.

**Cách làm.**
1. **Dữ liệu:** cặp câu hội thoại vi↔en/zh/ko từ nguồn có giấy phép rõ ràng. Cần kiểm nguồn và giấy phép trước. **Không dùng BTC.**
2. **Mô phỏng đầu vào ASR:** tạo phía nguồn kiểu ASR (chữ thường, bỏ dấu câu, lỗi ASR mô phỏng hoặc lấy từ chính ASR của mình), trộn với bản sạch.
3. **Huấn luyện:** LoRA trên RTX 4050.
4. **Đưa lên chip:** gộp vào trọng số, xuất lại ba đồ thị như hiện nay, biên dịch và link trên AI Hub, kiểm bằng `infer_check.py`.

**Chi phí.** 2 đến 3 ngày tính cả dữ liệu và biên dịch lại. **Không kịp trước 16/11**, hợp cho vòng field test (28/11).

**Rủi ro.** Tăng chiều này nhưng giảm chiều khác. Phải đo đủ 6 chiều.

---

## Đã cân nhắc và loại (có số)

| Phương án | Lý do loại |
|---|---|
| Dịch bắc cầu qua tiếng Anh | kém hơn dịch thẳng 3 đến 10 chrF ở cả 4 chiều **[Đo]** |
| 1.3B greedy | thua 600M beam 4 (vi→ko 26,9 so với 32,1) **[Đo]** |
| Mô hình ngôn ngữ n-gram cho ASR, CTC-WS cho bộ test ẩn | ASR tiếng Việt đã đạt WER khoảng 0,8 %; đồ thị SenseVoice không có xác suất từng khung; không biết trước tên riêng |
| MBR trên 4 ứng viên | tài liệu chỉ cho kết quả với hàng trăm mẫu và độ đo nơ-ron, không áp dụng được trên thiết bị |
| Chỉnh đuôi câu tiếng Hàn | 27/30 câu tham chiếu ở kiểu lịch sự, MT đã là 24–25/30; chênh nhỏ **[Đo]** |

## Thứ tự đề xuất

- **Trước 16/11:** 2 (bộ lọc đầu ra), 3 (kịch bản demo), 1 (tham số giải mã), 6 (tách câu). Rẻ, không cần biên dịch lại.
- **Khi có hộp:** 5 (ồn) và 4 (TTS), đo bằng runbook A9 và "heard".
- **Vòng field test:** 10 (LoRA), 9 (ASR tiếng Anh), 7 (dịch ngược), 8 (hai nút bấm).
