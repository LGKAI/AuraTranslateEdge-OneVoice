# Kết quả sửa format TTS hybrid — VI / EN / KO / ZH

## Đã thay đổi gì

Pipeline TTS hiện mặc định: bản dịch → parser có kiểu → lời đọc theo ngôn ngữ → xử lý từ/Unicode → chia đoạn đúng giới hạn graph → TTS. Mỗi span lưu vị trí, chuỗi nguồn, giá trị, loại và lời đọc. Không thay riêng các chuỗi test, không dùng model sinh lại số. Grammar nhận giờ/ngày/tiền/%/đơn vị và số; ngữ cảnh nhận mã; CRF hiện có đề xuất thêm mã ở EN/VI. KO/ZH dùng grammar và từ vựng ngữ cảnh, chưa có CRF riêng được huấn luyện cho hai ngôn ngữ này.

Đã sửa chia đoạn để không cắt giữa từ Latin/Hangul, giữ cụm đã chuẩn hóa khi vừa cửa sổ graph. Giới hạn Supertonic được tính **sau NFKD và thêm language token**, vì một âm tiết Hangul có thể thành nhiều Jamo. Piper được kiểm cả số từ lẫn UTF-8 byte. Đoạn vượt giới hạn không bị cắt âm thầm; audio chạm trần được thử chia nhỏ. Một từ không thể chia mà vượt graph sẽ báo lỗi.

Kế hoạch: [hybrid_format_plan.md](../../docs/hybrid_format_plan.md).

## Kiểm thử và phạm vi kết luận

- **55/55 ví dụ có lời đọc kỳ vọng** qua cả cấu hình có CRF; gồm giờ, trưa/đêm, ngày, tiền, số âm, số thập phân, %, đơn vị, mã có dấu, số 0 đầu, Unicode fullwidth và số lớn Trung.
- **1.600 ca sinh bằng seed cố định** kiểm trường giờ/ngày/thập phân/mã trên bốn ngôn ngữ. Đây là kiểm tra cấu trúc, không phải 1.600 ca nghe thực tế.
- **10 nhóm unittest đều qua**, có ngày mơ hồ/không hợp lệ, thời gian không hợp lệ, ranh giới từ và ngân sách Hangul sau NFKD. Log: [test_hybrid_format.log](../../logs/test_hybrid_format.log).
- **600 số VI/EN/KO** đối chiếu thêm với thư viện độc lập num2words: khớp sau bỏ khác biệt cách nối/zero-hundred. Không coi đây là kiểm chứng nghe hay đánh giá tiếng Trung.
- **50 bản dịch en-vi** được chuẩn hóa lại; **49/49 bản có text tạo WAV thành công**, EV-2748 rỗng nên bỏ qua. Không thay file gốc. Văn bản TTS đã lưu của cả 49 ca được kiểm lại khớp formatter cuối cùng; mọi chunk VI nằm trong 26 từ và 512 byte.
- Tạo **4 audio TTS câu tổng hợp**, mỗi ngôn ngữ một audio; cả bốn chạy thành công. Câu tổng hợp EN trước đây làm neural normalizer lỗi, nay không phụ thuộc bước sinh span của LLM.
- Chạy S2S thật trên **3 audio EN gốc** và **cùng audio VI sang EN/KO/ZH**, tổng 6 ca. Không truyền transcript vào ASR. Một lần VI→KO đầu tiên bị guard giới hạn NFKD chặn; đã sửa cách tính và chạy lại thành công. Không che lỗi lần đầu: log còn tại lịch sử thao tác; log cuối lưu lần chạy đã sửa.
- Tổng **59 WAV mới trong bộ đánh giá** đã kiểm đọc được, có mẫu và hữu hạn: 49 replay TTS + 4 probe TTS + 6 S2S. Thành công tạo WAV không có nghĩa mọi phát âm đúng.
- Server localhost đã cập nhật hybrid, không mở Chrome; một yêu cầu upload audio EV-5301 qua HTTP thực tế đã hoàn tất và có `tám giờ` trong đầu vào TTS. Đây không phải thử mic vật lý.
- Cùng một Translator được thử ca hợp lệ rồi EV-2748 rỗng: không sinh audio và không tái sử dụng transcript/trace TTS của ca trước. [empty_input_check.json](empty_input_check.json).

Chi tiết máy đọc được: [evaluation.json](evaluation.json), [semantic_examples.json](semantic_examples.json), [cardinal_crosscheck.json](cardinal_crosscheck.json).

## Lỗi cũ đã xử lý trong văn bản đưa vào TTS

- VI `5:30 sáng`: “năm : ba mươi” → **“năm giờ ba mươi phút sáng”**.
- VI `08:00`: “tám : không” → **“tám giờ”**.
- VI ZIP `67297`: số đếm → **“sáu bảy hai chín bảy”**. Số nhà `734` giữ **“bảy trăm ba mươi bốn”**, không còn lỗi ByT5 đổi thành “năm ba mươi bốn”.
- EN: giờ/ngày được cấu trúc đầy đủ, tiền USD có dollars/cents; ZIP/điện thoại giữ số 0 đầu; mã đọc từng ký tự. Không còn tách `hundred` thành hai phần.
- KO: `3:30` → **`세 시 삼십 분`**; mã và điện thoại đọc từng ký tự. Giới hạn 96 ký tự tính trên đầu vào graph thực, không cắt Jamo.
- ZH: `3:30` → **`三点三十分`**; năm `2027` → **`二零二七`**; mã/điện thoại đọc từng ký tự. Đồng thời sửa cách đọc thang số 10^12 trong codec Trung (dùng 万亿).

## Hiệu năng và dung lượng

Đo trên cùng 49 bản dịch, 4 lượt sau warm-up, không ghi log, chỉ tính **format CPU**, không gồm G2P, TTS, ASR, MT hoặc chuẩn bị nguồn Qwen:

- Hybrid: trung bình **0,23 ms**, median **0,17 ms**, p95 **0,47 ms**.
- Legacy: trung bình **0,008 ms**. Hybrid làm nhiều việc hơn nhưng phần tăng vẫn dưới 1 ms trong phép đo này.
- Ba file TN Python mới khoảng **27,5 KB**, CRF hiện có **62 KB**. Đây là kích thước file, không phải RAM, không phải toàn bộ dung lượng dự án.
- Nhánh format TTS không cần nạp Qwen/ByT5/T5. Nếu bật `S2S_NEURAL_TN=1`, Qwen vẫn dùng cho chuẩn bị nguồn trước MT và vẫn tốn thời gian/dung lượng riêng. Không xóa các model cũ.

Đây là phép đo cục bộ, tập nhỏ; không phải latency NPU hay cam kết p95 sản xuất. [evaluation.json](evaluation.json) có số đo đầy đủ và hash.

## Đánh giá lại audio và những lỗi còn lại

Whisper Small cục bộ nhận dạng lại **7 audio mới**, không cung cấp prompt tham chiếu. [audio_crosscheck.json](audio_crosscheck.json) lưu nguyên văn; **ASR không thay thế người nghe**:

- Ba ca EN→VI: ASR nghe lại ra **5 giờ 30 phút**, **8 giờ**, **734** và **67297**, phù hợp mục tiêu format. Từ taxi/tên địa chỉ vẫn có sai khác.
- Câu tổng hợp EN: ASR khôi phục đúng giờ, ngày, tiền, nhiệt độ, ZIP và điện thoại; VN bị chép thành VIN.
- Câu tổng hợp VI: ASR nghe lại vẫn sai khác tiền, điện thoại, số nhà dù `tts_text` đúng; cần người nghe phân biệt lỗi ASR nghe lại và lỗi TTS thực.
- Câu tổng hợp KO: ASR nghe lại sai khác ngày/giờ và một chữ số điện thoại; cần người nói tiếng Hàn kiểm tra.
- Câu tổng hợp ZH: ASR nghe lại sai khác năm, tiền, ZIP/điện thoại; cần người nói tiếng Trung kiểm tra.

**Kết luận:** đã cải thiện và kiểm chứng được format tại đầu vào TTS; chưa tuyên bố chất lượng âm thanh hoàn hảo. Cần chấm nghe để xác định các sai khác trên thuộc TTS hay ASR đối chiếu. Chưa có MOS, thử nghe mù hoặc kiểm chứng phát âm mọi chữ cái.

ASR/MT vẫn có lỗi độc lập: 32→2, trưa→tối, từ “báo động”, sai nghĩa thành ngữ; formatter không tự sửa các giá trị nguồn đã sai. Trong audio VI chạy S2S, ASR mất/sai thông tin giờ/ngày/tiền/nhiệt độ rồi MT sai theo. Các kết quả này được giữ nguyên, không thay bằng bản dịch đúng viết tay.

## Audio và transcript

Probe TTS (câu tự soạn để kiểm riêng format, **không phải S2S**):

- [Việt](../multilingual_tts_format/hybrid/vi.wav) — [trace](../multilingual_tts_format/hybrid/vi.json).
- [Anh](../multilingual_tts_format/hybrid/en.wav) — [trace](../multilingual_tts_format/hybrid/en.json).
- [Hàn](../multilingual_tts_format/hybrid/ko.wav) — [trace](../multilingual_tts_format/hybrid/ko.json).
- [Trung](../multilingual_tts_format/hybrid/zh.wav) — [trace](../multilingual_tts_format/hybrid/zh.json).

S2S EN→VI từ WAV gốc en-vi:

- [EV-3775: 5:30](../en_vi_format_replay/hybrid/EV-3775_out.wav) — [input/ASR/MT/TTS](../en_vi_format_replay/hybrid/EV-3775.json).
- [EV-5301: 08:00](../en_vi_format_replay/hybrid/EV-5301_out.wav) — [input/ASR/MT/TTS](../en_vi_format_replay/hybrid/EV-5301.json).
- [EV-0358: số nhà/ZIP](../en_vi_format_replay/hybrid/EV-0358_out.wav) — [input/ASR/MT/TTS](../en_vi_format_replay/hybrid/EV-0358.json).

S2S từ audio VI đặc biệt (còn lỗi ASR/MT):

- [VI→EN](s2s/audio/vi_to_en.wav) — [transcript](s2s/vi_to_en.json).
- [VI→KO](s2s/audio/vi_to_ko.wav) — [transcript](s2s/vi_to_ko.json).
- [VI→ZH](s2s/audio/vi_to_zh.wav) — [transcript](s2s/vi_to_zh.json).

49 output replay bản dịch en-vi: [summary.json](en_vi_tts/summary.json).

## Cấu hình và giới hạn hỗ trợ

`S2S_TTS_FORMAT=hybrid` là mặc định mới. `legacy` phục hồi TN/chia đoạn cũ; `neural` chọn decoder neural EN/VI cũ. `S2S_NEURAL_TN` chỉ điều khiển chuẩn bị nguồn ASR trước MT.

Ngày số mơ hồ như 03/04/2027 được đọc từng ký tự kèm cảnh báo, trừ khi cấu hình `S2S_DATE_ORDER=DMY` hoặc `MDY`. Dấu thập phân mặc định theo ngôn ngữ (VI dấu phẩy, EN/KO/ZH dấu chấm); cần chọn đúng locale nguồn. Dấu `$` hiện quy ước USD, không tự nhận mọi đồng dollar. Grammar `H:MM` mặc định là giờ, chưa phân biệt mọi ngữ cảnh tỷ lệ/thời lượng. Ký hiệu khoa học và khoảng số được phát hiện và cảnh báo, chưa có lời đọc ngữ nghĩa đầy đủ. Không cam kết mọi tiền tệ, đơn vị, lịch, ordinal, phiên bản, Unicode identifier hoặc cách viết trên thế giới. Ngữ cảnh nhận mã và tên chữ cái là tài nguyên mở rộng theo ngôn ngữ, cần thêm kiểm thử thực tế khi mở miền dữ liệu.

Chạy lại unit test:

`/private/tmp/s2s-original-env/bin/python -m unittest discover -s OneVoice_integrated/tests -v`

Harness: `tools/probe_multilingual_tts_format.py hybrid`, `tools/recheck_en_vi_format.py hybrid`, `tools/replay_en_vi_tts_hybrid.py`. Lưu ý `run_e2e.py` cần đặt thư mục kết quả riêng để không ghi đè baseline.

Thiết kế tham khảo [NeMo Hybrid TN](https://github.com/NVIDIA/NeMo-text-processing/blob/main/nemo_text_processing/hybrid/README.md) và [ICU RBNF](https://unicode-org.github.io/icu/userguide/format_parse/numbers/rbnf.html). Triển khai này dùng grammar/codec riêng và CRF sẵn có, không tuyên bố đã tích hợp đầy đủ NeMo hay ICU.
