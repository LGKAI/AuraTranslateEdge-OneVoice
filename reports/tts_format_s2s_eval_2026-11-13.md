# Báo cáo kiểm thử TTS Hybrid Format (4 ngôn ngữ) & Speech-to-Speech E2E

**Ngày kiểm thử:** 2026-11-13  
**Mục tiêu:** Tài liệu bàn giao kỹ thuật cho agent tiếp theo về hiện trạng định dạng văn bản TTS (`S2S_TTS_FORMAT=hybrid` vs `legacy`) trên cả 4 ngôn ngữ (`vi`, `en`, `ko`, `zh`) và kết quả chạy Speech-to-Speech (S2S) đầy đủ trên cả 6 chiều dịch (`en↔vi`, `ko↔vi`, `zh↔vi`).

---

## 1. Phạm vi, môi trường & quy ước đọc báo cáo

* **Workspace:** `/Users/khoa/study/Onevoice_final_ves/OneVoice_E2E_HybridTTS`
* **Python runtime:** `/private/tmp/s2s-original-env/bin/python` (Python 3.11.15; có `onnxruntime`, `numpy`, `scipy`, `pycrfsuite`, `num2words`, `mlx_whisper`)
* **Registry & graph ONNX CPU:** `models/models.cpu.json`
  * ASR: Zipformer CTC (`zip150_ctc_static_1500.onnx` cho `vi`), SenseVoice (`sv_frontend_v3.onnx` + `sv_enc_cls.onnx` cho `en`, `ko`, `zh`)
  * Punctuation / Source TN: `viBERT-CAPU` mặc định khi `S2S_NEURAL_TN=0` (`S2S_VI_PUNCT=model`); `neural_textnorm` (`mlx_lm` với `models/source/qwen3-4b-instruct-mlx4`, `Apple Metal / MLX 4-bit`) khi `S2S_NEURAL_TN=1`
  * MT: `NllbBeam` (`nllb_enc.onnx` + `nllb_decB.onnx`, log `[s2s] MT decoder in use: mt_decB`)
  * TTS: `TtsVi` (`piper_vi_full_g8.onnx`, 22050 Hz, budget `26` words / `512` UTF-8 bytes), `TtsEnKo` (`supertonic_full_N96_L150_s5.onnx`, 44100 Hz, budget `N=96` sau NFKD + `<lang>`), `TtsZh` (`melo_one.onnx`, 44100 Hz, budget `20` chars / `112` UTF-8 bytes)
* **Whisper đối chứng:** `mlx_whisper` dùng model tại `/private/tmp/onevoice-audit-whisper`
* **Ràng buộc runtime / Metal:** Các model ONNX CPU (`AsrVi`, `AsrXx`, `viBERT-CAPU`, `NllbBeam`, `TtsVi`, `TtsEnKo`, `TtsZh`, `HybridNormalizer`, `IdentifierTagger`) chạy hoàn toàn trên CPU không cần Metal. Riêng `mlx_whisper` và `S2S_NEURAL_TN=1` dùng Apple Metal (`mlx`), sẽ lỗi `RuntimeError: [Event::Event] Failed to create Metal shared event.` nếu chạy trong môi trường không cho phép tạo Metal shared event.
* **Trạng thái repo:** Không chỉnh sửa file mã nguồn nào trong `s2s_edge/` hoặc `tests/` trong phiên test này. Toàn bộ script và dữ liệu thô (JSON + WAV) được lưu tại `/Users/khoa/.gemini/antigravity/brain/3375595f-fc60-4f1f-aff8-333160691ee7/scratch/`.
* **Lưu ý phương pháp:** Back-ASR bằng ONNX (`AsrVi`/`AsrXx`) và `mlx_whisper` là phép kiểm tra độ rõ âm thanh tự động (intelligibility proxy), **không thay thế đánh giá nghe MOS của người thật**. Mọi giá trị `TtsVi.oov` trong code (`s2s_edge/s2s_edge/tts.py#L83,L92`) được cộng dồn theo instance, do đó bảng dưới đây báo cáo đúng **delta OOV** cho từng lượt `synthesize()`.

---

## 2. Kết quả Unit / Regression Tests của module Format

Lệnh chạy tại `OneVoice_E2E_HybridTTS/`:

```sh
PYTHONPATH=s2s_edge /private/tmp/s2s-original-env/bin/python -m unittest discover -s tests -v
```

* **Kết quả:** `Ran 15 tests in 0.109s` → `OK` (`tests/test_hybrid_format.py` + `tests/tests_identifiers.py`).

---

## 3. Kiểm thử TTS Format trực tiếp trên 4 ngôn ngữ (`vi`, `en`, `ko`, `zh`)

Script thực thi: `scratch/eval_tts_format_4langs.py`, phần đo delta OOV riêng cho `vi`: `scratch/eval_extra_s2s.py`, phần back-ASR: `scratch/eval_onnx_backcheck.py` và `scratch/eval_mlx_metal.py`.

### 3.1. Đối chiếu `legacy` vs `hybrid` trên câu probe chuẩn

Mỗi ngôn ngữ chạy cùng một kịch bản nội dung: giờ `3:30` (`3:30 PM` cho EN), ngày `21/12/2027` (`2027년 12월 21일` / `2027年12月21日`), tiền `$18.25` / `18.25달러` / `18.25美元`, phần trăm `12,5%` / `12.5%`, nhiệt độ `36,8°C` / `36.8°C`, ZIP `04782`, SĐT `0908 246 135`, mã bay `VN247` (câu `vi` có thêm `Số nhà 734.`).

| Ngôn ngữ | Mode | Thời gian format / synth / độ dài audio | Đầu ra chuẩn hóa thực tế đi vào TTS (`tts_input_text`) & trace | Kiểm tra ngược (`onnx_back_asr` / `whisper_back_asr`) |
| :--- | :--- | :--- | :--- | :--- |
| **`vi`** | `legacy` | `—` / `2.515s` / `18.495s` (`22050 Hz`, `oov_delta=1`) | `"cuộc họp lúc ba : ba mươi chiều ngày hai mươi mốt / mười hai / hai nghìn không trăm hai mươi bảy . giá vé $ mười tám phẩy hai năm , giảm mười hai phẩy năm phần trăm. nhiệt độ ba mươi sáu phẩy tám °c. mã dip bốn nghìn bảy trăm tám mươi hai . điện thoại chín trăm lẻ tám hai trăm bốn mươi sáu một trăm ba mươi lăm . chuyến bay vơn hai trăm bốn mươi bảy . số nhà bảy trăm ba mươi bốn ."` | ONNX: mất số `0` đầu (`"bốn nghìn bảy trăm tám hai"`, `"chín trăm lẻ tám..."`), mất `"độ"` trước `"c"`, `"VN"` thành `"vần"`. |
| **`vi`** | `hybrid` | `12.224 ms` / `2.3244s` / `18.315s` (`peak=0.8612`, `rms=0.1382`, `oov_delta=1`) | `normalized_text`: `"Cuộc họp lúc ba giờ ba mươi phút chiều ngày hai mươi mốt tháng mười hai năm hai nghìn không trăm hai mươi bảy. Giá vé mười tám đô la hai mươi lăm xen, giảm mười hai phẩy năm phần trăm. Nhiệt độ ba mươi sáu phẩy tám độ xê. Mã ZIP không bốn bảy tám hai. Điện thoại không chín không tám hai bốn sáu một ba năm. Chuyến bay vê en hai bốn bảy. Số nhà bảy trăm ba mươi bốn."` (`tts_input_text` đổi `"ZIP"` → `"dip"` do `latin_to_vi`). `9` spans, `0` warnings, `4` chunks. | ONNX: giữ đúng `không bốn bảy tám hai`, `không chín không tám hai bốn sáu một bà năm`, `vn hai bốn bảy`, `độ c`. Whisper: `"3h30 phút chiều ngày 21 tháng 12 năm 2027, giá vẽ 18,25 do là 12,5% nhiệt độ 36,8 độ C, mã 04782, 9,249, 08246, 1,3,5, 9 bay, VN247, Vô Nga 7, 130,4."` (SĐT 10 số đọc liên tiếp dễ bị dính âm). |
| **`en`** | `legacy` | `—` / `10.2415s` / `28.079s` (`44100 Hz`) | `"The meeting is at three : thirty PM on twenty-one / twelve / two thousand twenty-seven . The ticket costs $ eighteen point two five , with a twelve point five percent discount. The temperature is thirty-six point eight °C. My ZIP code is four thousand seven hundred eighty-two . Phone nine hundred eight two hundred forty-six one hundred thirty-five . Flight VN two hundred forty-seven ."` | ONNX: `"3 30 pm on 21122027"`, `"36.8 to C"`, `"4,782"`, `"908-246,135"`, `"Flight VN 247"`. |
| **`en`** | `hybrid` | `0.993 ms` / `8.5128s` / `26.465s` (`peak=0.5602`, `rms=0.0507`) | `"The meeting is at three thirty pee em on December twenty first two thousand twenty-seven. The ticket costs eighteen dollars twenty-five cents, with a twelve point five percent discount. The temperature is thirty-six point eight degrees Celsius. My ZIP code is zero four seven eight two. Phone zero nine zero eight two four six one three five. Flight vee en two four seven."` `8` spans, `0` warnings, `6` chunks (`preprocessed_chunk_lengths`: `[50, 57, 95, 77, 50, 94] <= 96`, `audio_cap_retries=[]`). | ONNX: `"December 21st 2027"`, `"$18.25"`, `"12.5%"`, `"36.8 degrees Celsius"`, `"04782"`, `"0908-24,6,1,3,5"`. Whisper: gần như 100% chính xác (`"3.30 p.m."`, `"December 21, 2027"`, `"$18.25"`, `"12.5%"`, `"36.8 degrees Celsius"`, `"04782"`, `"0908 246135"`, `"V in 247"`). |
| **`ko`** | `legacy` | `—` / `5.3748s` / `24.015s` (`44100 Hz`) | `"회의는 이천이십칠 년 십이 월 이십일 일 삼 : 삼십 에 있습니다. 가격은 십팔 점 이 오 달러이고 십이 점 오 퍼센트 할인됩니다. 온도는 삼십육 점 팔 °C입니다. 우편번호는 사천칠백팔십이 입니다. 전화번호는 구백팔 이백사십육 백삼십오 입니다. 항공편은 VN 이백사십칠 입니다."` | ONNX: `"3 30"`, `"36.8더"`, `"4782"`, `"98 246, 135"`. |
| **`ko`** | `hybrid` | `0.628 ms` / `7.2288s` / `24.265s` (`peak=0.4794`, `rms=0.0483`) | `"회의는 이천이십칠 년 십이 월 이십일 일 세 시 삼십 분에 있습니다. 가격은 십팔 달러 이십오 센트이고 십이 점 오 퍼센트 할인됩니다. 온도는 삼십육 점 팔 도입니다. 우편번호는 영 사 칠 팔 이입니다. 전화번호는 영 구 영 팔 이 사 육 일 삼 오입니다. 항공편은 브이 엔 이 사 칠입니다."` `8` spans, `0` warnings, `5` chunks (`preprocessed_chunk_lengths`: `[88, 84, 86, 66, 48] <= 96`, `audio_cap_retries=[]`). | Whisper: nhận đúng `04782`, `VN247`, `36.8도`, `12.5%`; bị nghe nhầm `"이십일 일 세 시"` → `"20일 1시"`, `"센트"` → `"cm"`, `"135"` → `"145"`. ONNX cũng bị tương tự ở `"20일1 3시"`. |
| **`zh`** | `legacy` | `—` / `13.5197s` / `21.038s` (`44100 Hz`) | `"会议在二千零二十七年十二月二十一日三:三十开始。价格是十八点二 五美元，折扣百分之十二点五。温度是三十六点八°C。邮政编码是四千七百八十二。电话号码是九百零八 二百四十六 一百三十五。航班号是VN二百四十七。"` | ONNX: `"33时"`, `"36.8。"` (rớt `°C`), `"4782"`, `"908246，13我"`, `"航班耗时247"`. |
| **`zh`** | `hybrid` | `0.705 ms` / `10.7255s` / `19.877s` (`peak=0.5343`, `rms=0.0615`) | `"会议在二零二七年十二月二十一日三点三十分开始。价格是十八美元二十五美分，折扣百分之十二点五。温度是三十六点八摄氏度。邮政编码是零四七八二。电话号码是零九零八二四六一三五。航班号是维恩二四七。"` `8` spans, `0` warnings, `7` chunks, `audio_cap_retries=[]`. | Whisper: nhận rất tốt (`3點30分`, `18美元25美分`, `百分之12.5`, `36.8攝氏度`, `04782`, `維安247`; chỉ lệch `2027`→`2020`, `0908`→`098`). ONNX nhận đúng cả `2027年12月21日3点30分` & `0908246135` (chỉ nhầm `18`→`48`, `维恩`→`为N`). |

### 3.2. Danh sách bug edge case tìm được trong chế độ `hybrid`

Tất cả các ca sau đều tái lập trực tiếp trong `tts_format_eval/tts_format_4langs_report.json` (`edges`) và `extra_s2s_eval.json` (`vi_oov_deltas`):

1. **Bug 1 (High) — SĐT quốc tế có `+`/gạch nối hoặc ngoặc đơn bị vỡ thành số âm/dương và dính chữ:**
   * **Input `vi`:** `"Gọi +84-908-246-135, code AB-012. Tỷ lệ 3-5, hệ số 1e-3, âm -12,50%."`
     * **Actual `tts_input_text`:** `"gọi dương tám mươi bốnâm chín trăm lẻ támâm hai trăm bốn mươi sáuâm một trăm ba mươi lăm, ..."` (`oov_delta=3`).
   * **Input `en`:** `"Flight VN 123, ZIP 67297, phone (202) 555-0123, +1-800-555-0199."`
     * **Actual `normalized_text`:** `"... phone (two zero two close parenthesis five five five dash zero one two three, plus oneminus eight hundredminus five hundred fifty-fivedash zero one nine nine."`
   * **Input `zh`:** `"... 电话0908-246-135。"`
     * **Actual `normalized_text`:** `"... 电话零九零八负二百四十六负一百三十五。"` (riêng `0908` ăn warning leading-zero, `-246`/`-135` biến thành số âm).
   * **Nguyên nhân gốc (`s2s_edge/s2s_edge/hybrid_textnorm.py` + `s2s_edge/s2s_edge/format_locales.py`):**
     * `IDENTIFIER_CUES` (`format_locales.py#L13-L19`) có `'gọi số'`, `'điện thoại'`, `'电话号码'`, nhưng thiếu `'gọi'` và `'电话'`.
     * `_context_identifier` (`hybrid_textnorm.py#L166-L173`) giới hạn `[^\d,，;；]{0,16}$`, nên khi có 2 SĐT kế nhau sau `'phone'` (`(202) 555-0123, +1-800-...`), SĐT thứ hai vượt ngưỡng bị `context=False` và rớt xuống `NUM` (`priority=5`).
     * Khi chuỗi `+84-908-246-135` bị rớt sang `NUM` (`r'[+\-−]?\d+(?:[.,，．]\d+)*'`), dấu `-` trước `908`/`246`/`135` bị ăn vào làm dấu âm nên không còn ký tự ngăn cách giữa các span liền kề (`...bốnâm...`, `oneminus...`), và trong tiếng Việt các từ ghép như `bốnâm`, `támâm`, `sáuâm` tạo ra OOV (`oov_delta=3`).
     * Regex `identifier` (`hybrid_textnorm.py#L303`: `r'(?<![A-Za-z\d])(?:\+?\d[\d ()-]*\d|...)'`) bắt đầu bằng `\d` nên bỏ sót ký tự `(` đầu trong `(202) 555-0123`, rồi lại đọc `)` thành `close parenthesis`.
     * *Lưu ý đối chiếu:* Ở ca `ko` (`"전화번호 +82-10-1234-5678"`), vì có cue `'전화번호'` ngay sát nên regex bắt trọn `+82-10-1234-5678` và đọc đúng `"플러스 팔 이 하이픈..."`.

2. **Bug 2 (High) — Dấu phẩy tiếng Trung không khoảng trắng ghép hai số khác nhau & chữ Hán chặn lookaround `unsupported`:**
   * **Input `zh`:** `"1000000000000元，1000000001000，3-5天，系数1e-3。"`
   * **Actual `normalized_text`:** `"一万亿元，一万亿零一千点三负五天，系数一e负三。"`
   * **Spans thực tế:** `[('money', '1000000000000元', '一万亿元'), ('number', '1000000001000，3', '一万亿零一千点三'), ('number', '-5', '负五'), ('number', '1', '一'), ('number', '-3', '负三')]` (`0 warnings`).
   * **Nguyên nhân gốc (`s2s_edge/s2s_edge/hybrid_textnorm.py#L21,L295-L296`):**
     * Trong Python `re` (Unicode mặc định), `\w` khớp luôn chữ Hán! Ở dòng 295 `(?<![\w-])\d+(?:[.,]\d+)?[eE][+\-]?\d+(?!\w)` bị trượt vì trước `1` là chữ `"数"`; ở dòng 296 `(?<![\w-])\d+...(?![\w-])` bị trượt vì sau `5` là chữ `"天"`. Do không có span `unsupported` (`priority=4`), `NUM` (`L21`: `r'[+\-−]?\d+(?:[.,，．]\d+)*'`) coi dấu `，` là dấu phân tách số và ghép `"1000000001000，3"` thành một số thập phân (`1000000001000.3`).

3. **Bug 3 (Medium) — Regex `time` nuốt dấu chấm hết câu sau `AM.` / `PM.`:**
   * **Input `vi`:** `"... lúc 00:05 và 12:00 PM. Tốc độ 100km/h..."` → `raw='12:00 PM.'` → `"mười hai giờ trưa Tốc độ..."` (mất dấu `.` ngắt hai câu).
   * **Input `ko`:** `"... 08:00에 있습니다. 12:00 PM."` → `raw='12:00 PM.'` → `"오후 열두 시"` (mất dấu `.` cuối câu).
   * **Nguyên nhân gốc (`s2s_edge/s2s_edge/hybrid_textnorm.py#L290`):** phần `(?:\s*(?P<ampm>[aApP]\.?[mM]\.?))?` khớp tham lam dấu `.` cuối ở `PM.`, và khi `render('time')` trả về chuỗi nói thì không giữ lại dấu chấm kết thúc câu.

4. **Bug 4 (Medium) — Viết tắt `14h`, `"độ C"` (VI) và `"3시 30분"` (KO):**
   * `vi` `"14h"`: pattern `time` yêu cầu phút `:\d{2}` / `h\d{2}`, nên `14` chỉ khớp `number` → `"mười bốnh"` (`oov_delta=1` ở `edges['vi'][1]`, trong khi `"14h45"` khớp đúng `"mười bốn giờ bốn mươi lăm phút"`).
   * `vi` `"37.5 độ C"` (rất phổ biến trong đầu ra NLLB S2S): `UNITS` (`format_locales.py#L37-L40`) chỉ có `'°c'`, `'℃'`, `'°'`, không có `'độ c'` nên giữ nguyên `'độ c'` (không gây tăng OOV ở Piper, nhưng kém chuẩn hóa khi viết bằng chữ).
   * `ko` `"오후 3시 30분"`: không khớp `time` (chỉ bắt `:`/`：`/`h`/`H`), đọc thành Hán-Hàn `"삼시 삼십분"` thay vì thuần Hàn `"세 시 삼십 분"` (khác với `"3:30"` đọc chuẩn `"세 시 삼십 분"`).

5. **Bug 5 (Low/Medium) — Mã tách khoảng trắng (`"VN 123"`, `"VN 247"`):**
   * `"VN247"` (viết liền) được đọc chữ cái + từng số trên cả 4 ngôn ngữ (`"vê en hai bốn bảy"`, `"vee en two four seven"`, `"브이 엔 이 사 칠"`, `"维恩二四七"`).
   * Khi NLLB hoặc văn bản gốc tách khoảng trắng thành `"VN 247"` / `"VN 123"`, regex `identifier` chỉ bắt `247`/`123`, để phần `"VN"` lại (`"vơn"` qua `latin_to_vi` ở `vi`, `"VN"` ở `en`/`ko`/`zh`).

6. **Điểm hoạt động tốt ở edge cases:**
   * Sinh `warnings` bảo thủ và đánh vần ký tự cho ngày mơ hồ `03/04/2027` (`S2S_DATE_ORDER` chưa đặt), giờ không hợp lệ `15:30 PM`, hoặc mã có `0` đầu không có cue (`0908`).
   * Xử lý đúng tiền tệ đa dạng (`€1.250,50` VI, `€1,250.50` EN, `50000원` & `€18.25` KO, `500元` & `1000000000000元` ZH), đơn vị (`100km/h`, `1kg`), phần trăm âm (`-12,50%`), ký tự fullwidth (`０８：００`, `３６．８℃`), và chunking `TtsEnKo` câu Hàn dài giữ tất cả chunk `<= 96` (`[95, 87, 96]`, `audio_cap_retries=[]`).

---

## 4. Kết quả chạy Speech-to-Speech (S2S) đầy đủ trên cả 6 chiều / 4 ngôn ngữ

Script thực thi: `scratch/eval_s2s_all_directions.py` (`14` ca) và `scratch/eval_extra_s2s.py` (`en_to_vi_probe`, tổng **`15` ca** mặc định `S2S_TTS_FORMAT=hybrid`, `S2S_NEURAL_TN=0`, không dùng reference text). Bổ sung **`3` ca** đối chứng `S2S_NEURAL_TN=1` trong `scratch/eval_mlx_metal.py`.

* **Toàn bộ `15/15` ca S2S mặc định:** `decoder="mt_decB"`, `fallbacks=[]`, tất cả WAV hợp lệ (`finite=true`), `0` ca bị chạm `audio_cap_retries`.

### 4.1. Nhóm `en → vi`, `ko → vi`, `zh → vi` (chiều dịch sang tiếng Việt)

| Ca test | In → Out (Elapsed) | ASR → MT | `tts_format` → `tts_text` (`oov_delta`) | Whisper back-ASR (trích yếu) |
| :--- | :--- | :--- | :--- | :--- |
| `en_to_vi_EV3775` *(giọng người, BTC)* | `5.248s` → `4.644s` (`2.090s`) | `"at530 AM"` → `"lúc 5:30 sáng."` | `time` `5:30` → `"năm giờ ba mươi phút sáng"` (`0`) | `"lúc 5 giờ 30 phút sáng"` |
| `en_to_vi_EV5301` *(giọng người, BTC)* | `3.200s` → `2.554s` (`1.841s`) | `"at 800."` → `"lên 08:00."` | `time` `08:00` → `"tám giờ"` (`0`) | `"lên 8 giờ."` |
| `en_to_vi_EV0358` *(giọng người, BTC)* | `5.720s` → `3.913s` (`1.760s`) | `"734 Main Street, Harper Valley Oregon,67297."` → giữ nguyên | `734` (`number`) + `67297` (`CRF`, `"sáu bảy hai chín bảy"`) (`0`) | `"734 Main Street, HVAA 67297"` |
| `en_to_vi_special` | `20.271s` → `15.378s` (`8.524s`) | `"330 pm"`, `"$15.50"`, `"37.5..."` → `"330 giờ tối"`, `"15 đô la. 50"`, `"37. 5 độ C"` | Đọc đúng `date`/`%`/`VN 123`/`67297`, nhưng đọc `"ba trăm ba mươi giờ"`, `"mười lăm đô la. năm mươi"`, `"ba mươi bảy. năm độ c"` (`0`) | `"330 giờ tối ... 15 đô là 50 ... 37, 5 độ sệ ... vấn 1, 2, 3 ... 6, 7, 2, 9, 7"` |
| `en_to_vi_probe` | `24.973s` → `17.370s` (`8.850s`) | `"330 PMm"`, `"V and 247"` → `"18 đô la. 25"`, `"36. 8 độ C"` | Bắt chuẩn `04782` (`CRF`), `0908246135`, `247` (`context`), đọc `"v và hai bốn bảy"` (`0`) | `"330 giờ chiều ... 18 đô là 25 ... 36, 8 độ xe ... 04782 ... 0908246135... V247"` |
| `zh_to_vi_probe` | `17.642s` → `18.698s` (`6.708s`) | ASR nhầm `18`→`48`, `VN247`→`为N247`; MT ra `21/12/2027`, `3:30`, `48 USD, 25 cents`, `36. 8 độ C`, `04782`, `0908246135`, `N247` | Đọc đúng `date`, `time`, `48 USD`, `04782`, `0908246135`, `N247` (`en hai bốn bảy`); `cents`→`"sen tơ sơ"`, `36. 8` bị tách thành `"ba mươi sáu. tám"` (`0`) | `"ngày 20 mút tháng 12 năm 2000, 027, lúc 3 giờ 30 phút ... 48 đô là 25cm xe tơ sờ ... 36. 8 độ sệ ... 04782 ... 0908246135 ... N247"` |
| `zh_to_vi_repo_e2e` | `15.256s` → `14.966s` (`5.531s`) | `"5000元"` → MT đổi sai đơn vị thành `"5.000 đô la"`; `"375，摄氏多"` → `"375 độ C"` | `CRF` bắt đúng `12345678` (`"một hai ba bốn năm sáu bảy tám"`) (`0`) | `"5.000 USD ... 375 độ ... 1, 2, 3, 4, 5, 6, 7, 8"` |
| `ko_to_vi_repo_e2e` | `17.148s` → `10.543s` (`6.058s`) | ASR Hàn rớt & gãy (`115천`, `3,75로씨`); MT mất gần hết giờ/ngày, nhầm `"9 1 2 tuổi"` | Format bình thường phần MT sinh ra (`0`) | `"cuộc họp vào ngày 14h4h ... 3,7 nam độ xe ... 9 1 2 tuổi"` |
| `ko_to_vi_probe` | `21.897s` → `13.143s` (`5.478s`) | ASR Hàn nghe `"18달 25"`, `"090 824 61 3호"`, `"브이엔니 47"`; MT ra `"18 tháng 25"`, `"Bienny 47"` (mất `04782`) | Bắt `warning` đúng `"18 tháng 25"` (`unresolved`), `context` chuẩn `090 824 61 3` & `47` (`0`) | `"09082461397"`, `"36, 8 độ"`, mất nhiều ngữ nghĩa từ ASR/MT |

### 4.2. Nhóm `vi → {en, ko, zh}` (mặc định `S2S_NEURAL_TN=0`)

| Ca test | In → Out (Elapsed) | ASR + `viBERT-CAPU` → MT | Vai trò của `tts_format` | Whisper back-ASR (trích yếu) |
| :--- | :--- | :--- | :--- | :--- |
| `vi_to_en_special` | `11.958s` → `20.294s` (`12.177s`) | CAPU chèn dấu cắt ngang số: `"Mười tháng mười một. Năm, hai nghìn, không. Hai, sáu."`, `"Gọi số không. Chín, một..."` → MT ra `"no. Two, six."`, `"Call the number. Nine..."` (rớt `0`) | `0` spans (`0.902 ms`) vì MT xuất toàn chữ | `"5.2000 No. 2.6 ... 9.12.345678"` |
| `vi_to_ko_special` | `11.958s` → `16.869s` (`8.480s`) | Cùng input ASR/CAPU; MT ảo giác `"노트북에 전화해"` và giữ `"10%"`, `"5°C"` | Chuyển đúng `10%`→`십 퍼센트`, `5°C`→`오 도` (`0.566 ms`) | `"14일 40 ... 10% ... 5도 ... 9, 1, 2, 3, 4, 5, 6, 7, 8"` |
| `vi_to_zh_special` | `11.958s` → `15.256s` (`14.079s`) | Cùng input ASR/CAPU; MT ra `"11月十一"`, `"五千元"`, `"没有"` | Chuyển đúng `11`→`十一` (`0.354 ms`) | `"11月11 ... 5000元 ... 91234567吧"` |
| `vi_to_en_probe` | `16.486s` → `23.917s` (`13.411s`) | `AsrVi` nghe mất `"ba mươi"`, `"phẩy"`, `"một ba"`→`"một bà"`; CAPU + MT ra `"two years C"`, `"a five-year-old girl"`, `"flight VN2, four, seven"` | Bắt đúng `VN2` (`context`) → `"vee en two"`, `chunks <= 92` | `"two years C ... a 5-year-old girl, flight VN2, 4-7..."` |
| `vi_to_ko_probe` | `16.486s` → `19.418s` (`11.102s`) | MT ra `"2천2천7번"`, `"18달러 2년 C"`, `"3~6~8°C"`, `"여인"` | Chuyển đúng `18달러`→`십팔 달러`, `8°C`→`팔 도`, đọc hết chuỗi số `0 9 0 8...` (`94` NFKD len `<= 96`) | `"18달러 2년씩 ... 368도 ... 0구영 82465여인, 비행기 247"` |
| `vi_to_zh_probe` | `16.486s` → `17.265s` (`14.512s`) | MT ra `"18美元"`, `"3-68摄氏度"`, `"一个女人五"`, `"越南2"` | Đọc đúng `18美元`→`十八美元`, nhưng `"3-68"` không thuộc `3-5` hay range chuẩn nên đọc `"三负六十八"` | `"18美元 ... 3付60把碩時度 ... 9082246一个女人5"` |

### 4.3. Đối chiếu bật `S2S_NEURAL_TN=1` (`scratch/mlx_metal_eval.json`)

| Ca test | Elapsed (`source_normalize_s`) | Tác động thực tế lên pipeline |
| :--- | :--- | :--- |
| `en_to_vi_special_neural` (`en→vi`) | `15.008s` (`4.09s + 0.67s + 1.79s`) | Sửa `"330 pm"` → `"3:30 PM"`, giúp NLLB dịch `"3:30 tối"`, `"15,50 đô la"`, `"37,5 độ C"` (hết lỗi tách dấu `.` của `15 đô la. 50` và `37. 5`). `tts_format` đọc đúng toàn bộ `"ba giờ ba mươi phút tối ... mười lăm đô la năm mươi xen ... ba mươi bảy phẩy năm độ c"`. |
| `vi_to_en_special_neural` (`vi→en`) | `11.825s` (`3.71s`) | Gộp đúng SĐT `"không chín một hai ba bốn năm sáu bảy tám"` → `"0912345678"`, `tts_format` đọc `"zero nine one two ..."` và Whisper nghe đúng `0912345678`. Tuy nhiên các chuỗi bị `AsrVi` đọc sai/thiếu chữ trước đó (`"hai nghìn không hai sáu"`, `"ba bảy năm độ c"`) vẫn bị NLLB dịch lỗi (`"no two six"`, `"three to seven years c"`). |
| `vi_to_en_probe_neural` (`vi→en`) | `16.470s` (`3.49s + 1.35s`) | Gây thêm lỗi ngắt câu ở `"không chín không tám. Hai bốn sáu..."`, khiến NLLB dịch `"không chín"` thành `"It's not nine"`. |

---

## 5. Phân tích nguyên nhân theo chặng & Backlog đề xuất cho agent tiếp theo

### Tổng quan đóng góp và bottleneck
* **`tts_format` (`hybrid`):** Rất nhẹ (`0.1–2.2 ms`/segment ở chế độ warm; `~12 ms` lần đầu nạp CRF), hoàn toàn không phải bottleneck hiệu năng (chi phí chính là `mt_s` + `tts_s` neural từ `1.8s` tới `14.5s`). Trên input chuẩn hoặc khi MT giữ đúng ký hiệu, chất lượng đọc trên cả 4 ngôn ngữ vượt trội rõ rệt so với `legacy`.
* **Bottleneck lớn nhất trong S2S thực tế không nằm ở TTS format:**
  1. **`vi → *` với `S2S_NEURAL_TN=0`:** `AsrVi` không viết số, sau đó `viBERT-CAPU` đặt dấu chấm/phẩy cắt ngang năm và SĐT khiến `NllbBeam` vỡ nghĩa hoàn toàn.
  2. **`NllbBeam` với chữ số từ EN/ZH/KO → VI:** Hay chèn khoảng trắng sau dấu chấm thập phân (`"36. 8"`, `"15 đô la. 50"`), làm `tts_format` hiểu nhầm thành dấu chấm hết câu và đọc tách hai số nguyên; thỉnh thoảng đổi nhầm đơn vị tiền tệ (`5000元` → `5.000 đô la`).
  3. **`latin_to_vi` (`tts.py#L103`):** Chạy **sau** `tts_format.prepare(text, 'vi')`; khi từ vựng không nằm trong glossary/Piper (`"ZIP"`→`"dip"`, `"VN"` đứng riêng→`"vơn"`, `"cents"`→`"sen tơ sơ"`), có thể phiên âm lệch hoặc sinh OOV.

### Danh sách việc ưu tiên cho agent tiếp theo

1. **[P0 - Format Engine] Sửa 3 nhóm bug regex trong `s2s_edge/s2s_edge/hybrid_textnorm.py` & `s2s_edge/s2s_edge/format_locales.py`:**
   * **Phone/ID quốc tế & mở ngoặc:** Cho phép optional `\(` đầu trong regex `identifier` (`L303`) và nhận diện chuỗi SĐT quốc tế dạng `\+\d{1,3}[- ]\d...` hoặc `\(\d{2,4}\)\s*\d...` bất kể độ dài cửa sổ cue; bổ sung `'gọi'`, `'số'`, `'电话'`, `'热线'` vào `IDENTIFIER_CUES` (`format_locales.py#L13-L19`), hoặc nâng `alphanumeric`/phone grammar để `+84-908-246-135` và `+1-800-555-0199` không rơi xuống `NUM` đọc thành số âm/dương dính chữ.
   * **Unicode `\w` & dấu phẩy `，` ở CJK:** Thay `(?<![\w-])` / `(?![\w-])` ở `unsupported` (`L295-L296`) bằng boundary ASCII `(?<![A-Za-z0-9_.\-])` / `(?![A-Za-z0-9_.\-])` để `"1e-3"` và `"3-5"` khớp được ngay cả khi kề chữ Hán/Hàn; xem xét không cho `NUM` (`L21`) nhận fullwidth `，` làm dấu thập phân (hoặc chỉ cho nhận `,`/`，` khi nhóm sau đúng 3 chữ số nếu không phải dấu thập phân chuẩn).
   * **Dấu chấm cuối câu sau `AM/PM` (`L290`):** Chỉ cho `ampm` ăn dấu chấm nội bộ (`a\.?m|p\.?m` kèm `(?!\w)` hoặc lookahead `(?:\.(?=\s|$))?`), hoặc giữ lại trailing `.` nếu ký tự ngay sau `m`/`M` là kết thúc câu.
2. **[P1 - Format & Lexicon cho ngữ cảnh MT]**
   * Xử lý biến thể viết tắt giờ không phút ở `vi` (`14h` khi đứng sau từ khóa giờ hoặc trước/sau dấu câu) và giờ thuần Hàn `(\d{1,2})시\s*(\d{1,2})분` ở `ko`.
   * Chuẩn hóa cụm `"độ C"` / `"độ F"` (rất hay gặp từ NLLB dịch sang `vi`) sang `"độ xê"` / `"độ ép"`; thêm `"ZIP"` → `"díp"` hoặc giữ đọc chữ cái `"dét i pê"` khi đi cùng `IDENTIFIER_CUES` tránh `latin_to_vi` biến thành `"dip"`.
   * Khi `"VN 247"` / mã hãng bay tách bởi 1 khoảng trắng nằm sau cue (`"chuyến bay"`, `"flight"`, `"항공편"`, `"航班"`), cân nhắc cho span `identifier` ôm cả prefix 2–4 chữ hoa `([A-Z]{2,4}\s+\d{2,})`.
3. **[P1 - Upstream S2S Pipeline (`en/vi` trước MT & hậu xử lý MT)]**
   * Thêm rule/post-edit nhẹ sửa pattern lỗi đặc trưng của `NllbBeam` trước khi vào `tts_format`, đặc biệt lỗi chèn khoảng trắng sau dấu thập phân (`r'(\d+)\.\s+(\d+)\b'` trong ngữ cảnh số đo/nhiệt độ hoặc giá tiền `đô la. 50` → `,50`).
   * Ở `vi → *` khi `S2S_NEURAL_TN=0`, `AsrVi` + `viBERT-CAPU` là nguyên nhân chính khiến S2S gãy số liệu: cân nhắc tích hợp ITN (inverse text normalization) nhẹ bằng rule cho chuỗi số/ngày/SĐT tiếng Việt từ `AsrVi` trước `viBERT-CAPU` hoặc sửa cách ngắt câu của `S2S_VI_PUNCT=model` trên chuỗi toàn số đọc chữ.

---

## 6. Tham chiếu dữ liệu thô & cách tái chạy nhanh

Tất cả artifact kiểm thử thô của phiên này (không nằm trong git repo):
* Báo cáo format 4 ngôn ngữ (probes + 17 edge cases): `/Users/khoa/.gemini/antigravity/brain/3375595f-fc60-4f1f-aff8-333160691ee7/scratch/tts_format_eval/tts_format_4langs_report.json`
* Báo cáo S2S 14 ca đầu (6 chiều): `/Users/khoa/.gemini/antigravity/brain/3375595f-fc60-4f1f-aff8-333160691ee7/scratch/s2s_eval/s2s_eval_report.json`
* OOV delta chi tiết `vi` + `en_to_vi_probe`: `/Users/khoa/.gemini/antigravity/brain/3375595f-fc60-4f1f-aff8-333160691ee7/scratch/extra_s2s_eval.json`
* ONNX ASR back-check: `/Users/khoa/.gemini/antigravity/brain/3375595f-fc60-4f1f-aff8-333160691ee7/scratch/onnx_backcheck_report.json`
* Whisper MLX back-check + `S2S_NEURAL_TN=1`: `/Users/khoa/.gemini/antigravity/brain/3375595f-fc60-4f1f-aff8-333160691ee7/scratch/mlx_metal_eval.json`
* Script đánh giá: `eval_tts_format_4langs.py`, `eval_s2s_all_directions.py`, `eval_extra_s2s.py`, `eval_onnx_backcheck.py`, `eval_mlx_metal.py` cùng nằm trong `/Users/khoa/.gemini/antigravity/brain/3375595f-fc60-4f1f-aff8-333160691ee7/scratch/`.
