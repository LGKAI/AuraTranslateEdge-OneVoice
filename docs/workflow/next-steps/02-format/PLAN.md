# P0: sửa formatter theo loại dữ liệu

**Executor:** Antigravity. **Spec:** `../SPEC.md`. **Goal:** chặn lỗi parser đã biết, giữ nguyên giá trị và dấu câu. **Architecture:** candidate có boundary → chọn type → parse theo locale → render; không vá output theo chuỗi. **Tech:** Python grammar/CRF hiện có.

## Baseline và giới hạn

Đọc `reports/tts_format_s2s_eval_2026-11-13.md`; freeze 17 edge cases từ JSON gốc mà báo cáo trỏ tới vào fixture không lẫn training. Giữ 55 ví dụ cũ. Mỗi task thêm đối chứng mới bằng seed cố định và test cả CRF bật/tắt; không dùng test để train CRF. Chạy sau khi 01 tạo baseline/config manifest; không cần đợi hoàn tất toàn bộ benchmark để bắt đầu parser tests.

## T1 - Candidate phone/mã có cấu trúc

**Files:** `s2s_edge/s2s_edge/hybrid_textnorm.py`, `format_locales.py`; thêm `tests/test_format_boundaries.py`.
**Interface:** helper `identifier_candidates(text: str, lang: str) -> list[dict]`, mỗi dict `{start,end,raw,reason}` trỏ đúng substring; renderer/trace Span hiện có được giữ. Classifier dùng cấu trúc và ngữ cảnh, không “dài thì là phone”.

- [ ] Viết failing cases: phone có +country, ngoặc cân bằng, grouping space/dash; hai phone sau một cue; cue CJK; mã prefix tách khoảng trắng có cue.
- [ ] Expected assertions: raw bao trọn, không có NUMBER nằm bên trong ID được chọn, giữ ký tự/thứ tự/0 đầu và policy dấu; không dính từ “bốnâm”. Đối chứng `-12.5%`, `3-5 ngày`, `330 đồng`, số nhà, `12:30`, mã `AB-012` và code không có cue.
- [ ] Implement candidate và context-list boundary tổng quát; không chỉ tăng cửa sổ cue hay thêm từ khóa cho từng test. Dấu ngoặc/hyphen không cân bằng hoặc ngữ cảnh mơ hồ phải warning/unknown, không tự gán type chắc chắn.
- [ ] Run `python -m unittest discover -s tests -p test_format_boundaries.py -v`, rồi regression hiện có. Không đổi spelling policy dấu mà không ghi rõ raw/canonical mapping.

## T2 - Locale và Unicode boundary

**Files:** cùng parser; thêm `tests/test_format_locale_boundaries.py`.
**Interface:** `numeric_fields(raw, lang)` giữ API; candidate lexical boundary phải quyết định dấu câu trước khi hàm này nhận raw. Không mở rộng decoder scientific/range trong task này.

- [ ] Failing tests: nhiều số ngăn bằng `，` sát chữ Hán; scientific/range sát Hán/Hàn phải được bao trọn với unsupported warning. Kiểm span raw/start/end, không chỉ có warning.
- [ ] Đối chứng: EN `1,234.50`, VI `1.234,50`, fullwidth digits/dot; negative decimal, phone, `AB-012`, version và URL không bị bắt một phần thành entity khác.
- [ ] Implement boundary phù hợp script/locale; không dùng Unicode \w như “chỉ ASCII”, không global-replace dấu phẩy. Dạng separator mơ hồ phải có policy rõ trong trace.
- [ ] Run `python -m unittest discover -s tests -p test_format_locale_boundaries.py -v` và regression; chữ số/đơn vị của các entity có nhãn không đổi, các dạng chưa hỗ trợ không bị phân rã im lặng.

## T3 - Dấu câu cuối AM/PM

**Files:** `hybrid_textnorm.py`; test vào `test_format_boundaries.py`.

- [ ] Đọc diff nhỏ từ `origin/feat/tts-hybrid-format`; port logic/test AM/PM, không copy toàn module vì path CRF khác.
- [ ] Expected: `Meet at 3:30 PM. Next...` vẫn hai câu; giữ đúng meridiem. Test AM/PM/a.m./p.m. ở giữa/cuối câu trên cả bốn language renderers.
- [ ] Giữ punctuation ngoài time span; run targeted test rồi regression. Không sửa nội dung MT trong task này.

## T4 - Đầu vào graph và trace thực tế (P1)

**Files:** `tts_format.py`, `tts.py`; thêm `tests/test_tts_chunk_trace.py`.
**Interface:** trace phân biệt `planned_chunks`, `attempted_chunks`, `synthesized_chunks` (text, normalized/preprocessed lengths, cap/retry, span mapping). Giữ keys cũ tương thích, không gọi planned là actual.

- [ ] Inject cap retry để test log cuối chứa đúng các piece đã được synth thành công; không mất thứ tự hoặc ghi lại parent như output đã thành công.
- [ ] Test G2P thay độ dài và cùng lời đọc xuất hiện hai lần: mapping span không được bảo vệ nhầm occurrence. Nếu không map chắc chắn, warning rõ; không báo protected thành công bằng str.find mơ hồ.
- [ ] TTS graph thật chỉ trên các case bị ảnh hưởng, mỗi ngôn ngữ có phone/time/CJK thích hợp; OOV báo delta theo lượt, không lấy bộ đếm cộng dồn.

**Đạt vòng P0:** 55 ca cũ + 17 edge + bộ đối chứng mới qua expected span/value; TTS input đúng trên test có nhãn. WAV hợp lệ chỉ chứng minh runtime; nghe/ASR đối chiếu để riêng. Lưu `EXECUTION_REPORT.md`, test logs, JSON/WAV theo run ID. P1 biến thể `14h`, giờ chữ Hàn, `độ C/F`, nhãn ZIP chỉ bắt đầu sau các P0 này, không trộn thành một patch lớn.
