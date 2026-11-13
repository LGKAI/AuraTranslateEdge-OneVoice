# TTS format hybrid: bàn giao và tiếp tục phát triển

Branch: **`feat/tts-hybrid-format`**, base **`contest_VNG`** tại `abe8e30`.

## Phạm vi thay đổi

Đây là nâng cấp **TTS và trace**, không phải thay ASR/MT hay ghép các model exporter vào Git. Runtime hiện có trong `s2s_edge/` vẫn chạy speech → ASR → MT → TTS khi có graph phù hợp. Branch không chứa weights ONNX/DLC, Qwen, ByT5 hoặc NVIDIA T5. Không có phụ thuộc Apple Metal cho phần format này.

Luồng TTS: nhận diện span có kiểu → giữ giá trị/chuỗi ký tự → sinh lời đọc bằng tài nguyên ngôn ngữ → G2P/Unicode → chia đoạn theo ngân sách graph → audio. Grammar xử lý VI/EN/KO/ZH; CRF đề xuất thêm mã ở EN/VI. Model không sinh lại số hoặc toàn câu. Cùng grammar xử lý các giá trị khác nhau; không có bảng thay riêng chuỗi test.

- Giờ/ngày/tiền/thập phân/%/đơn vị có cấu trúc riêng.
- ZIP, điện thoại và mã đọc từng ký tự, giữ số 0 đầu. Dấu trong mã được giữ; khoảng trắng là phân nhóm.
- Giữ số nhà như số lượng khi chưa có ngữ cảnh mã, ví dụ 734.
- Supertonic tính giới hạn 96 codepoint **sau** NFKD/Jamo và language token; không cắt giữa từ. Piper kiểm 26 từ và 512 UTF-8 byte. Đoạn quá lớn được chia lại hoặc báo lỗi thay vì cắt âm thầm.
- `conversation.py` trả thêm `tts_text` và `tts_format` cho cả batch/stream; ca không có MT không tái dùng trace cũ.

## Người tiếp tục cần pull branch nào?

Repo mới:

```sh
git clone --branch feat/tts-hybrid-format --single-branch https://github.com/Khanhhh239/AuraTranslateEdge.git
cd AuraTranslateEdge
```

Repo đã có (lưu/commit thay đổi riêng trước khi switch):

```sh
git fetch origin
git switch --track origin/feat/tts-hybrid-format
```

Nếu đã có local branch này, dùng `git switch feat/tts-hybrid-format` rồi `git pull --ff-only origin feat/tts-hybrid-format`.

## Kiểm tra không cần model ASR/MT/TTS

Python 3.11 đã được dùng kiểm thử; phần grammar không cần torch/transformers/mlx.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install python-crfsuite==0.9.12
python -m unittest discover -s s2s_edge/tests -v
```

Bộ test gồm 56 ví dụ có lời đọc kỳ vọng, 1.600 trường hợp cấu trúc sinh bằng seed cố định và kiểm tra packaging/CRF/không nạp neural stack. Đây là regression phát triển, không phải độ chính xác trên mọi văn bản hay MOS.

## Chạy TTS graph thật hoặc toàn bộ S2S

Cài `s2s_edge/requirements.txt`, sau đó làm theo [runtime README](../../README.md) để đặt các graph và cấu hình registry. Các đường dẫn ONNX tương đối được tính từ **file registry JSON**, không phải từ thư mục chạy lệnh.

```sh
python -m pip install -r s2s_edge/requirements.txt
python s2s_edge/tools/tts_format_smoke.py --models /path/to/models.json --out /path/to/output --lang all
```

Smoke test dùng câu tự soạn VI/EN/KO/ZH, tạo WAV và JSON chứa text thực sự vào TTS, chunk, thời gian và hash. Chỉ cần các graph TTS tương ứng; đây **không phải** test S2S.

Để chạy audio → ASR → MT → TTS, từ thư mục `s2s_edge/`:

```sh
python -m s2s_edge file --src en --tgt vi --in /path/to/input.wav --out /path/to/output.wav --backend cpu --models /path/to/models.json
```

CPU smoke test không chứng minh parity/latency NPU. Trên máy bàn giao, `OneVoice_integrated` và `OneVoice_E2E_HybridTTS` là các thư mục ngoài checkout có đủ graph; không tồn tại tự động khi người khác clone branch này. Cần lấy graph từ quy trình dựng/cấp model của dự án và điều chỉnh registry.

## Cấu hình

- `S2S_TTS_FORMAT=hybrid`: mặc định mới.
- `S2S_TTS_FORMAT=legacy`: chuẩn hóa/chia đoạn baseline, dùng để đối chứng hoặc quay lại.
- `S2S_DATE_ORDER=DMY` hoặc `MDY`: chỉ định khi ngày dạng số mơ hồ. Nếu không có, formatter đọc bảo toàn ký tự và ghi cảnh báo.
- `S2S_IDENTIFIER_MODEL=/path/to/identifier.crfsuite`: tùy chọn ghi đè CRF. Mặc định dùng `s2s_edge/assets/normalization/identifier.crfsuite` đã kèm branch.
- `S2S_TN_AUDIT=/path/to/normalization.jsonl`: trace chuẩn hóa; tạo thư mục cha trước. Không commit log chứa lời nói của người dùng.

Nhánh này chỉ có **hybrid/legacy**. Chế độ neural và chuẩn bị nguồn `S2S_NEURAL_TN` trong workbench thử nghiệm không được đưa vào bản port, nhằm giữ phạm vi TTS và tránh phụ thuộc model lớn.

## Tái tạo CRF

`identifier_training.json` ghi hash model, hash dữ liệu, seed và phiên bản thư viện. `tests/fixtures/identifier_train.json` chứa 2.304 ví dụ **tổng hợp**; script bổ sung dữ liệu theo seed thành 15.304 câu. Không có audio/transcript bộ đề trong commit.

```sh
python s2s_edge/tools/train_identifier_tagger.py --out /path/to/retrained-crf
```

Đối chiếu hash/đầu ra trước khi thay checkpoint đã kiểm thử; khác nền tảng/thư viện có thể cho artifact khác dù seed giống nhau. Không ghi đè model đang dùng chỉ để thử huấn luyện.

## Báo cáo và bằng chứng

- [PDF báo cáo](bao-cao-format-tts-hybrid.pdf): snapshot workbench ngày 12/11/2026, gồm việc đã làm, giới hạn và ưu tiên tiếp theo.
- [Tóm tắt số đo workbench](workbench_summary.json): 49 replay TTS, 4 câu TTS, 6 ca S2S; format CPU trung bình 0,23 ms trong tập nhỏ. Không gồm ASR/MT/G2P/TTS/model load.
- `port_validation.json` ghi 14 nhóm test với 56 ví dụ (thêm ca giữ dấu chấm sau AM/PM) của checkout đã port; `smoke_results.json` ghi 4 lần gọi graph TTS thực với câu tự soạn.

PDF giữ nguyên kết quả lịch sử: các đường dẫn `OneVoice_integrated/results/...` và tùy chọn neural mô tả workbench, **không phải** các file/model được đưa hết vào branch này. Dữ liệu bộ đề và 59 WAV workbench không được công khai trong commit. Người khác có thể tái chạy regression và smoke test bằng lệnh trên; không thể suy ra chất lượng nghe chỉ từ số test qua.

## Việc nên làm tiếp theo

1. Người bản ngữ nghe xác nhận giờ, tiền, mã; tách lỗi audio TTS và ASR nhận dạng lại. Chưa có MOS; ASR đối chiếu workbench vẫn sai khác ở VI/KO/ZH.
2. Giữ số/mã xuyên ASR → MT → TTS để phát hiện lỗi trước formatter (32→2, trưa→tối). PR này không sửa các lỗi đó.
3. Mở rộng ngữ cảnh giờ/tỷ lệ/thời lượng, locale, tiền tệ và tagger KO/ZH trên tập độc lập. Kiểm bảng đọc chữ cái với người bản ngữ.
4. Đo p50/p95 cold/warm, RAM và kiểm graph trên thiết bị đích; không dùng số CPU Mac như số đo NPU.

Giới hạn hiện tại: `H:MM` mặc định là giờ; `$` quy ước USD; chưa đọc đầy đủ mọi ký hiệu khoa học/khoảng số/ordinal/phiên bản. Một số dạng được đọc ký tự kèm cảnh báo. Không cam kết đúng mọi văn bản.

Thiết kế tham khảo: [NeMo Hybrid TN](https://github.com/NVIDIA/NeMo-text-processing/blob/main/nemo_text_processing/hybrid/README.md), [ICU RBNF](https://unicode-org.github.io/icu/userguide/format_parse/numbers/rbnf.html). Triển khai là grammar/codec riêng cùng CRF có sẵn, không phải bản tích hợp đầy đủ các thư viện này.
