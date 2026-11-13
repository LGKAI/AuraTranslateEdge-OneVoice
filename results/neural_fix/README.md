# Chuẩn hóa học từ dữ liệu và thử S2S Anh - Việt

Bản triển khai ngày 12/11/2026. Thay đổi chỉ ở `OneVoice_integrated`; không ghi đè các checkout nhánh gốc. Tên riêng không nằm trong phần sửa mới.

## Đã thay đổi gì

1. **Trước MT:** Qwen3-4B-Instruct-2507 4-bit dự đoán các đoạn cần dịch và phục hồi dấu câu/định dạng. Bộ kiểm tra giữ nguyên thứ tự chữ, chữ số và các ký hiệu quan trọng; cho phép đổi dấu câu và khoảng trắng. Không dùng câu mẫu thay transcript ASR. Mỗi đoạn phải vừa cửa sổ token của NLLB.
2. **Dịch:** thêm `translate_atomic` để dịch một đoạn đã phân tách, không chạy bộ regex tách câu cũ vốn cắt tại dấu chấm bên trong số thập phân. Giữ nguyên weights NLLB và beam 4.
3. **Nhận diện mã:** CRF học BIO từ 15.304 câu huấn luyện tổng hợp, có ngữ cảnh, dạng ký tự, độ dài và các ví dụ âm tính. Tokenizer chỉ tách khoảng trắng, không dùng regex nhận diện ngày/giờ/điện thoại. Cả mã dạng ký tự và điện thoại đã được ASR đọc thành từ đều được nhận diện. Trước MT, các từ chữ số đã nhận dạng chắc theo bảng chữ số được đưa về ký tự để tránh NLLB dịch “năm” thành “years”; không bổ sung chữ số bị mất.
4. **Trước TTS VI:** ByT5-Vi-Normalization đã fine-tune tạo cách đọc ngày, giờ, số thập phân, tiền và phần trăm. Phát âm các ký hiệu chữ Latin còn lại bằng bảng tên chữ cái.
5. **Trước TTS EN:** model dự đoán các span cần đọc, sau đó decoder NVIDIA T5 đã fine-tune chuyển từng span thành lời đọc. Dùng tokenizer Moses theo đầu vào huấn luyện của decoder, ngữ cảnh trái/phải 3 từ. Không dùng covering grammar/WFST và không dùng hàm đọc số thủ công cũ trong chế độ mới.

**Phần cố định cần nói rõ:** chính sách đọc mã từng ký tự dùng bảng tên chữ số/chữ cái. Đây là ánh xạ phát âm dùng chung cho mọi giá trị mã, không phải model sinh cách đọc. Ba nhóm ngày giờ, thập phân và tiền do model xử lý; không thêm `if/else` hoặc regex chuyển đổi riêng từng định dạng. Dữ liệu tổng hợp có các mẫu huấn luyện; điều này không đồng nghĩa với quy tắc chạy trực tiếp khi suy luận.

## Cách bật và chạy lại

Chế độ cũ vẫn là mặc định. Bật chế độ mới bằng biến môi trường:

```sh
cd /Users/khoa/study/Onevoice_final_ves/OneVoice_integrated
S2S_NEURAL_TN=1 \
S2S_RESULT_DIR="$PWD/results/neural_fix/final" \
HF_HOME=/private/tmp/s2s-hf \
/private/tmp/s2s-original-env/bin/python tools/run_e2e.py en vi
```

Đổi `en vi` thành `vi en` cho chiều còn lại. Có thể thêm đường dẫn WAV làm đối số thứ ba. Chạy trên Mac với GPU Metal cho model chuẩn hóa; các graph ASR/NLLB/TTS vẫn là ONNX CPU. Chưa export/deploy phần chuẩn hóa mới lên Hexagon NPU, chưa tối ưu latency hay bộ nhớ cho edge.

Các script tái tạo: `tools/train_identifier_tagger.py`, `tools/download_normalizer.py`, `tools/download_vi_normalizer.py`, `tools/prepare_nemo_decoder.py`. `implementation.patch` ghi diff với source contest_VNG; `code_sha256.json` và `requirements.resolved.txt` ghi phiên bản chạy. Bản decoder NVIDIA được đổi tên khóa weights, nạp strict và lưu sang Hugging Face; không train lại decoder.

## Kiểm tra

- `../neural_normalizer_evaluation.json`: 20 ca hồi quy mức văn bản, kiểm tra các cụm đọc yêu cầu và không còn chữ số. Bộ này được dùng trong phát triển, không phải tập test độc lập và không chứng minh đúng mọi văn bản.
- Cấu hình cuối qua 20/20 phép kiểm tra đó. Kiểm tra theo cụm từ chỉ là kiểm tra một phần; chưa chấm MOS hay đánh giá toàn bộ ngữ nghĩa tự động.
- Giá trị ví dụ hướng dẫn model phân đoạn/span khác giá trị của các ca kiểm tra. Không đưa transcript tham chiếu hoặc đầu ra mong muốn vào ASR/MT khi chạy S2S.
- Hai test S2S cuối dùng lại đúng hai WAV tổng hợp của lần chạy trước để đối chiếu. Văn bản `written` là nội dung tham chiếu của fixture; không phải kết quả ASR và không được truyền vào pipeline.
- `final/{en_to_vi,vi_to_en}.json`: ASR thô, sau chuẩn hóa, bản dịch, câu thực tế vào TTS, timing, trace và đường dẫn audio. Các file JSONL là audit ghi từng bước ngay khi chạy.
- Các log/thư mục thử khác là thử nghiệm trong phát triển, không phải bản audio bàn giao. Bản bàn giao nằm trong `final/`.

## Giới hạn còn lại

- ASR VI của fixture cũ nhận sai/mất từ số: “năm mươi nghìn” thành “năm nghìn”, “mười lăm phần trăm” thành “mười phần trăm”. Model chuẩn hóa không lấy đáp án mẫu để bù lại.
- NLLB vẫn có thể dịch sai ngữ nghĩa hoặc đơn vị; ví dụ “3:30 PM” thành “3:30 tối”. Khôi phục dấu hai chấm không tự sửa hết lỗi dịch về thời điểm trong ngày.
- ByT5 có thể đọc tiền kiểu thập phân, ví dụ “15,50 đô la” thành “mười lăm phẩy năm mươi đô la”, thay vì tách đô la/xu. Giá trị được giữ trong ví dụ này nhưng cách đọc chưa đồng nhất.
- Nhận diện mã là mô hình thống kê; không có bảo đảm bao phủ mọi mã, ký hiệu hoặc locale. Chưa đánh giá đầy đủ điện thoại quốc tế, định dạng mơ hồ và dữ liệu ngoài miền.
- Audio hoàn tất và có tín hiệu không đồng nghĩa mọi phát âm đều đúng. Cần người nghe xác nhận và tập test độc lập trước khi coi là bản dùng thực tế.

## Nguồn model

- Qwen3-4B-Instruct-2507: https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507 ; bản MLX: https://huggingface.co/mlx-community/Qwen3-4B-Instruct-2507-4bit . Revision đã lưu trong `models/source/qwen3-4b-instruct-mlx4/provenance.json`.
- ByT5-Vi-Normalization: https://huggingface.co/nmcuong/ByT5-Vi-Normalization . Model card ghi được fine-tune trên 73.000 cặp do Gemini sinh, có thể có nhiễu. Revision trong `models/source/byt5-vi-normalization/provenance.json`.
- NVIDIA Neural Text Normalization T5, v1.5.0: https://catalog.ngc.nvidia.com/orgs/nvidia/nemo/models/neural_text_normalization_t5 . Chỉ dùng decoder, không dùng Albert tagger hoặc các quy tắc hậu xử lý của NeMo. Hash archive và cách ánh xạ weights ở `models/source/nemo-en-tn/hf/provenance.json`.
