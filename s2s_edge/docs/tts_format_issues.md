# TTS đọc các định dạng đặc biệt: tình hình hiện tại

Cập nhật 11/11/2026. Nguồn: so sánh câu dịch với ASR nghe lại audio đầu ra trên log 30 clip mỗi chiều, và câu chữ thật mà TTS nhận vào sau bước chuẩn hóa (`textnorm.normalize`, `names.latin_to_vi`).

TTS làm mất chrF ("heard" so với chữ dịch): en→vi −2,6 (Piper), vi→ko −2,5 (Supertonic), vi→zh −1,0 (MeloTTS), các chiều khác gần 0.

## 1. Tiếng Việt (Piper): lỗi chắc chắn, đã xác định từ câu chữ đưa vào TTS

| Định dạng | Ví dụ | TTS đang đọc | Nên đọc | Trạng thái |
|---|---|---|---|---|
| Giờ có dấu hai chấm | 3:30 | "ba : ba mươi" | "ba giờ ba mươi" | **lỗi** |
| Giờ kiểu `h` | 14h45 | "mười bốn h bốn mươi lăm" | "mười bốn giờ bốn mươi lăm" | **lỗi** |
| Ngày | 15/11/2026 | "mười lăm / mười một / hai nghìn không trăm hai mươi sáu" | "ngày mười lăm tháng mười một năm hai nghìn không trăm hai mươi sáu" | **lỗi** |
| Số điện thoại | 0912 345 678 | "chín trăm mười hai, ba trăm bốn mươi lăm…" | đọc từng chữ số "không chín một hai…" | **lỗi** |
| Mã chữ và số | VN123 | "vơn một trăm hai mươi ba" | "vê en một hai ba" | **lỗi** |
| Mã bưu chính, số dài | 67297 | "sáu mươi bảy nghìn hai trăm chín mươi bảy" | thường đọc từng chữ số | lỗi theo ngữ cảnh |
| Từ và tên tiếng Anh | Main Street, Johnson | giữ chữ Latin; Piper đọc thành "ma ai…" hoặc bỏ | phiên âm Việt | **lỗi** |
| Từ ghép có gạch nối | check-out | "che caot" | "chéc ao" | **lỗi** |
| Năm | 1990 | "một nghìn chín trăm chín mươi" | đúng | đúng |
| Tiền, phần trăm | 50.000 đồng, 15 % | "năm mươi nghìn đồng", "mười lăm phần trăm" | đúng | đúng |
| Số thập phân, số phòng | 37,5; phòng 204 | "ba mươi bảy phẩy năm", "hai trăm lẻ bốn" | đúng | đúng |

**Cần nghe để xác nhận:** vài câu ngắn có dấu hiệu sai thanh điệu: "Hãy chọn" thành "thấy chọn", "đêm qua" thành "đê quả", "con cá sấu" thành "cò cá sấu", "anh" thành "ánh". Chưa rõ lỗi do Piper đọc sai hay do ASR chấm sai.

## 2. Tiếng Hàn (Supertonic): nghi ngờ lặp âm tiết, cần nghe

ASR nghe lại ra "아무 것도**도** 없어요", "**스** 스트레스", "도움이 **이**". Có thể Supertonic đọc lắp, cũng có thể ASR nghe sai. Cách đọc số, giờ và tiền tiếng Hàn chưa kiểm.

## 3. Tiếng Trung (MeloTTS): phần lớn là lỗi của cách chấm

"她" bị nghe thành "他", "国产" thành "过产": các chữ đồng âm mà không ASR nào phân biệt được qua tiếng. Một phần mức mất 1,0 chrF là ảo. Cách đọc số dài, số điện thoại và giờ chưa kiểm.

## 4. Tiếng Anh (Supertonic): chưa thấy lỗi

Câu nghe lại gần như trùng hết. Cách đọc giờ (3:30 PM), tiền ($15.50) và số điện thoại chưa kiểm.

## 5. Câu để nghe thử

Dùng trang nghe thử (`tools/tts_web.py`) hoặc `python -m s2s_edge file`. Ghi lại cho từng câu: sai chữ nào, có lặp âm, sai thanh, hay đọc số kiểu lạ không.

| Ngôn ngữ | Câu thử |
|---|---|
| vi | `Cuộc họp lúc 3:30 chiều, phòng 204.` |
| vi | `Chuyến bay VN123 khởi hành lúc 14h45 ngày 15/11/2026.` |
| vi | `Xin gọi số 0912 345 678 để đặt bàn.` |
| vi | `Tên tôi là Patricia Johnson, tôi muốn check-out hôm nay.` |
| vi | `Hãy chọn.` / `Đó là từ đêm qua.` / `Tôi có một con cá sấu.` / `Xe của anh màu gì?` |
| ko | `아무 것도 없어요.` |
| ko | `피로와 스트레스로 인한 질병이 아니었던 것 같습니다.` |
| ko | `회의는 3시 30분에 시작합니다. 가격은 15,000원입니다.` |
| zh | `会议下午3点30分开始，票价是150元。` |
| zh | `我的电话号码是13812345678。` |
| en | `The meeting is at 3:30 PM in room 204, and the price is $15.50.` |
| en | `Call 555-0123 before November 15, 2026.` |

## 6. Kế hoạch sửa

1. **Tiếng Việt (làm được ngay, không cần nghe):** thêm quy tắc chuẩn hóa cho giờ (`3:30`, `14h45`), ngày (`dd/mm/yyyy`), số điện thoại (chuỗi bắt đầu bằng 0, có nhóm số), mã chữ-số (`VN123` đọc từng ký tự), từ ghép có gạch nối; cải thiện phiên âm tên và từ tiếng Anh. Kiểm bằng cách so câu chữ đưa vào TTS trước và sau, rồi đo "heard".
2. **Tiếng Hàn, Trung, Anh:** sửa sau khi có kết quả nghe ở mục 5.
3. **Đánh giá:** khi so TTS, chấm "heard" bằng một ASR khác ASR của pipeline, và nghe mù một nhóm nhỏ câu (xem đề xuất kiểm TTS).
