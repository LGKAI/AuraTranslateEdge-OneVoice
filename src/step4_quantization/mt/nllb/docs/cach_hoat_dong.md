# Giải thích dễ hiểu

## Vấn đề: dịch máy là vòng lặp

NLLB dịch bằng cách sinh từng token: sinh token 1, đưa vào để sinh token 2, ... cho tới token kết thúc.
Nếu mỗi vòng là một lần gọi chip qua AI Hub thì một câu 15 token cần 15 lần gọi, mỗi lần vài chục phút. Không dùng được.

## Cách giải: trải vòng lặp vào trong graph

Thay vì lặp bên ngoài, ta **chép các bước giải mã nối tiếp nhau trong cùng một graph** (số bước T cố định).
Graph nhận câu nguồn và trả về T token một lần. Bộ nhớ của bước trước (KV-cache) truyền nội bộ giữa các bước.
Khi dịch xong sớm, các token thừa sau token kết thúc (EOS) bị bỏ ở máy chủ.

```
src_ids [32 token] ─► Encoder ─► bước 1 ─► bước 2 ─► ... ─► bước T ─► out_ids [T token]
                                  (token bước 1 là mã ngôn ngữ đích, ép sẵn)
```

## Giới hạn của cách này

- Số bước T cố định. Mỗi bước thêm khoảng 400 layer vào graph; graph quá lớn thì AI Hub báo quá thời gian
  (đã thử: T=16 chạy được, T=24 không).
- Câu nguồn tối đa 32 token, bản dịch tối đa T−1 token. Câu dài hơn phải chia câu, hoặc nối hai lần gọi (chưa làm).

## Bảng từ vựng

NLLB có 256.206 token cho 200 ngôn ngữ. Phần lớn không cần cho tiếng Anh, Việt, Hàn, Trung.

| Cách | Số token | Kết quả so với HF `generate` (48 câu FLEURS) |
|---|---|---|
| Chỉ giữ token xuất hiện trong văn bản FLEURS + token ngắn | 42.345 | **31/48** giống — hỏng (thiếu nhiều từ Anh/Việt) |
| Giữ **mọi** token của bốn chữ viết (Latin/Việt/Hàn/Hán) | **138.974** | **48/48** giống |

Vì vậy dùng phương án 139k. Ngoài việc nhỏ hơn, việc cắt giúp file ONNX dưới 2 GB (AI Hub không nhận file lớn hơn)
và làm phép nhân cuối (chọn token trên cả bảng từ) nhỏ hơn.
Khi tách token đầu vào, mảnh nào ngoài tập giữ lại sẽ được cắt tiếp thành mảnh nhỏ hơn có trong tập.

## Kiểm tra đúng sai

Mọi graph đều được so với bản CPU (HF `generate`, tham lam, cùng đầu vào). Cả bản T=2, 8, 16 trên chip cho **đúng từng token**
trên 8 câu. Chưa có BLEU trên tập lớn nên chưa kết luận chất lượng dịch tổng thể.

## Bài học khi deploy

- Chạy thật trên chip rồi so với CPU; đừng chỉ tin mô phỏng. (Xem nhánh `supertonic`: một phép `ReduceSum` sai trên chip làm hỏng cả graph.)
- AI Hub đặt tên đầu ra `output_N`; sắp theo số N, đừng tin thứ tự dict.
- Có giới hạn thời gian thực thi; graph lớn (~11000 layer) bị loại. Thử tăng dần kích thước để tìm mức tối đa.
