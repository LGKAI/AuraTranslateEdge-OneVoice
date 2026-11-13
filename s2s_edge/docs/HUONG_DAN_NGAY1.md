# Ngày 1 tại VNG: cắm hộp, chạy 2 bộ dữ liệu, lấy kết quả

Mục tiêu trong một ca (3 giờ): chạy xong bộ BTC (920 clip) và bộ nhiễu ADV (600 clip) **trên chip**, có một file zip chứa toàn bộ kết quả, biết lỗi nằm ở đâu, và thử lại được các câu khó.

**Tóm tắt 6 lệnh**

| # | Ở đâu | Lệnh | Mất bao lâu |
|---|---|---|---|
| 1 | Laptop | `adb devices` (phải thấy 1 dòng `... device`) | 1 phút |
| 2 | Laptop | `python tools\laptop_push.py` (đẩy gói và dữ liệu, tự kiểm checksum) | 5–15 phút |
| 3 | Laptop | `adb shell` | ngay |
| 4 | Hộp | `su - TÊN_TÀI_KHOẢN`, rồi `cd /tmp/onevoice/s2s_edge_pkg && bash tools/start_box.sh` (cài, kiểm tra model, mở web). **Giữ cửa sổ này mở** | 5–8 phút |
| 5 | Laptop, cửa sổ mới | `python tools\laptop_web.py` (mở web trên trình duyệt) | ngay |
| 6 | Web | Bấm **Chạy nhanh 5 câu**, xem ổn, rồi bấm **Chạy TOÀN BỘ 2 bộ**. Xong thì file zip tự tải về | 1–2 giờ |

Các lệnh trên laptop chạy trong thư mục `C:\Users\Admin\Downloads\onevoice_deploy\s2s_edge_pkg` (mở PowerShell tại đó).

---

## 1. Trước khi đi (đã làm sẵn trên LOQ của Khanh)

- Đã cài: adb (`C:\platform-tools`), Google USB Driver (`C:\usb_driver`), PuTTY (`C:\putty\putty.exe`, chỉ dùng nếu BTC cho mượn cáp micro-USB), Python 3.10 kèm numpy, soundfile, sounddevice.
- Đã có thư mục mang đi `C:\Users\Admin\Downloads\onevoice_deploy\`: gói `s2s_edge_pkg` (code, model chip, thư viện offline), `onevoice_btc_data`, `onevoice_btc_adv`, `SHA256SUMS`.
- Kiểm nhanh: `python tools\laptop_check.py --pkg .` → mọi dòng là OK, trừ "adb devices" (chưa cắm hộp).
- Mang theo: **1 cáp USB-C ↔ USB-C có truyền dữ liệu** (đây là kết nối duy nhất; thử trước với điện thoại Android: `adb devices` phải hiện thiết bị), sạc laptop, ổ cắm nối dài. Laptop cần có cổng USB-C (nhìn thân máy). Nếu có thêm được một cáp dự phòng thì nên mang: cáp là điểm hỏng duy nhất.

## 2. Nối phần cứng

```
Laptop (cổng USB-C) ──cáp USB-C ↔ USB-C──► cổng Type-C của hộp    adb, chép file, web: mọi thứ đi qua sợi cáp này
Điện ─────────────────────────────────────► nguồn của hộp         chờ khoảng 1 phút cho hộp khởi động
```
Chỉ có một sợi cáp, không có đường vào dự phòng (không console serial, không LAN, không ổ USB). Vì vậy: cắm cáp chắc chắn, đừng rút khi đang chạy, thử cổng USB-C khác trên laptop nếu `adb devices` trống.
- Không cần màn hình, bàn phím, mạng. Hộp không có mic/loa: dùng mic và loa của laptop qua web.
- Tài khoản do BTC cấp. **Không** đổi mật khẩu, **không** flash lại hệ điều hành, **không** chép đè `/usr/lib`, **không** sửa cấu hình mạng. Mọi thứ của đội nằm trong một thư mục riêng (`laptop_push.py` tự chọn `/tmp/onevoice` khi thư mục home không ghi được).

## 3. Từng bước

**Bước 1. Kết nối.** Mở PowerShell (cửa sổ mới, để có adb trong PATH):
```powershell
adb devices
```
Phải thấy `xxxxxxxx    device`. Nếu trống hoặc `unauthorized`, xem mục 6.

**Bước 2. Đẩy gói và dữ liệu lên hộp.**
```powershell
cd C:\Users\Admin\Downloads\onevoice_deploy\s2s_edge_pkg
python tools\laptop_push.py
```
Script tự đẩy gói (bỏ phần dự phòng không cần: bộ model IQ-9075 và thư viện Python 3.12) và hai bộ dữ liệu vào một thư mục riêng trên hộp (thường là `/tmp/onevoice`), rồi kiểm checksum **trên hộp**. Kết thúc phải in `checksum (...): OK`. Nếu chậm, cứ chờ: ghi lại thời gian, đừng ngắt giữa chừng (chạy lại lệnh là chép tiếp phần còn thiếu và kiểm lại checksum).

**Bước 3 và 4. Trên hộp: đăng nhập tài khoản được cấp, rồi cài, kiểm tra, mở web.**

Vào hộp bằng `adb shell` thì đang là người dùng `adb` (không phải root). Người dùng này **không dùng được NPU** (thiết bị `/dev/adsprpc-smd` thuộc người dùng `system`) và không ghi được vào `/home`. Vì vậy phải đăng nhập tài khoản BTC cấp:
```powershell
adb shell
```
```bash
su - TÊN_TÀI_KHOẢN        # root thì gõ: su -     (hộp hỏi mật khẩu: tự gõ, màn hình không hiện ký tự)
whoami                    # phải ra tên tài khoản vừa đăng nhập
cd /tmp/onevoice/s2s_edge_pkg && bash tools/start_box.sh
```
Trên hộp này, `su` cho root nhưng root **không mở được NPU** (thiết bị `/dev/adsprpc-smd` thuộc người dùng `system`). `start_box.sh` tự phát hiện và chạy lại bằng người dùng `system` (lệnh `runuser`), nên bạn vẫn gõ đúng lệnh trên.

(`/tmp/onevoice` là thư mục `laptop_push.py` đã in ở bước 2; nếu nó in thư mục khác thì dùng thư mục đó.) Không đổi mật khẩu, không dùng `sudo` để sửa hệ thống. Nếu `su` báo sai mật khẩu, hỏi lại BTC; đừng thử nhiều lần.
Script in lần lượt:
1. Thông tin hộp: hệ điều hành (phải là Ubuntu 22.04, Python 3.10), RAM, ổ đĩa, thiết bị FastRPC (phải có `fastrpc` hoặc `adsprpc`, nếu không có thì NPU không chạy được).
2. `install`: cài thư viện offline vào `.venv` trong thư mục gói (1–3 phút). Dòng cuối phải là `imports OK`.
3. Kiểm tra từng model trên NPU (`03_golden_models_qcs8550` phải là `exit 0`), rồi dịch thử 2 câu.
4. Mở web: chờ dòng `web UI + API on 127.0.0.1:8770`. **Để nguyên cửa sổ này** (đây là server; log từng clip hiện ở đây).

Nếu bước 3 báo `exit 1`, xem mục 6 trước khi chạy tiếp.

**Bước 5. Mở web trên laptop.** Mở PowerShell **mới**:
```powershell
cd C:\Users\Admin\Downloads\onevoice_deploy\s2s_edge_pkg
python tools\laptop_web.py
```
Trình duyệt tự mở `http://127.0.0.1:8770`. Đầu trang phải có nhãn xanh **NPU**, SoC id và tên bộ model. Nếu thấy nhãn đỏ thì kết quả **không** phải từ chip.

**Bước 6. Chạy 2 bộ dữ liệu.**
1. Bấm **Chạy nhanh 5 câu mỗi chiều** (khoảng 60 clip, vài phút). Xem: có số ở các cột không, nghe vài câu dịch, thời gian mỗi clip bao nhiêu. Thanh tiến độ ước tính thời gian cho cả bộ.
2. Ổn thì bấm **▶ Chạy TOÀN BỘ 2 bộ (BTC + ADV)**: 1.520 clip. Việc chạy diễn ra **trên hộp**: đóng trình duyệt hay laptop ngủ vẫn chạy tiếp, mở lại trang là thấy tiến độ. Nhưng không được rút cáp nguồn của hộp hay đóng cửa sổ `adb shell` của bước 4.
3. Chạy xong, file **`onevoice_<lần chạy>.zip`** tự tải về thư mục Downloads của laptop. Muốn tải lại hay tải khi đang chạy dở: bấm **⬇ Tải toàn bộ kết quả**.

Trong file zip có (khoảng 0,7 GB cho cả 1.520 clip; tạo file mất 1–2 phút nên trình duyệt chờ một lúc mới bắt đầu tải):
- **`in/`**: âm thanh **gốc** của từng clip (WAV 16 kHz).
- **`out/`**: âm thanh **đã dịch** do hộp tạo ra (TTS). Tên file ghép đôi với `in/` (cùng bộ, chiều, mã clip).
- **`report.html`**: mở bằng trình duyệt, xem offline. Mỗi clip một dòng có hai nút nghe (gốc và dịch), chữ ASR, chữ MT, chữ ASR nghe lại, thời gian từng chặng, vấn đề. Bảng tổng hợp ở đầu trang.
- **`results.csv`**: mỗi clip một dòng (mở bằng Excel): transcript gốc, chữ ASR và thời gian ASR, lỗi ASR, chữ MT, thời gian MT, chrF, chữ ASR nghe lại, lỗi TTS, thời gian TTS, thời gian NPU, độ dài âm thanh ra, tổng thời gian, vấn đề tự phát hiện.
- **`segments.csv`**: từng **khúc** của mỗi clip (clip dài được cắt ở chỗ im lặng): chữ ASR, chữ MT và thời gian ASR / MT / TTS / tổng của từng khúc.
- **`summary.csv`, `summary.json`**: số liệu tổng hợp theo chiều và theo mức ồn (lỗi ASR, chrF, lỗi TTS, chrF nghe lại, thời gian trung bình từng chặng, p50, p90, RTF), kèm thông tin máy.
- **`mic/`**: các lần thu âm, gửi file, thử câu chữ của phiên này (âm thanh vào, âm thanh ra, JSON).

## 4. Đọc kết quả

| Cột trên web | Ý nghĩa | Bộ nào có |
|---|---|---|
| Lỗi ASR (WER/CER %) | Chữ ASR nghe ra so với transcript gốc. WER cho vi, en; CER (theo ký tự) cho zh, ko | cả hai |
| chrF MT | Bản dịch so với bản dịch chuẩn, càng cao càng tốt | chỉ BTC |
| **Lỗi TTS %** | Cho ASR **nghe lại** âm thanh dịch rồi so với chữ MT. Cao nghĩa là TTS đọc sai hoặc đọc khó nghe | cả hai |
| chrF nghe lại | Chữ nghe lại so với bản dịch chuẩn: đúng cách BTC chấm (ASR-chrF) | chỉ BTC |
| ASR s, MT s, TTS s, Tổng s | Thời gian từng chặng đo trên hộp. p90 = 90 % clip nhanh hơn số này | cả hai |
| Âm thanh | Nút "nghe" ở cột vào và cột TTS để tự nghe | cả hai |

Bộ ADV không có bản dịch chuẩn (chỉ có transcript của từng clip), nên chỉ chấm ASR và TTS, còn bản dịch thì nghe hoặc đọc. Bảng tổng hợp của ADV tách theo 4 mức ồn để thấy ồn làm hỏng bao nhiêu.

**Vấn đề tự phát hiện** (khung bên phải; bấm vào một vấn đề để lọc đúng các clip đó):

| Vấn đề | Thường do | Cách xử lý |
|---|---|---|
| ASR sai nhiều / không nghe ra | Ồn, giọng, clip rất ngắn, nhãn của BTC sai | Nghe âm thanh vào. Nhãn sai thì bỏ qua. Lỗi do ồn thì ghi lại để báo cáo (bộ ADV cho thấy mức độ) |
| chrF MT thấp, ASR đúng | Lỗi dịch (yếu nhất ở chiều ko, zh) | Ghi lại câu. Cách sửa dài hạn: mô hình fine-tune trên Kaggle (`_keep/kaggle_nllb_ft`) |
| MT lặp từ / độ dài bất thường | Giải mã lỗi trên câu khó | Bấm "chạy lại"; nếu lặp lại ổn định thì báo lỗi kèm câu |
| **TTS đọc khác chữ MT** | Số, giờ, ngày, số điện thoại, tên riêng, từ tiếng Anh đọc sai; với tiếng Trung đôi khi chỉ là ASR nghe nhầm từ đồng âm | Nghe âm thanh ra. Thử lại câu đó ở tab 2 → "Thử câu chữ" (chọn vi→vi để chỉ đọc TTS). Lỗi đọc định dạng tiếng Việt đã có danh sách trong `docs/tts_format_issues.md` |
| TTS không ra âm thanh | Câu MT chỉ có ký tự TTS không đọc được | Thử câu đó ở "Thử câu chữ" |
| Chậm (RTF > 1) | Câu dài, NPU chưa ở chế độ nhanh, máy nóng | Xem cột MT s, TTS s. Mặc định là `BURST`. Muốn nhanh hơn: dùng bộ `models_fast.json` (bỏ beam, chất lượng giảm), xem mục 6 |
| Lỗi chạy | Model, bộ nhớ hay NPU lỗi | Xem cửa sổ `adb shell` của bước 4; tra mục 6 |

## 5. Thử lại hard case (tab 2 trên web)

- **Chạy lại** một clip: nút "chạy lại" ở cuối mỗi dòng (kết quả thay cho dòng cũ).
- **Phát nối câu** (ô tích sẵn, mặc định bật): nói một đoạn nhiều câu thì hộp dịch và đọc từng câu, laptop phát câu 1 trong lúc hộp làm câu 2. Cuối khung hiện "Âm thanh đầu tiên: x s" và "dịch xong cả đoạn sau y s". Bỏ tích để so với cách đợi dịch xong cả đoạn.
- **Thu âm**: chọn "hội thoại" (tự nhận ngôn ngữ trong cặp vi-en / vi-ko / vi-zh) hoặc "chiều cố định", bấm ● Thu âm, nói, bấm ■ Dừng và dịch. Lần đầu trình duyệt hỏi quyền mic: chọn Cho phép. Kết quả có ASR, MT, TTS, thời gian trên hộp, và ASR nghe lại.
- **Gửi file**: chọn một file WAV hoặc MP3 bất kỳ.
- **Thử câu chữ**: gõ câu, chọn chiều → chạy MT và TTS (bỏ qua ASR). Chọn cùng ngôn ngữ (vd. vi → vi) để chỉ kiểm cách TTS đọc, ví dụ `3:30`, `15/11/2026`, `0912 345 678`, `VN123`.
- Mọi lượt thu âm lưu trên hộp ở `logs/webui/mic/`.

## 6. Sự cố và cách sửa

| Dấu hiệu | Làm gì |
|---|---|
| `adb devices` trống | Thử cổng USB-C khác trên laptop (có cổng chỉ để sạc), rút cắm lại cáp, bật lại nguồn hộp. Device Manager có dấu chấm than vàng: Update driver → Browse → `C:\usb_driver`. Cắm thử cáp này vào điện thoại Android để biết cáp có dữ liệu không. Vẫn không được: **không còn đường dự phòng** (chỉ một cáp); nhờ chuyên gia BTC, họ có thể có cáp hoặc console của hộp |
| `unauthorized` | Hộp đang chờ cho phép: rút cắm lại cáp, `adb kill-server` rồi `adb devices`; nếu vẫn thế, hỏi BTC |
| `laptop_push.py` chậm hoặc đứt giữa chừng | Chạy lại lệnh đó: adb chép lại, rồi kiểm checksum. Mặc định đã bỏ phần không cần (4 GB model dự phòng và thư viện Python 3.12) |
| `checksum FAILED` | Chạy lại `laptop_push.py` (adb push ghi đè file lỗi) |
| install báo sai Python | `python3 --version` phải là 3.10 (Ubuntu 22.04) hoặc 3.12. Hộp khác phiên bản: báo BTC; dùng `tools/docker_run.sh` nếu có Docker |
| install báo lỗi khác | Xem `logs_install.txt` trong thư mục gói. Thiếu `python3-venv` và thiếu `libatomic` đã được gói tự xử lý |
| `start_box.sh` báo lỗi `Failed to create device`, `openSessionForPriority failed ... 0x200`, hoặc `unsigned PD` | Thường là chương trình không mở được `/dev/adsprpc-smd`. Script tự chạy lại bằng người dùng `system`; nếu vẫn lỗi, chạy `bash tools/npu_diag.sh 2>&1 | tee /tmp/npu_diag.txt` (chạy bằng `runuser -u system -- bash -c 'cd <thư mục gói> && bash tools/npu_diag.sh'`) và gửi kết quả |
| `mkdir: Permission denied` khi đẩy file, hoặc `Permission denied` ở `/dev/adsprpc-smd` | Đang là người dùng `adb` (không root). Đẩy file vào `/tmp/onevoice` (script tự chọn) và đăng nhập tài khoản được cấp bằng `su - TÊN_TÀI_KHOẢN` trước khi chạy `start_box.sh`. Tài khoản đó phải dùng được NPU (thường là root); nếu `start_box.sh` vẫn báo lỗi quyền, chạy `id` và `ls -l /dev/adsprpc-smd` rồi gửi kết quả |
| Không thấy thiết bị FastRPC, hoặc `03_golden` lỗi "No Snapdragon SOC", "unsigned PD", "device" | Image của hộp thiếu FastRPC hoặc chặn unsigned PD: **báo chuyên gia BTC ngay**. Thử `S2S_SYSTEM_QNN=1 bash tools/start_box.sh` (dùng thư viện QNN của hộp thay vì bản trong gói) |
| `03_golden` lỗi "incompatible", "version", "soc" | `session0.sh` đã tự thử bộ IQ-9075 nếu có trên hộp. Đẩy thêm: `python tools\laptop_push.py --all` rồi chạy lại bước 4 |
| Lỗi bộ nhớ (MEM_ALLOC, bị kill) | Dừng server (Ctrl+C) rồi `bash tools/box_serve.sh vi,ko models_qcs8550/models_fast.json` (bỏ beam). MT cũng tự hạ beam khi lỗi |
| Web không mở / "không kết nối được hộp" | Cửa sổ bước 4 có còn chạy không (dòng `web UI ... 8770`)? Chạy lại `python tools\laptop_web.py`. Mở lại trang nếu mất kết nối giữa chừng; việc chạy trên hộp không dừng |
| Muốn mở web ngay, không chờ kiểm tra | `SKIP_CHECK=1 bash tools/start_box.sh` (lần chạy thứ hai trở đi) |
| Tải zip không được | `python tools\laptop_pull.py` → kéo toàn bộ `logs/` của hộp về một file zip |
| Mic laptop ra im lặng | Windows Settings → Privacy & security → Microphone: bật cho ứng dụng desktop và trình duyệt; chọn đúng mic mặc định |

Không còn đường lùi chạy CPU (`--backend cpu`): gói chỉ có model cho NPU.

## 7. Cuối ca

1. Kiểm file zip đã nằm trong Downloads của laptop. Chép thêm một bản sang LOQ của Khánh hoặc ổ USB.
2. Phòng hờ: `python tools\laptop_pull.py` (kéo toàn bộ log của hộp về một file zip).
3. Dừng server: Ctrl+C trong cửa sổ `adb shell`, rồi `exit`.
4. Hỏi BTC có được để lại `/tmp/onevoice` cho ca 2 không (`/tmp` có thể bị dọn khi hộp khởi động lại). Nếu phải dọn: `rm -rf /tmp/onevoice`. Ca 2 đẩy lại bằng `laptop_push.py`.

Với một cáp duy nhất, laptop vừa là máy điều khiển vừa là mic và loa; cắm điện cho laptop bằng cổng sạc riêng (không dùng cổng USB-C đang nối hộp).

Quay video demo không mạng (ca 2): trên hộp `nmcli networking off || ip link set eth0 down`, rồi `ping -c 3 8.8.8.8` phải thất bại; laptop bật chế độ máy bay; adb và web vẫn chạy qua cáp USB.
