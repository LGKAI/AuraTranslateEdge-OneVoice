# OneVoice trên HSPTEK AIBOX 8550: set up ngày 2 (13/11) và những gì đã học ngày 1

Viết 12/11/2026 sau ca 1. Đọc file này là đủ để set up lại không vấp lỗi cũ. Mọi bản sửa dưới đây **đã nằm trong gói** `C:\Users\Admin\Downloads\onevoice_deploy\s2s_edge_pkg` (và đã đẩy lên hộp ở `/tmp/onevoice` ngày 1).

## 1. Sự thật về hộp (đo ngày 1, không đoán)

| Mục | Giá trị |
|---|---|
| Máy | hostname `kalama`, QCS8550 (SoC id 603), Hexagon v73, Ubuntu 22.04.2, Python 3.10.12, RAM 14 GB, đĩa `/` còn khoảng 88 GB |
| Kết nối | 1 cáp USB-C ↔ USB-C, adb serial `46ec9e7f`. Hộp còn có mạng `eth0 172.25.197.61` (cùng Wi-Fi với laptop 172.25.196.x), sshd cổng 22, và Tailscale `100.124.22.110` (của BTC, **không dùng**) |
| `adb shell` | vào bằng người dùng **`adb`** (uid 2000): HOME `/home/adb` không tồn tại, không ghi được `/home`. Chỉ **`/tmp`** ghi được (nằm trên ổ chính, không phải RAM, có thể bị dọn khi reboot) |
| root | `su - root` (mật khẩu BTC cấp, người dùng tự gõ). Nhưng root ở đây **gần như không có quyền**: `CapEff 0xc0` (chỉ setuid/setgid). Không mở được NPU, không `chown` được |
| NPU | `/dev/adsprpc-smd` thuộc **`system:system`** (660). Dịch vụ `adsprpcd`, `cdsprpcd` chạy bằng `system`. **`runuser -u system` mở được NPU** → mọi thứ chạy bằng người dùng `system` |
| QNN của hãng | bản **2.46** (`/usr/lib`, `/usr/lib/rfsa/adsp`). Gói mang **2.51** riêng trong wheel `qai_appbuilder 2.51.40`; chạy bằng `system` thì 2.51 chạy tốt, không cần đụng thư viện của hộp |
| Không có | `sudo`, mic, loa. `libatomic.so.1` thì hộp có sẵn |

## 2. Các lỗi gặp ngày 1 và đã sửa (đừng mất thời gian chẩn đoán lại)

| # | Triệu chứng | Nguyên nhân | Bản sửa (trong gói) |
|---|---|---|---|
| 1 | `laptop_push.py`: `mkdir: cannot create directory '/home/adb': Permission denied` | adb shell là người dùng `adb` | `laptop_push.py` tự dùng `/tmp/onevoice` khi HOME không ghi được |
| 2 | `Failed to create device`, `openSessionForPriority failed ... 0x200`, `unsigned PD ... not supported` | Tiến trình (kể cả root) không mở được `/dev/adsprpc-smd` | `start_box.sh` tự **chạy lại bằng `system`** (`runuser`); tự tạo `logs/` ghi được cho `system` |
| 3 | Warmup dấu câu: `backend library does not exist: .../libQnn/tmp/...libs.so`, rồi `'onnxruntime' is not a package` | `qai_appbuilder` **thay thế module `onnxruntime`** bằng wrapper QNN khi được import | `s2s_edge/__init__.py` giữ module onnxruntime thật (`_ort_real`); mô hình CPU (dấu câu, GTCRN) dùng nó |
| 4 | ASR rỗng, TTS câm; golden in `length 3.0268e-43 vs 216` | Đầu ra **số nguyên nhỏ** (độ dài, id token) trả về dạng float32 chứa **bit của số nguyên** | `backends.fix_int_output`: lấy kiểu thật từ file golden, khớp theo vị trí và kích thước (MeloTTS trả đầu ra đảo thứ tự) |
| 5 | SenseVoice (en/zh/ko) chỉ ra "Okay." / "。" với mọi clip | Bộ mã hóa w8a16: đầu vào `fbank` là **uint16 lượng tử hóa**; code đưa float vào chế độ native | `models_qcs8550/models.json` → `asr_xx_encoder.in_quant = {"fbank": [0.00015061978774610907, -38135]}` (đọc từ file `.bin`); `QnnGraph` tự lượng tử hóa |
| 6 | Golden báo PASS sai cho SenseVoice (99,1 % với đầu ra rác) | So cả bộ đệm toàn số 0 | `golden.py` so trên các vị trí khác 0 |

Sau các bản sửa: **golden PASS 8/8 trên chip**, cả 6 chiều dịch đúng (đã chạy BTC thật).

## 3. Ngày 2: các bước (khoảng 10 phút nếu `/tmp/onevoice` còn)

Laptop: PowerShell ở `C:\Users\Admin\Downloads\onevoice_deploy\s2s_edge_pkg` (adb ở `C:\platform-tools`, đã trong PATH; dùng `python` = 3.10).

1. **Cắm cáp, kiểm tra:** `adb devices` → thấy `46ec9e7f  device`.
2. **Gói còn trên hộp không?** `adb shell ls /tmp/onevoice` → phải thấy `s2s_edge_pkg onevoice_btc_data onevoice_btc_adv`.
   - Không còn (hộp reboot / bị dọn): `python tools\laptop_push.py` (khoảng 5 phút, cuối phải có `checksum (...): OK`).
   - Còn nhưng muốn chắc là bản mới nhất: chạy `laptop_push.py` lại cũng được (ghi đè, kiểm checksum).
3. **Vào hộp và mở server** (cửa sổ 1, giữ mở):
   ```
   adb shell
   su - root                      # gõ mật khẩu BTC
   cd /tmp/onevoice/s2s_edge_pkg && bash tools/start_box.sh
   ```
   Phải thấy: `re-running as user 'system'` → `03_golden_models_qcs8550 exit 0` → `warmup vi punctuation model: 0.4 s` → `web UI + API on 127.0.0.1:8770`. Lần sau trong cùng ngày: `SKIP_CHECK=1 bash tools/start_box.sh` (bỏ golden + smoke + bench, nhanh hơn khoảng 3 phút).
4. **Laptop, cửa sổ 2:** `adb forward tcp:8770 tcp:8770`, rồi mở web: `python tools\laptop_web.py` (http://127.0.0.1:8770).
5. **Chạy dữ liệu và lấy kết quả về máy** (cửa sổ 3 và 4; đặt một STAMP chung, ví dụ `20261113_1500`):
   ```
   python tools\laptop_stream.py 20261113_1500            # chép từng clip xong về Downloads\outputOnevoice\<chiều>\<BTC|ADV>\
   python tools\laptop_run_halves.py btc 20261113_1500    # BTC: 2 nửa x 460 clip, mỗi nửa xong thì tải zip về Downloads
   python tools\laptop_run_halves.py adv 20261113_1500    # ADV: 2 nửa x 300 clip
   ```
   Chạy lại cùng lệnh và cùng STAMP là chạy tiếp, bỏ qua clip đã xong.

Kiểm tra nhanh khi nghi ngờ: kết quả golden nằm ở `logs/session0_<giờ>/03_golden_models_qcs8550.log` (`grep -aE "^(PASS|FAIL)"`); đọc được qua `adb shell` không cần root. Log server: cửa sổ 1.

## 4. Số đo thật ngày 1 (trên chip, 23 clip thử + BTC đang chạy)

- Một clip BTC: tổng **khoảng 3,4–9,3 s**. ASR 0,5 s (vi) / 0,9–1,8 s (en/zh/ko); **MT 1–5 s** (beam 4, phần chậm nhất); TTS 1,2 s (vi), 1,9–3,7 s (zh), 2,3–4,5 s (en/ko). Ước tính chạy: khoảng 6 s/clip BTC (gồm ASR nghe lại), khoảng 8–10 s/clip ADV.
- Golden: asr_vi 0,5 s cho cửa sổ 15 s; SenseVoice frontend 0,17 + encoder 0,7 s; mt_enc 22 ms; mt_decB 176 ms/bước lúc warmup; nạp 9 đồ thị khoảng 18 s.
- Chất lượng (mẫu nhỏ, chưa phải số tổng): ASR tiếng Việt phần lớn câu 0 % lỗi, có câu sai từ (vd. "mang" nghe thành "xem"); vi→en/ko/zh có nghĩa; ko→vi còn sai ý ở một số câu ngắn (vd. "전화가 왔어요" → "Tôi đã đến đây"). Số tổng hợp đầy đủ: `Downloads\outputOnevoice\all_results.csv` và các zip `onevoice_btc_half*_*.zip`, `onevoice_adv_half*_*.zip`.

## 5. Còn lại cho ngày 2

1. Nếu ADV chưa chạy xong ngày 1: chạy tiếp `laptop_run_halves.py adv <STAMP cũ>` (STAMP ngày 1: `20261112_1610`).
2. Quay video demo offline: trên hộp (cửa sổ root) `nmcli networking off || ip link set eth0 down` → `ping -c 3 8.8.8.8` phải lỗi; laptop chế độ máy bay; demo qua web tab "Thử hard case" (thu âm bằng mic laptop, "Phát nối câu" bật). adb vẫn chạy qua cáp. **Bật lại mạng của hộp sau khi quay** nếu BTC yêu cầu trả máy nguyên trạng.
3. Cuối ca: kéo log (`python tools\laptop_pull.py`), hỏi BTC có để lại `/tmp/onevoice` không.

## 6. Quy tắc với máy dùng chung

Không đổi mật khẩu, không flash OS, không sửa `/usr/lib` hay cấu hình mạng vĩnh viễn, không `chmod/chown` file hệ thống. Mọi thứ của đội nằm trong `/tmp/onevoice`. Mật khẩu root do người dùng tự gõ; Claude không nhập mật khẩu.
