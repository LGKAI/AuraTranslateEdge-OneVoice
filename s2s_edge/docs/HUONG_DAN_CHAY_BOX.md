# Hướng dẫn chạy trên HSPTEK AIBOX 8550 (bản gộp)

Bản này gộp ba phần vào nhánh `integrate/aibox8550-full`:

| Phần | Gồm những gì |
|---|---|
| ASR (nhánh `asr-improvement`, `asr-speed-accuracy`) | dấu câu / viết hoa tiếng Việt, VAD hysteresis, ASR nhiều cửa sổ, đầu RNN-T (bật khi có model) |
| TTS + đo lường (nhánh `feat/plan01-measurement`) | chuẩn hoá chữ trước TTS kiểu hybrid (giờ, số, mã, điện thoại…), runtime profile, bộ test Plan-01 |
| Phần chạy trên hộp (11–13/11) | web điều khiển trên hộp (chạy BTC/ADV theo lô, xuất zip), MT NLLB-3.3B theo route, lọc rác ASR/MT, kính ngữ tiếng Hàn, tách câu không cắt số |
| Web demo (`web_demo/`) | giao diện demo mới: mic, tải file, **sửa chữ nhận dạng rồi dịch lại**; nay chạy bằng model thật trên NPU |

Mọi thứ nằm trong thư mục gói. Không cài gì lên hệ thống, không cần mạng.

## 1. Đưa gói lên hộp (trên laptop)

```bash
python s2s_edge_pkg/tools/laptop_push.py          # copy qua cáp USB (adb) rồi kiểm checksum trên hộp
```

Gói nằm ở `/tmp/onevoice/s2s_edge_pkg` (hoặc thư mục `--dest` bạn chọn).

**Chỗ đặt gói:** user `system` (chủ của NPU) phải đọc được thư mục. **Không đặt trong `/root`.** Dùng `/tmp/onevoice` hoặc `/opt/onevoice`. Nếu đặt sai, `start.sh` sẽ báo và in sẵn lệnh để chuyển.

## 2. Chạy trên hộp: một lệnh

```bash
adb shell
su -                                  # root (mật khẩu theo tài liệu hộp)
cd /tmp/onevoice/s2s_edge_pkg
bash start.sh demo                    # web demo, cổng 8000
```

`start.sh` tự làm hết các bước sau:
1. **Tự chuyển từ root sang user `system`.** Trên image này, root không mở được `/dev/adsprpc-smd` (NPU), còn `system` thì mở được. Script tự cấp quyền ghi cho `logs/` và `.venv/` trong thư mục gói.
2. **Lần đầu:** cài offline vào `./.venv` (1–3 phút, log ở `logs_install.txt`).
3. **Lần đầu:** kiểm từng đồ thị trên NPU và chạy thử (`tools/session0.sh`, 2–5 phút). Model set đạt kiểm tra được ghi vào `logs/models_ok.txt`.
4. Nạp và chạy thử mọi model một lần (1–2 phút), rồi mở web. Khi thấy dòng `OneVoice S2S demo  ->  http://127.0.0.1:8000` là đã sẵn sàng.

Trên laptop:

```bash
python s2s_edge_pkg/tools/laptop_web.py --demo     # = adb forward tcp:8000 tcp:8000 + mở http://localhost:8000
```

Mic trên trình duyệt chỉ chạy trên `localhost` hoặc https, nên phải mở qua `adb forward` (hoặc `ssh -L 8000:localhost:8000 ...`), không mở bằng IP của hộp.

### Các chế độ khác

| Lệnh | Việc |
|---|---|
| `bash start.sh demo` | web demo (mic / tải file / sửa chữ rồi dịch lại), cổng 8000 |
| `bash start.sh box` | web điều khiển: chạy BTC/ADV theo lô, gắn thẻ lỗi, xuất zip, cổng 8770. Laptop: `python tools/laptop_web.py` |
| `bash start.sh check` | chỉ cài + kiểm NPU + chạy thử (in PASS/FAIL từng đồ thị) |
| `bash start.sh live vi ko` | hội thoại bằng mic/loa USB cắm vào hộp, ngay trong terminal |

Hai web không chạy cùng lúc được, vì mỗi cái nạp một bộ model riêng và RAM không đủ. Tắt cái này (Ctrl+C) rồi mới mở cái kia.

## 3. Dùng web demo

- Chọn chiều dịch ở trên (vi↔en, vi↔zh, vi↔ko). Bấm mic để nói, bấm lại để dịch. Hoặc giữ **Space** để nói (thả ra là dịch), **Esc** để huỷ.
- **Tải file / kéo thả** file âm thanh bất kỳ. Trình duyệt tự đổi về WAV 16 kHz mono.
- Mỗi lượt hiện thời gian từng bước: Nhận dạng → Dịch → Đọc.
- **Sửa chữ nhận dạng:** bấm nút bút chì ✎ ở lượt đó, sửa chữ, bấm **Enter** hoặc ✓. Hộp sẽ dịch và đọc lại theo chữ đã sửa (gọi `POST /api/text`); lượt đó có ghi "đã sửa tay". **Esc** để huỷ.
- Đầu trang hiện engine đang chạy, ví dụ `s2s_edge · NPU · MT 3.3B+600M`. Nếu thấy `(DEGRADED)` hoặc `MT 600M` nghĩa là 3.3B không nạp được và hộp đang dùng 600M. Lý do nằm trong log ở `logs/start_demo_*.log`.

## 4. Công tắc (biến môi trường, đặt trước `bash start.sh ...`)

| Biến | Tác dụng |
|---|---|
| `SKIP_CHECK=1` | bỏ bước kiểm NPU |
| `MODELS=models_qcs8550/models_600m.json` | chỉ dùng NLLB-600M (nhẹ hơn, nhanh hơn) |
| `S2S_MT=600m` | như trên, không nạp 3.3B |
| `S2S_MT_STRICT=1` | 3.3B lỗi thì báo lỗi luôn, không lùi về 600M |
| `S2S_ASR_CLEAN=0` | tắt lọc rác ASR/MT |
| `S2S_MT_SPLIT=0` | không tách câu ở tầng pipeline |
| `S2S_TTS_FORMAT=legacy` | tắt chuẩn hoá chữ hybrid trước TTS |
| `S2S_VAD_PROFILE=legacy` | VAD cũ, nếu thấy mã số / số điện thoại bị cắt |
| `PORT=9000`, `HOST=0.0.0.0` | đổi cổng; mở cho cả mạng LAN (mic sẽ không chạy nếu không có https) |
| `S2S_WARMUP=0` | web demo không chạy thử model trước (khởi động nhanh hơn, nhưng lượt đầu chậm) |

## 5. Lỗi thường gặp

| Thấy gì | Làm gì |
|---|---|
| `user 'system' cannot read ...` | gói đang ở thư mục riêng tư (`/root`): chạy lệnh `mv`/`chmod` mà script in ra, hoặc push lại vào `/tmp/onevoice` |
| `install failed` | xem `logs_install.txt`. Python phải là 3.10 (Ubuntu 22.04) hoặc 3.12; wheelhouse tương ứng phải có trong gói |
| `NPU device ... not accessible` | đang chạy bằng user khác root/system: dùng `su -` rồi chạy lại |
| `SOME FAILED` ở bước check | xem `logs/session0_*/03_golden_*.log`. Thường gặp: thư viện QAIRT 2.41 của hộp lấn bản 2.51. `tools/env.sh` đã xử lý, nên đừng tự `export LD_LIBRARY_PATH` |
| `MT DEGRADED` / `MT 600M` | 3.3B không nạp được (thiếu RAM hoặc lỗi NPU). Hộp vẫn dịch bằng 600M. Muốn ép dùng 3.3B: `S2S_MT_STRICT=1` để thấy lỗi thật |
| Web không mở được trên laptop | `adb devices` phải thấy hộp; chạy lại `python tools/laptop_web.py --demo`, script chờ tối đa 5 phút trong lúc hộp nạp model |
| Mic không thu | mở bằng `http://localhost:8000` (không dùng IP); cho phép micro trong trình duyệt |

## 6. Đã kiểm, chưa kiểm (13/11)

Đã kiểm trên laptop:
- Mọi test của cả hai phía qua trên code đã gộp, cả trong môi trường Python 3.10 chỉ có các gói của wheelhouse hộp. Riêng `RealAsrVadRegressionTests` cần model ONNX trong `build/` (không có trong repo).
- Mọi module import được khi không có torch / mlx / crfsuite.
- `web_demo` với engine thật chạy trên model giả, qua toàn bộ code s2s_edge: VAD → ASR → lọc rác → MT → TTS, cùng `/api/translate`, `/api/text`, `/api/health` (`web_demo/test_engine.py`).
- Giao diện (bản mock) trên trình duyệt: 6 chiều dịch, sửa chữ rồi dịch lại, bản điện thoại không bị cuộn ngang.
- Cài offline và khởi động trong container Ubuntu 22.04 arm64 không mạng: xem mục 7.

**Chưa kiểm trên hộp thật:** chất lượng và thời gian của bản gộp trên NPU, RAM khi nạp 3.3B cùng ASR/TTS, mic thật trên web demo. Lần đầu lên hộp hãy chạy `bash start.sh check`, rồi `bash start.sh demo`.

## 7. Cho người phát triển

```bash
# test (từ thư mục s2s_edge/ của repo, hoặc thư mục gói)
for t in tests/test_*.py; do python $t; done
python ../web_demo/test_engine.py           # trong repo; trong gói: python web_demo/test_engine.py
# sửa giao diện web demo không cần model
S2S_ENGINE=mock python ../web_demo/server.py
# đưa code mới từ repo vào gói đã có model (không đụng file .bin)
python tools/sync_package.py ../../onevoice_deploy/s2s_edge_pkg
```

Engine của web demo nằm trong `web_demo/engine.py` (`S2SEngine`). Nó dùng đúng đường xử lý của CLI và web trên hộp (`Engine.translate_text_tagged`), nên sửa ở pipeline thì demo cũng đổi theo.
