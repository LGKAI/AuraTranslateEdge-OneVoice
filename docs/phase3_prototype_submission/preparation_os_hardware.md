# Chuẩn bị Hệ điều hành & Phần cứng — AIBOX 8550 tại VNG Campus

**Phụ trách:** Khánh (chính), Khoa (hỗ trợ) · **Cập nhật:** 11/11/2026

**Lịch BTC:** **12/11 và 13/11/2026, 15:00–18:00** · Thiết bị: **AIBOX 8550** · Saigon AI Hub, VNG Campus, Z06 đường số 13, P. Tân Thuận, TP.HCM.

**Đồ nhóm có:** Lenovo LOQ (Windows, có USB-C, USB-A, cổng LAN) + MacBook (USB-C). **Không** có mic/loa rời, cáp HDMI, cáp LAN → dùng **mic và loa của laptop**, điều khiển hộp **qua cáp USB-C (ADB)**.

**Nguồn:** email BTC, `docs/KIMQ8550/` (4 tài liệu HSPTEK), [report_aibox_8550.pdf](file:///d:/ChuyenNganhAI/AuraTranslateEdge-OneVoice/docs/phase3_prototype_submission/report_aibox_8550.pdf), [handbook_pipeline.pdf](file:///d:/ChuyenNganhAI/AuraTranslateEdge-OneVoice/docs/phase3_prototype_submission/handbook_pipeline.pdf).

> [!IMPORTANT]
> **Quy tắc trên máy BTC (dùng chung):** không đổi mật khẩu · **không flash OS** (QDL/fastboot xóa sạch máy) · không chép đè `/usr/lib`, `/usr/lib/rfsa/adsp` · không sửa `/etc/dhcpcd.conf`. Mọi thứ của đội để trong một thư mục riêng.

---

## 1. Hiểu nhanh thiết bị

| Thành phần | Thông số | Ý nghĩa với đội |
| :--- | :--- | :--- |
| SoC / NPU | QCS8550, Hexagon **v73** | 9 graph đã link cho `QCS8550 (Proxy)`, QAIRT 2.51, đã profile trên AI Hub |
| RAM / lưu trữ | 16 GB / UFS 128 GB | Một chiều dịch ~2,2 GB RAM |
| OS | **Ubuntu 22.04** (Python 3.10, glibc 2.35) | `install.sh` tự chọn `wheelhouse_cp310/`, không cần Docker |
| **USB Type-C** | ADB shell / chép file (hoặc xuất hình qua adapter HDMI) | **Đường chính của đội** |
| Micro-USB | Serial console 115200 | Đường dự phòng khi ADB không lên |
| RJ45 | Ethernet, mặc định DHCP | Dự phòng nếu có cáp LAN (LOQ có cổng LAN) |
| 4× USB-A | Phím, chuột, ổ USB, mic/loa USB | Hộp **không có** mic, loa, jack 3.5 mm |

**Thuật ngữ:** *ADB* = shell + chép file qua USB (`adb shell`, `adb push/pull`) · *FastRPC/CDSP* = đường CPU gọi NPU (`/dev/fastrpc-cdsp`) · *Stub/Skel* = hai nửa thư viện QNN (CPU/DSP), phải **cùng bản QAIRT** · *Unsigned PD* = quyền chạy mã chưa ký trên DSP · *.bin* = graph đã biên dịch cho đúng SoC + QAIRT 2.51.

> [!WARNING]
> HSPTEK hướng dẫn cài QAIRT **2.41** vào `/usr/lib/rfsa/adsp`. Model của ta là **2.51**. Đừng trỏ `ADSP_LIBRARY_PATH` vào thư viện hệ thống; để `qai_appbuilder` tự dùng Skel 2.51 đi kèm wheel.

---

## 2. Âm thanh: dùng mic/loa laptop thế nào?

Hộp không có mic/loa, laptop không "cho mượn" mic qua USB-C được. Cách chắc nhất, không phải mua gì:

```text
 Laptop (mic)  ──ghi WAV 16 kHz──►  adb push  ──►  AIBOX 8550: ASR → MT → TTS trên NPU
 Laptop (loa)  ◄──phát WAV───────  adb pull  ◄──  out.wav + log độ trễ từng chặng
```

- Độ trễ **trên thiết bị** (ASR/MT/TTS/total, đo trong log của hộp) mới là số nộp; thời gian `adb push/pull` (vài chục ms với file vài giây) ghi riêng, nói rõ trong báo cáo.
- `adb` đi qua cáp USB, **không phải mạng**, nên vẫn chứng minh được "offline" khi tắt Wi-Fi laptop và tắt mạng trên hộp.
- Đánh giá số liệu (`run_btc.sh`) dùng file WAV/MP3 của BTC, không cần mic.

### 2.1. Script bấm-để-nói trên laptop (Hiếu + Khoa viết, chạy được cả Windows và Mac)

`tools/laptop_ptt.py` — nhấn Enter bắt đầu nói, Enter lần nữa để dịch:

```python
import subprocess, sys, time, sounddevice as sd, soundfile as sf, numpy as np

PAIR = sys.argv[1] if len(sys.argv) > 1 else "vi,ko"
BOX_DIR = "/root/onevoice"          # sửa theo thư mục thật trên hộp
# Lệnh dịch 1 file trên hộp: KHANH ĐIỀN đúng CLI của s2s_edge (đầu vào WAV, đầu ra WAV + JSON độ trễ)
BOX_CMD = f"cd {BOX_DIR}/s2s_edge_pkg && <python -m s2s_edge ... --pair {PAIR} --in /tmp/in.wav --out /tmp/out.wav --backend npu>"

def adb(*a): return subprocess.run(["adb", *a], check=True, capture_output=True, text=True).stdout

while True:
    input("[Enter] bắt đầu nói..."); buf = []
    st = sd.InputStream(samplerate=16000, channels=1, dtype="int16", callback=lambda d, *_: buf.append(d.copy()))
    st.start(); input("Đang ghi... [Enter] để dịch"); st.stop()
    sf.write("in.wav", np.concatenate(buf), 16000)
    t = time.time()
    adb("push", "in.wav", "/tmp/in.wav")
    print(adb("shell", BOX_CMD))      # in transcript, bản dịch, độ trễ đo trên hộp
    adb("pull", "/tmp/out.wav", "out.wav")
    print(f"vòng laptop↔hộp: {time.time()-t:.2f}s")
    d, sr = sf.read("out.wav"); sd.play(d, sr); sd.wait()
```

Cài: `pip install sounddevice soundfile numpy`. Thử trước ở nhà với một "hộp giả" (chạy cùng lệnh trên CPU trong WSL/Docker) để chắc phần ghi/phát chạy ổn.

> [!NOTE]
> Nếu tại VNG có sẵn mic/loa USB hoặc màn hình, dùng `python -m s2s_edge live ...` ngay trên hộp như báo cáo của Khanh. Script trên là phương án mặc định vì đội không có thiết bị rời.

---

## 3. Cần mua/mượn (tổng ~150–250k, xong tối 11/11)

| # | Món | Bắt buộc? | Ghi chú |
| :- | :--- | :- | :--- |
| 1 | **Cáp USB-C ↔ USB-C có truyền dữ liệu** (1 m) | **Bắt buộc** | Cáp sạc điện thoại thường là cáp chỉ sạc → hộp không hiện trong `adb devices`. Thử ở nhà: nối điện thoại–laptop phải chép được file. Mua 2 sợi cho chắc |
| 2 | **Cáp micro-USB ↔ USB-A có truyền dữ liệu** | Rất nên | Serial console, cắm vào cổng USB-A của LOQ. Cứu khi ADB không lên |
| 3 | **Cáp LAN Cat6 1–2 m** | Nên (~30k) | Cắm thẳng LOQ ↔ hộp; dự phòng chép file 1 Gbps và SSH |
| 4 | Ổ USB ≥ 32 GB (exFAT) | Nên | Cắm thẳng vào hộp để chép gói 8,2 GB nếu ADB chậm/lỗi |
| 5 | Ổ cắm nối dài | Nên | Cho 2 laptop |
| — | Cáp HDMI / adapter Type-C→HDMI | Không cần | Quay màn hình laptop thay cho màn hình của hộp |
| — | Mic/loa USB | Không cần | Dùng laptop (mục 2). Nếu mượn được thì mang thêm |

> [!TIP]
> Cổng Type-C của hộp chỉ có một (theo hướng dẫn HSPTEK nó vừa dùng cho ADB vừa cho xuất hình), nên không xuất hình là hợp lý: dành cổng này cho ADB.

---

## 4. Cài sẵn trên laptop (xong 21:00 ngày 11/11)

**Lenovo LOQ (máy chính, Khánh):**
| Công cụ | Cách cài / kiểm tra |
| :--- | :--- |
| Android Platform-Tools (`adb`) | Giải nén vào `C:\platform-tools`, thêm vào `PATH` → `adb version` |
| Google USB Driver | Để Windows nhận ADB (theo tài liệu HSPTEK "UPDATE OS BY FASTBOOT") |
| PuTTY | Serial: cổng `COMx` (Device Manager), 115200 |
| OpenSSH, WinSCP | `ssh -V`; WinSCP để kéo thả file |
| WSL2 Ubuntu 22.04 + Docker Desktop | Diễn tập cài offline (mục 4.1) |
| Python + `sounddevice soundfile numpy` | Chạy `laptop_ptt.py` |
| OBS Studio | Quay màn hình + âm thanh mic/loa cho video |

**MacBook (máy phụ, Khanh/Hiếu):** `brew install android-platform-tools` (`adb`) · serial: `screen /dev/tty.usbmodem* 115200` · `pip install sounddevice soundfile numpy` · quay bằng QuickTime/OBS. Mac phải cấp quyền Microphone cho Terminal (System Settings → Privacy).

### 4.1. Diễn tập trên laptop (Khánh + Khoa, 11/11)

```bash
# WSL, trong thư mục cha của s2s_edge_pkg
cd s2s_edge_pkg && find . -type f -print0 | xargs -0 sha256sum > ../SHA256SUMS && cd ..
docker run --rm -it --platform linux/arm64 --network none -v $PWD/s2s_edge_pkg:/pkg ubuntu:22.04 bash
#  (trong container — cần image đã có python3.10; nếu chưa thì cài khi còn mạng rồi chạy lại --network none)
cd /pkg && bash tools/recon.sh && bash install.sh && python -m s2s_edge --help
```
Mục tiêu: `install.sh` cài được **không cần Internet** với Python 3.10. Thiếu gói nào → báo Khanh bổ sung `wheelhouse_cp310/` ngay.

Gói `.deb` dự phòng (nếu hộp thiếu venv/alsa):
```bash
docker run --rm --platform linux/arm64 -v $PWD/debs:/debs ubuntu:22.04 bash -c \
 "apt-get update && cd /debs && apt-get download python3.10-venv python3-pip-whl python3-setuptools-whl"
```

### 4.2. Email hỏi BTC (Khánh gửi hôm nay, trả lời vào thread xác nhận lịch)
1. Hộp có bật sẵn **ADB qua Type-C** không? Tài khoản cấp có **sudo** không?
2. Image có **FastRPC/CDSP** và **Unsigned PD** không? QAIRT cài sẵn bản nào?
3. Tại chỗ có **mic/loa USB, màn hình, cáp LAN** cho mượn không?
4. Dữ liệu của đội có được **giữ từ ca 12/11 sang 13/11** không?

---

## 5. Kết nối tại VNG

```text
 Lenovo LOQ ──USB-C ↔ USB-C (data)──► Type-C   : ADB (chính)
 Lenovo LOQ ──micro-USB ↔ USB-A─────► Micro-USB: serial console (dự phòng)
 Lenovo LOQ ──cáp LAN (nếu có)──────► RJ45     : SSH/scp (dự phòng)
 Ổ USB (nếu ADB chậm) ──────────────► USB-A
 MacBook: theo dõi log / quay video / chạy song song qua ADB khi cần
```

**A. ADB (chính):**
```powershell
adb devices                          # phải ra serial của hộp
adb shell                            # đăng nhập (nếu hỏi, dùng tài khoản BTC cấp)
adb shell "df -h ~ /data"            # chọn chỗ trống ≥ 20 GB
adb push D:\s2s_edge_pkg /root/onevoice/s2s_edge_pkg
adb push D:\onevoice_btc_data /root/onevoice/onevoice_btc_data
adb push D:\SHA256SUMS /root/onevoice/
```
(Thư mục `/root/onevoice` là ví dụ; đổi theo quyền của tài khoản BTC cấp.)

**B. Serial (khi `adb devices` trống):** cắm micro-USB → Device Manager lấy `COMx` → PuTTY 115200 → Enter → đăng nhập. Từ đây xem lỗi, chạy lệnh, lấy IP.

**C. LAN thẳng LOQ ↔ hộp:** LOQ → *Network Connections* → card Wi-Fi → *Properties → Sharing* → cho phép chia sẻ với card Ethernet (hộp nhận IP `192.168.137.x` qua DHCP). Hoặc IP tạm: laptop `netsh interface ip set address "Ethernet" static 192.168.10.1 255.255.255.0`, hộp `sudo ip addr add 192.168.10.2/24 dev eth0`. Rồi `scp -r` / `ssh`.

**D. Ổ USB:** `lsblk` → `sudo mount /dev/sda1 /mnt/usb` → `cp -r` → `sync && sudo umount /mnt/usb`.

**Kiểm tra sau khi chép:** `cd /root/onevoice/s2s_edge_pkg && sha256sum -c ../SHA256SUMS --quiet` (không in gì = đúng).

---

## 6. Chạy trên hộp (theo báo cáo của Khanh)

```bash
cd /root/onevoice/s2s_edge_pkg
bash tools/recon.sh                # OS, glibc, Python, FastRPC, QNN, USB
bash install.sh                    # Ubuntu 22.04 → wheelhouse_cp310/
bash tools/session0.sh             # golden check: models_qcs8550 → models (IQ-9075) → .dlc; ghi logs/models_ok.txt
bash tools/run_btc.sh /root/onevoice/onevoice_btc_data 30   # 30 câu × 6 chiều trên NPU (0 = toàn bộ)
```

Kiểm tra nhanh song song: `cat /etc/os-release; python3 --version; free -h; ls /dev | grep -iE "rpc|dsp"`.

| Dấu hiệu | Xử lý |
| :--- | :--- |
| `adb devices` trống | Đổi cáp (cáp chỉ sạc), cài Google USB Driver, thử cổng khác, hoặc thử bằng MacBook → nếu vẫn không được: serial |
| `adb push` 8 GB quá chậm | Chép bằng ổ USB hoặc LAN; hoặc chỉ đẩy `models_qcs8550/` + code, bỏ bộ IQ-9075 |
| `install.sh` sai Python / thiếu venv | `python3 --version` phải 3.10; `dpkg -i debs/*.deb` (nếu có sudo) |
| Không có `/dev/*rpc*` | Image thiếu FastRPC → báo chuyên gia BTC; tạm `--backend cpu` để chứng minh chức năng |
| `.bin` bị từ chối | `session0.sh` tự lùi IQ-9075 → `.dlc` |
| "unsigned PD not supported" | Nhờ BTC bật, hoặc dùng Skel đã ký **bản 2.51** |
| Bị kill khi nạp | `models_fast.json` hoặc `--pair` thay `--auto` |
| Laptop ghi âm im lặng | Cấp quyền mic (Windows Privacy / macOS Privacy), chọn đúng thiết bị `sd.default.device` |

---

## 7. Kế hoạch 2 ca

**Ca 1 — 12/11, 15:00–18:00: dựng môi trường + số đo**
| Giờ | Việc | Ai |
| :--- | :--- | :--- |
| 14:30 | Có mặt ở sảnh | Cả đội |
| 15:00–15:10 | Cắm USB-C (+ micro-USB), `adb devices`, chụp `recon.sh` | Khánh |
| 15:10–15:35 | Chép gói + dữ liệu, kiểm checksum | Khoa |
| 15:35–16:00 | `install.sh` → `session0.sh` (so BURST/mặc định nếu kịp) | Khánh + Khanh |
| 16:00–17:15 | `run_btc.sh … 30` | Khanh theo dõi, Hiếu ghi độ trễ |
| 17:15–17:40 | Thử `laptop_ptt.py` với mic/loa laptop (vi↔ko, vi↔en, vi↔zh) | Hiếu, Khoa, Kin |
| 17:40–18:00 | `adb pull` toàn bộ `logs/*.tgz`, chụp ảnh setup, ghi lỗi | Khánh |

**Tối 12/11:** sửa lỗi theo log, chốt kịch bản video với Kin.

**Ca 2 — 13/11:** 15:00–15:20 kết nối lại (chép lại nếu dữ liệu bị xóa) → 15:20–16:30 `run_btc.sh … 0` hoặc bản sửa → 16:30–17:40 **quay video offline** → 17:40–18:00 kéo log + video, trả máy như lúc nhận.

---

## 8. Demo offline và quay video

```bash
# Trên hộp (qua adb shell) — adb vẫn chạy vì đi qua USB, không qua mạng
sudo nmcli networking off || sudo ip link set eth0 down
ip -br addr; ping -c 3 8.8.8.8        # quay cảnh ping thất bại
```
Trên laptop: **tắt Wi-Fi** (bật chế độ máy bay), rồi chạy `python tools/laptop_ptt.py vi,ko`.

**Bố trí quay:** OBS trên LOQ quay màn hình (cửa sổ `adb shell` + log độ trễ, cửa sổ `laptop_ptt.py`) và thu âm cả mic lẫn loa; điện thoại của Kin quay cảnh thật.

**Cảnh bắt buộc:** hộp AIBOX 8550 + cáp USB-C nối laptop → laptop bật chế độ máy bay, hộp ping thất bại → hai người nói luân phiên vào laptop, loa laptop phát bản dịch → log độ trễ từng chặng **đo trên hộp**. Ghi chú trong video/báo cáo: mic/loa là của laptop, xử lý ASR→MT→TTS 100% trên NPU của AIBOX.

---

## 9. Checklist trước khi đi (sáng 12/11)

- [ ] 2 cáp USB-C ↔ USB-C **data** (đã thử chép file), cáp micro-USB data, cáp LAN, ổ cắm nối dài
- [ ] Ổ USB exFAT: `s2s_edge_pkg` + `SHA256SUMS` + `onevoice_btc_data` + `debs/` (+ bộ `.dlc` nếu đủ chỗ)
- [ ] LOQ: `adb`, PuTTY, OBS, Python + `laptop_ptt.py` chạy được, bản sao gói trên ổ cứng, sạc
- [ ] MacBook: `adb`, quyền Microphone, sạc
- [ ] Khanh đã điền `BOX_CMD` (CLI dịch một file) trong `laptop_ptt.py`
- [ ] Diễn tập `install.sh` offline trong container ARM64 thành công
- [ ] Đã gửi email mục 4.2; mang CCCD/thẻ sinh viên

**Chia việc OS/HW:** **Khánh** — kết nối, đăng nhập, recon/install, xử lý sự cố, kéo log. **Khoa** — mua/thử cáp, chép gói + checksum, `.deb` dự phòng, cùng Hiếu làm `laptop_ptt.py`, ghi lỗi theo giờ.
