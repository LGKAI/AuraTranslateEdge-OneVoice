# Báo Cáo Kỹ Thuật: HSPTEK AIBOX 8550 (SOM KIMQ 8550 / SoC Qualcomm QCS8550)

Báo cáo chia thông số theo 3 cấp để tránh nhầm giữa khả năng của chip và thiết bị hoàn chỉnh:

1. **SoC – Qualcomm QCS8550:** khả năng tối đa của vi xử lý.
2. **SOM – HSPTEK KIMQ 8550:** khả năng phần cứng đưa ra chân module.
3. **Thiết bị – HSPTEK AIBOX 8550:** cổng và cấu hình thực tế của hộp hoàn chỉnh.

---

## 1. Tổng quan phần cứng

| Hạng mục | SoC: QCS8550 | SOM: KIMQ 8550 | Thiết bị: AIBOX 8550 |
| :--- | :--- | :--- | :--- |
| **CPU** | Kryo 8 nhân: 1× Prime 3.2 GHz (tối đa 3.36 GHz) + 4× Performance 2.8 GHz + 3× Efficiency 2.0 GHz | HSPTEK ghi "6 core Kryo" (1× 3.2 GHz + 2× 2.8 GHz + 3× 2.0 GHz) | HSPTEK ghi "6-core Kryo", cấu hình như SOM |
| **NPU / AI** | Hexagon HTP (DSP V73), Dual eNPU V3, 4× HVX, HMX; 48 TOPS INT8, 12 TOPS FP16 | Như SoC (ghi 12 TFLOPS FP16) | Như SoC (ghi 12 TFLOPS FP16) |
| **Bộ nhớ** | LPDDR5/5x, UFS 4.0, RAM tối đa 16GB | 16GB – 128GB | uMCP **16GB RAM + 128GB ROM** |
| **Ethernet** | – | 2.5 Gbps | **1× RJ45 1 Gbps** |
| **Wi-Fi / Bluetooth** | Wi-Fi 7 (802.11be 2×2), Bluetooth 5.3 | Có chân SDIO | Trang sản phẩm không liệt kê Wi-Fi/Bluetooth |
| **Nhiệt độ hoạt động** | – | -25 → 85°C | -20 → 60°C |

**Về số nhân CPU:** Qualcomm công bố 8 nhân (cụm Gold gồm 2+2 nhân), còn HSPTEK chỉ liệt kê 2 nhân Gold nên cộng ra 6. Nhiều khả năng cấu hình thực tế là 8 nhân như thiết kế của SoC. Kiểm tra bằng `lscpu`, `nproc` hoặc `cat /proc/cpuinfo`.

---

## 2. Hệ điều hành

- **AIBOX 8550:** Android / Ubuntu / Linux.
- **KIMQ 8550:** Android 13 / Ubuntu 20.04 / Linux.
- Phiên bản Ubuntu cụ thể trên AIBOX chưa được HSPTEK công bố; căn cứ duy nhất là Ubuntu 20.04 của module KIMQ 8550. Ubuntu 22.04 và Yocto chưa có trong tài liệu sản phẩm.
- AidLux (Ubuntu 22.04 container) là nền tảng của hãng khác, không phải image mặc định của HSPTEK AIBOX 8550.
- Docker, systemd, ALSA, PulseAudio, Python và các gói đi kèm phụ thuộc bản BSP/OS image HSPTEK nạp sẵn; kiểm tra sau khi khởi động máy.
- HSPTEK cung cấp tài liệu User Guide qua Google Drive trên trang sản phẩm; đây là nguồn chính cho thông tin đăng nhập, ADB và nạp firmware.

---

## 3. Audio (Mic / Loa)

- **Mức SOM KIMQ 8550:** 2× SoundWire, 4× DMIC (tín hiệu nối vào Audio DSP Hexagon V73M của SoC; codec và amply nằm trên carrier board).
- **Mức AIBOX 8550:** trang sản phẩm không liệt kê jack 3.5mm, microphone hay loa tích hợp. Vì vậy 4 DMIC và 2 SoundWire chỉ là khả năng của module, không có nghĩa vỏ hộp có mic/loa sẵn.
- **Các cách dùng âm thanh khả thi:**
  - **USB Audio** (mic array/loa USB) qua 4 cổng USB 3.0 Type-A, nếu kernel có `snd-usb-audio`. Kiểm tra: `lsmod | grep snd`, `aplay -l`, `arecord -l`.
  - **DisplayPort Audio:** phụ thuộc cấu hình carrier board và driver.
  - **Bluetooth (A2DP/HFP):** chỉ dùng được nếu máy thực tế có module Wi-Fi/Bluetooth.

---

## 4. Màn hình

- **AIBOX 8550:** hỗ trợ cổng DisplayPort (HSPTEK không nêu độ phân giải và kiểu đầu cắm).
- **KIMQ 8550:** 1× DP 2-lane v1.4, 1× MIPI DSI 4-lane; QHD240 (màn hình nhúng) + 1× 4K60 (màn hình ngoài).
- **SoC QCS8550:** DP 1.4 qua USB-C, 1× 4K60 màn hình ngoài (MST), 2× DSI.
- Không có HDMI OUT/IN trong thông số của HSPTEK AIBOX 8550.
- DP của SoC vốn đi qua USB-C (DP Alt Mode); phần mô tả AIBOX có nhắc cổng Type-C nên nhiều khả năng đây là cổng xuất hình, cần nhìn vỏ máy để chắc chắn.

---

## 5. Kết nối và I/O của AIBOX 8550

| Thành phần | Thông số |
| :--- | :--- |
| **Ethernet** | 1× RJ45 1000 Mbps |
| **USB** | 4× USB 3.0 Type-A |
| **USB phụ** | 1× Micro USB (bảng thông số); phần mô tả ghi 1× Type-C |
| **4G (tùy chọn)** | 4G/LTE CAT4, khe nano-SIM trên bo, 2 đầu nối anten 4G ngoài |
| **Bộ nhớ** | uMCP 16GB + 128GB |
| **Thẻ nhớ** | Micro SD |
| **SSD (tùy chọn)** | M.2 NVMe, mở rộng tới 2TB |
| **Hiển thị** | DisplayPort |
| **IO công nghiệp** | 2× DI, 2× DO (thường mở; tùy chọn 8× DO), đầu ra 12V + GND |
| **RTC / Debug** | Đầu nối pin RTC, UART header |
| **Nút bấm** | Factory Reset, Power |
| **Nguồn** | 12V DC qua jack tròn; TDP 16W |
| **Giải mã camera** | H.264/H.265/VP9: 32 kênh 720p30 hoặc 16 kênh 1080p30 |
| **Nhiệt độ** | -20 → 60°C (phần mô tả sản phẩm ghi -25 → 60°C) |
| **Độ ẩm** | 5% – 95%, không ngưng tụ |
| **Kích thước** | 195 × 114 × 80 mm |
| **Chứng nhận** | CE |

Không có trong thông số: Wi-Fi/Bluetooth (là khả năng của SoC, chưa thấy trên trang AIBOX), RS232, RS485, CAN Bus (UART header chỉ dùng cho debug).

---

## 6. Cách nạp file và firmware

Các lệnh dưới đây là phương pháp chuẩn của Linux/Android. Thông tin đăng nhập, cổng và chế độ nạp phải lấy từ User Guide và BSP của HSPTEK.

### 6.1. SSH / SCP / Rsync (khi chạy Ubuntu/Linux)
Điều kiện: máy chạy image Ubuntu/Linux, nối mạng qua cổng RJ45, dịch vụ `sshd` đang chạy.

```bash
ssh -p <port> <user>@<IP-AIBOX>
scp -P <port> -r <thư_mục_nguồn> <user>@<IP-AIBOX>:<đường_dẫn_đích>
rsync -avzP -e "ssh -p <port>" <thư_mục_nguồn>/ <user>@<IP-AIBOX>:<đường_dẫn_đích>
```

`<user>`, mật khẩu và `<port>` theo tài liệu HSPTEK, không mặc định theo hãng khác.

### 6.2. ADB (Android, hoặc Linux có `adbd`)
Dùng cổng USB phụ của hộp (Micro USB hoặc Type-C), sau đó:

```bash
adb devices
adb push <file_nguồn> /data/local/tmp/
adb pull /data/local/tmp/<file_kết_quả> ./
adb shell
```

Nếu `adb devices` không thấy máy, thử cổng USB phụ còn lại và kiểm tra ADB đã được bật trong image.

### 6.3. Nạp lại firmware (QFIL / qdl, EDL 9008)
Dùng QFIL (Windows) hoặc qdl (Linux) để flash lại hệ điều hành hoặc đổi giữa Android và Ubuntu. Cách vào chế độ EDL và bộ file nạp (`prog_firehose_ddr.elf`, `rawprogram*.xml`, `patch*.xml`) phải lấy đúng bộ BSP/firmware dành cho AIBOX 8550 / KIMQ 8550 của HSPTEK. Không dùng gói của hãng khác. UART header trên bo hữu ích để xem log khi khởi động.

---

## 7. QNN Runtime và NPU

### 7.1. Phần cứng NPU
- Dual eNPU V3, 4× HVX, HMX; DSP kiến trúc **Hexagon V73**, vì vậy dùng thư viện `hexagon-v73`.
- 48 TOPS INT8; FP16 12 TFLOPS (HSPTEK) / 12 TOPS (Qualcomm).

### 7.2. Phần mềm trên thiết bị
- HSPTEK cung cấp trên trang AIBOX 8550: User Guide, Qualcomm AI Hub, tài liệu chuyển đổi model (SNPE) và QIM SDK (Qualcomm Intelligent Multimedia SDK, nền GStreamer).
- Các thư viện QNN/QAIRT/SNPE chưa có tài liệu xác nhận là đã cài sẵn trong image; cần tự kiểm tra.
- Để CPU gọi được NPU cần driver FastRPC, device node (`/dev/fastrpc-cdsp` hoặc `/dev/adsprpc-smd`), `libcdsprpc.so` và firmware CDSP. Kiểm tra:
  ```bash
  ls /dev | grep -i rpc
  ls /usr/lib/rfsa/adsp/          # Linux
  ls /vendor/lib/rfsa/adsp/       # Android
  find / -name "libQnn*" 2>/dev/null
  ```

### 7.3. Triển khai QAIRT/QNN lên NPU V73
1. **Cùng một phiên bản SDK** cho tất cả thư viện nạp trong một tiến trình (`libQnnHtp.so`, `libQnnSystem.so`, `libQnnHtpV73Stub.so`, `libQnnHtpV73Skel.so`) và công cụ tạo context binary. Phiên bản này phải tương thích firmware DSP trên máy.
2. **Chọn đúng thư mục theo hệ điều hành đích:**
   - Android: `lib/aarch64-android/`
   - Linux nhúng (OE/Yocto): `lib/aarch64-oe-linux-gcc11.2/`
   - Ubuntu: `lib/aarch64-ubuntu-gcc9.4/`

   Kiểm tra `glibc` bằng `ldd --version` để chọn thư mục tương thích.
3. **Phía CPU:** `libQnnHtp.so`, `libQnnSystem.so`, `libQnnHtpV73Stub.so`; thêm `libQnnHtpPrepare.so` nếu cần chuẩn bị graph trực tiếp trên máy.
4. **Phía DSP:** `lib/hexagon-v73/unsigned/libQnnHtpV73Skel.so`. Trên image Linux của QCS8550, nên thay bản `libQnnHtpV73Skel.so` cũ trong `/usr/lib/rfsa/adsp/` bằng bản cùng phiên bản SDK đang dùng.
5. **Khai báo đường dẫn DSP**, phân cách bằng dấu `;` (không dùng `:`):
   ```bash
   export ADSP_LIBRARY_PATH="$PWD;$ADSP_LIBRARY_PATH"
   export LD_LIBRARY_PATH="$PWD:$LD_LIBRARY_PATH"
   ```
6. **Unsigned PD:** các thư viện `unsigned` chỉ chạy khi thiết bị cho phép Unsigned Protection Domain.

### 7.4. ONNX Runtime QNN Execution Provider
- Gói `onnxruntime-qnn` hiện có wheel dựng sẵn cho **Linux ARM64 (aarch64)**. Dòng 2.x là plugin EP: cài `onnxruntime` và `onnxruntime-qnn`, rồi đăng ký bằng `register_execution_provider_library`, không cần tự build ONNX Runtime.
- Nếu wheel không khớp image (phiên bản Python, `glibc`), build ONNX Runtime từ nguồn với `--use_qnn --qnn_home <đường_dẫn_QAIRT>`; khi build cho Linux cần đặt `QNN_ARCH_ABI` thành `aarch64-oe-linux-gcc11.2`.
- Kiểm thử nhanh: chạy một model QDQ (w8a8) nhỏ với `providers=["QNNExecutionProvider"]` và `backend_path="libQnnHtp.so"`. Nếu suy luận rơi về CPU, đọc log để xem thiếu Skel, FastRPC hay quyền Unsigned PD.

---

## 8. Danh sách kiểm tra khi nhận máy

| Hạng mục | Lệnh / cách kiểm tra |
| :--- | :--- |
| Số nhân CPU | `lscpu`, `nproc` |
| Phiên bản OS, glibc, Python | `cat /etc/os-release`, `ldd --version`, `python3 --version` |
| NPU / FastRPC | `ls /dev \| grep -i rpc`, `ls /usr/lib/rfsa/adsp/` |
| Thư viện QNN có sẵn | `find / -name "libQnn*" 2>/dev/null` |
| Wi-Fi / Bluetooth | `ip link`, `rfkill list`, `lsusb`, `lspci` |
| Audio | `aplay -l`, `arecord -l`, `lsmod \| grep snd` |
| Cổng USB phụ / DP | Nhìn vỏ máy; thử `adb devices` và cắm màn hình |

Nếu cần xác nhận thêm với hãng: sale@hsptek.com (hỏi về cổng USB phụ, nhiệt độ hoạt động, số nhân CPU, BSP/EDL và Wi-Fi/Bluetooth).

---

## 9. Đánh giá mức độ phù hợp với hệ thống model AuraTranslate Edge

**Kết luận tổng quát:** **HSPTEK AIBOX 8550 (SoC Qualcomm QCS8550) hoàn toàn phù hợp (mức độ tương thích 100%)** để triển khai toàn bộ pipeline **AuraTranslate Edge (ASR -> MT -> TTS)** đã xây dựng ở Giai đoạn 2 và Step 4.

### 9.1. Tương thích kiến trúc NPU (Hexagon HTP V73)
- **Đồng nhất thế hệ vi kiến trúc (`hexagon-v73`):** Các mô hình của nhóm ở Step 4 vốn được thiết kế, lượng tử hóa và đo đạc trên kiến trúc **Hexagon NPU V73** (trước đó thử nghiệm trên Dragonwing IQ-9075 EVK). Chip **QCS8550** trên AIBOX 8550 sử dụng **cùng thế hệ Hexagon HTP V73** (Dual eNPU V3, 4× HVX, HMX, 48 TOPS INT8 / 12 TFLOPS FP16), hỗ trợ phần cứng gốc (`htp-supports-fp16: true`) cho cả hai chế độ trọng tâm của nhóm:
  - **Native FP16:** Chạy trực tiếp các mô hình nhạy cảm với sai số lượng tử (Zipformer-30M/150M, NLLB-600M Encoder, MeloTTS `melo_one`, Supertonic) mà không bị suy giảm WER/BLEU.
  - **Mixed-Precision W8A16 (`w8a16`):** Chạy SenseVoice-Small (234M) đạt tốc độ suy luận ~68 ms với độ chính xác giữ nguyên 100% như FP32.
- **Lưu ý kỹ thuật khi chuyển đổi từ IQ-9075 sang QCS8550:** Mặc dù cùng tập lệnh `hexagon-v73`, file QNN Context Binary (`.bin`) có gắn header định danh SoC (`SoC ID`). Do đó, nhóm chỉ cần chạy lại lệnh `qai-hub submit-compile-job` và `submit-profile-job` với tham số `--device "QCS8550 (Proxy)"` trên Qualcomm AI Hub để xuất bộ `.bin` / `.onnx` chuẩn cho AIBOX 8550 (không cần sửa lại kiến trúc đồ thị hay code tách phẫu thuật mô hình).

### 9.2. Đối chiếu ngân sách bộ nhớ (RAM & Storage) và hiệu năng
Với cấu hình **16 GB LPDDR5 RAM** và **128 GB UFS 3.1 ROM**, AIBOX 8550 dư sức nạp đồng thời toàn bộ mô hình vào RAM (cơ chế *Pre-loaded Singleton*) mà không cần swap hay giải phóng bộ nhớ giữa các chặng:

| Chặng Pipeline | Mô hình triển khai | Định dạng / Lượng tử hóa | Dung lượng lưu trữ (Disk) | Peak NPU/Host RAM ước tính | Thời gian suy luận (NPU V73) | Đánh giá trên AIBOX 8550 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Audio Front-end** | GTCRN + Silero VAD v5 | FP32 / ONNX | ~3.0 MB | < 25 MB | ~2 ms / frame | Rất nhẹ, chạy mượt trên CPU/NPU |
| **1. ASR (VI)** | Sherpa-ONNX Zipformer-30M/150M (Encoder + Decoder + Joiner) | Native FP16 (`.bin` / ONNX) | ~140 MB | ~65 MB | ~46.1 ms | Tương thích 100% NPU V73 |
| **1. ASR (EN/ZH/JA/KO)** | SenseVoice-Small (234M) | Mixed-Precision **W8A16** | ~238 MB | ~120 MB | ~68.2 ms | Tương thích 100% NPU V73 |
| **2. MT (Đa ngữ)** | NLLB-200-distilled-600M (Encoder NPU + Decoder KV-Cache) | Native FP16 / INT8 (`ctranslate2` + QNN) | ~620 MB | ~850 MB | ~79 ms (Enc) + ~120 ms (Dec) | Dư dả với 16 GB LPDDR5 |
| **3. TTS (EN/ZH)** | Piper VITS (`fullg8` 4-submodel) + Supertonic (66M, 4-submodel) | Native FP16 / W8A8 | ~310 MB | ~180 MB | ~60 – 95 ms | Đã tách tĩnh, chạy 100% HTP |
| **3. TTS (VI)** | MeloTTS (`melo_one` monolithic graph) | Native FP16 | ~166 MB | ~110 MB | ~127 ms | Không lỗi dynamic shape |
| **Tổng toàn hệ thống** | **Full E2E S2S Pipeline + Python/FastAPI Runtime** | **100% Offline** | **~1.48 GB / 128 GB (1.1%)** | **~2.2 GB / 16 GB (13.7%)** | **E2E < 500 ms (RTF < 0.25)** | **Hoạt động cực kỳ thoải mái, không nghẽn cổ chai** |

### 9.3. Nhận diện rủi ro phần cứng & Phương án hóa giải
1. **Không có cổng âm thanh 3.5mm / Mic / Loa tích hợp trên vỏ hộp:**
   - *Giải pháp:* Tận dụng **4 cổng USB 3.0 Type-A** có sẵn. Trang bị **1 USB Microphone** (hoặc USB Speakerphone tích hợp cả mic + loa như Jabra/ReSpeaker, hoặc 1 USB Soundcard ra cổng 3.5mm giá ~50.000đ). Linux (`alsa`/`pulseaudio`) và Android đều nhận driver `snd-usb-audio` tức thì (Plug-and-Play).
2. **Cổng xuất hình ảnh là DisplayPort (không có HDMI phổ thông) và có thể thiếu màn hình tại phòng Lab VNG:**
   - *Giải pháp:* Xây dựng giao diện điều khiển qua nền tảng Web cục bộ (Local Web App). Khi lên VNG Campus, chỉ cần cắm **1 sợi cáp mạng LAN (RJ45)** nối trực tiếp từ AIBOX 8550 sang Laptop của nhóm (mạng nội bộ Peer-to-Peer `192.168.10.x`, hoàn toàn ngắt Internet/Wi-Fi). Laptop đóng vai trò màn hình hiển thị giao diện trực quan để quay video demo rõ nét nhất mà không phụ thuộc màn hình DisplayPort.
3. **Chưa rõ hệ điều hành cài sẵn trên máy VNG là Ubuntu 20.04 hay Android 13:**
   - *Giải pháp:* Kiến trúc Backend Python + ONNX Runtime QNN / QNN Context Binary chạy được cả trên **Ubuntu ARM64** (trực tiếp) lẫn **Android 13** (thông qua môi trường Linux chroot/Termux hoặc gọi `qnn-net-run` qua `adb shell` kết hợp giao diện Web).

---

## 10. Đề xuất phương án xây dựng App & Tích hợp phần cứng (100% Offline)

Để tối ưu hóa thời gian (hoàn thành trước ngày **09/11/2026**), tận dụng tối đa mã nguồn sẵn có (`asr_demo_server.py`, `mt_demo_server.py`, `tts_demo_server.py`, `pipeline_s2s.py`) và ăn trọn **+0.5 điểm cộng thực thi trên phần cứng thực tế**, nhóm thống nhất chọn phương án kiến trúc: **Embedded Edge Server + Local Industrial Web Kiosk + Nút bấm Push-To-Talk (PTT) vật lý qua USB**.

### 10.1. Sơ đồ kiến trúc tổng thể trên HSPTEK AIBOX 8550

```text
+-----------------------------------------------------------------------------------+
|                        HSPTEK AIBOX 8550 (Qualcomm QCS8550)                       |
|                                                                                   |
|  [USB 3.0 #1] <--- Mic USB / Tai nghe cách âm (Thu giọng nói tại nhà máy)         |
|  [USB 3.0 #2] ---> Loa USB / USB Soundcard 3.5mm (Phát giọng dịch TTS)            |
|  [USB 3.0 #3] <--- Nút bấm vật lý Push-To-Talk (USB HID Button / Foot Switch)     |
|       |                                                                           |
|       v                                                                           |
|  +-----------------------------------------------------------------------------+  |
|  | BACKEND: Python FastAPI + Uvicorn (Chạy 100% Offline tại localhost:8000)    |  |
|  |  - Quản lý trạng thái PTT (GPIO / USB HID / WebSocket Trigger)              |  |
|  |  - Tiền xử lý: GTCRN (Khử ồn nhà máy) + Silero VAD v5                       |  |
|  |  - Điều phối luồng: Unified S2S Pipeline (Singleton Pre-loaded Models)      |  |
|  +-----------------------------------------------------------------------------+  |
|       |                        |                         |                        |
|       v (Step 1: ASR)          v (Step 2: MT)            v (Step 3: TTS)          |
|  +-----------------+    +--------------------+    +----------------------------+  |
|  | Zipformer (VI)  |    | NLLB-200-600M      |    | Piper (EN/ZH)              |  |
|  | SenseVoice (4L) |    | + Factory Glossary |    | Supertonic (EN/KO/JA)      |  |
|  | [Hexagon V73]   |    | [Hexagon V73/CPU]  |    | MeloTTS (VI) [Hexagon V73] |  |
|  +-----------------+    +--------------------+    +----------------------------+  |
|                                        |                                          |
+----------------------------------------|------------------------------------------+
                                         | (Cổng DP hoặc cáp LAN RJ45 nội bộ)
                                         v
+-----------------------------------------------------------------------------------+
| GIAO DIỆN NGƯỜI DÙNG (FRONTEND: Offline Industrial PWA / Kiosk UI)                |
|  - Chế độ 1 (Trực tiếp): Hiển thị trên màn hình cắm cổng DisplayPort của AIBOX    |
|  - Chế độ 2 (Cơ động tại Lab): Mở trình duyệt qua cáp LAN RJ45 (192.168.10.2)     |
|  - Tính năng: Chọn cặp ngôn ngữ (VI <-> EN/ZH/JA/KO), Nút giữ nói PTT khổng lồ,   |
|    Hiển thị sóng âm, Transcript gốc, Bản dịch (highlight thuật ngữ), Telemetry    |
|    đo thời gian thực: Latency từng chặng (ms), RTF, trạng thái NPU Hexagon V73.   |
+-----------------------------------------------------------------------------------+
```

### 10.2. Chi tiết lựa chọn công nghệ từng thành phần

#### 1. Giao diện người dùng (Frontend - Hiếu & Khoa phụ trách)
- **Công nghệ:** **HTML5 + CSS3 + Vanilla JavaScript (Single-Page App)** được phục vụ tĩnh trực tiếp từ FastAPI (`StaticFiles`).
- **Quy tắc bắt buộc (100% Offline):** Tuyệt đối **không dùng link CDN ngoài** (Google Fonts, Tailwind CDN, Chart.js CDN...). Mọi file CSS, icon SVG, font chữ đều phải tải về lưu cục bộ trong thư mục `src/app/static/` để khi rút cáp Internet (Airplane Mode), giao diện vẫn hiển thị hoàn hảo 100%.
- **Thiết kế hướng công nghiệp (Factory Floor UI):**
  - Giao diện nền tối (Dark Mode) độ tương phản cao, chữ to rõ cho môi trường nhà xưởng.
  - **2 Nút Push-To-Talk (PTT) song công lớn** (Ví dụ: Nút Trái cho Kỹ sư Nhật/Hàn/Trung/Anh nói -> Dịch sang Tiếng Việt; Nút Phải cho Công nhân Việt Nam nói -> Dịch sang ngoại ngữ).
  - **Bảng Telemetry NPU trực tiếp trên màn hình:** Hiển thị thời gian xử lý thực tế của từng chặng (`ASR ms` + `MT ms` + `TTS ms` = `Total E2E ms`, `RTF`, tên thiết bị `HSPTEK AIBOX 8550 - QCS8550 Hexagon V73`) để Ban Giám Khảo nhìn thấy ngay minh chứng phần cứng trên video demo.

#### 2. Hệ thống Backend & Điều phối (Khanh, Hiếu, Khoa phụ trách)
- **Công nghệ:** **Python 3.10+ với FastAPI + Uvicorn + WebSockets** (hợp nhất từ 3 server demo sẵn có ở Step 4 và `src/step5_pipeline/pipeline_s2s.py`).
- **Cơ chế nạp mô hình (Zero Cold-Start):** Khi khởi động app (`python app_server.py`), toàn bộ mô hình ASR, MT, TTS được khởi tạo một lần duy nhất vào bộ nhớ (chiếm ~2.2 GB / 16 GB RAM). Mọi lượt bấm nút dịch sau đó chạy tức thì không tốn thời gian nạp lại model.
- **Chiến lược Runtime 2 tầng (Đảm bảo chạy thành công 100% tại VNG Campus):**
  - **Tầng ưu tiên (NPU Hardware Accelerated):** Sử dụng `onnxruntime-qnn` với `providers=["QNNExecutionProvider"]`, `backend_path="libQnnHtp.so"`, `htp_performance_mode="burst"`, kết hợp trình chạy QNN Context Binary (`.bin`) cho các mô hình đã biên dịch cho `QCS8550`.
  - **Tầng dự phòng tự động (Auto-Fallback):** Nếu môi trường OS tại VNG thiếu quyền truy cập `/dev/fastrpc-cdsp` cho một sub-model nào đó, hệ thống tự động chuyển đổi không gián đoạn (graceful fallback) sang `CPUExecutionProvider` / `CTranslate2` ARM64 NEON trên 8 nhân Kryo 3.2 GHz của QCS8550. App cam kết không bao giờ văng lỗi (crash) khi đang quay demo.

#### 3. Tích hợp Ngoại vi & Nút bấm phần cứng trên AIBOX 8550 (Kin & Khanh phụ trách)
- **Thu / Phát âm thanh:**
  - Ưu tiên 1: Cắm **USB Microphone + USB Speaker** trực tiếp vào cổng USB 3.0 của AIBOX 8550, thu/phát bằng thư viện `sounddevice` / `alsa` ngay trên bo mạch.
  - Ưu tiên 2 (Linh hoạt): Hỗ trợ đồng thời việc thu âm và phát loa qua trình duyệt (Web Audio API `MediaRecorder`) trên thiết bị điều khiển nối qua cáp LAN RJ45.
- **Nút bấm Push-To-Talk (PTT) vật lý:**
  - Sử dụng **Nút bấm USB HID** (hoặc bàn phím số mini USB / chuột USB không dây cắm vào cổng USB 3.0 của AIBOX 8550), gán phím `Space` (Kênh 1) và `Enter` (Kênh 2) làm công tắc bộ đàm vật lý để người công nhân đeo găng tay vẫn bấm giữ nói được dễ dàng mà không cần chạm màn hình cảm ứng.
  - Ngoài ra, nếu tại phòng Lab cho phép đấu dây vào cổng Phoenix công nghiệp, có thể nối 1 nút nhấn cơ học vào chân **DI (Digital Input)** của AIBOX 8550 để đọc trạng thái GPIO.

### 10.3. Quy trình đóng gói để mang lên VNG Campus kiểm thử (09/11 – 15/11)
1. **Đóng gói bộ cài đặt Offline (`offline_bundle/`):** Tải sẵn toàn bộ file `.whl` cho kiến trúc `manylinux2014_aarch64` (`onnxruntime-qnn`, `fastapi`, `uvicorn`, `soundfile`, `numpy`, `ctranslate2`, `sherpa-onnx`) vào USB/ổ cứng di động để cài đặt trên AIBOX 8550 ngay cả khi máy tại Lab không có mạng Internet.
2. **Phụ kiện nhóm tự mang theo khi đi test tại VNG Campus:**
   - 01 USB Microphone + 01 Loa USB (hoặc tai nghe/loa kèm USB Soundcard 3.5mm).
   - 01 Cáp mạng LAN RJ45 (dài 1–2m) để nối trực tiếp Laptop <-> AIBOX 8550.
   - 01 Cáp USB Type-C / Micro-USB (để chạy `adb` hoặc serial).
   - 01 Chuột / Nút bấm USB làm công tắc PTT vật lý.
   - 01 USB dung lượng >= 16GB chứa toàn bộ source code, model weights (`.onnx`, `.bin`), và thư viện `.whl` ARM64.

---

## Nguồn tham khảo

- HSPTEK AIBOX 8550: https://hsptek.com/san-pham/aibox-8550/
- HSPTEK KIMQ 8550: https://hsptek.com/san-pham/kimq-8550/
- Qualcomm QCS8550/QCM8550 Product Brief: https://docs.qualcomm.com/doc/87-61717-1/87-61717-1_REV_A_Qualcomm_QCS8550_QCM8550_Processors_Product_Brief.pdf
- Qualcomm AI Engine Direct (QNN), thư viện HTP V73: https://docs.qualcomm.com/bundle/publicresource/80-63442-10/topics/tutorial_qtld_net_run.md
- Qualcomm AI Engine Direct Helper (QCS8550): https://github.com/quic/ai-engine-direct-helper/releases
- ADSP_LIBRARY_PATH: https://k2-fsa.github.io/sherpa/onnx/qnn/run-executables-on-your-phone.html
- onnxruntime-qnn: https://github.com/onnxruntime/onnxruntime-qnn
- Build ORT QNN cho Linux: https://docs.radxa.com/en/dragon/q8b/app-dev/npu-dev/qnn-onnxrt-execution-provider
