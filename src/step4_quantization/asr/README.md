# Module ASR — Nhận dạng Giọng nói Tự động trên Qualcomm Hexagon NPU

Thư mục này quản lý toàn bộ mã nguồn lượng tử hóa, biên dịch NPU, đo kiểm phần cứng và ứng dụng Web Dashboard thời gian thực cho 4 ngôn ngữ trên nền tảng **Qualcomm Dragonwing IQ-9075 EVK (Hexagon NPU v73, 100 dense TOPS)**:

| Ngôn ngữ | Mô hình | Độ chính xác Lượng tử | NPU Offload | Độ trễ NPU (Profile thật) |
| :--- | :--- | :---: | :---: | :---: |
| 🇻🇳 **Tiếng Việt** | **Zipformer-150M-CR-CTC** | INT16 (W16A16) | **100.0%** (0 CPU fallback) | **126.78 ms** / 15s audio |
| 🇺🇸 **English** | **SenseVoice-Small** (2-stage) | W8A16 + FP16 FE | **100.0%** (0 CPU fallback) | **269.40 ms** / 29s audio |
| 🇨🇳 **中文** | **SenseVoice-Small** (2-stage) | W8A16 + FP16 FE | **100.0%** (0 CPU fallback) | **269.40 ms** / 29s audio |
| 🇰🇷 **한국어** | **SenseVoice-Small** (2-stage) | W8A16 + FP16 FE | **100.0%** (0 CPU fallback) | **269.40 ms** / 29s audio |

---

## 1. Cấu trúc Thư mục

```text
src/step4_quantization/asr/
├── README.md                      # Hướng dẫn module ASR & cách chạy Dashboard (Tài liệu này)
├── asr_demo_server.py             # FastAPI Web Dashboard tích hợp cả 4 ngôn ngữ, hỗ trợ batch & Cloudflare Tunnel
├── sensevoice/                    # Chuyên sâu SenseVoice-Small (En/Zh/Ko)
│   ├── README.md                  # Báo cáo kỹ thuật chi tiết toàn diện & bảng tra cứu Job ID
│   ├── export/                    # Xuất đồ thị ONNX (Frontend v3 + Encoder W8A16)
│   ├── hub/                       # Script submit compile, profile, inference lên Qualcomm AI Hub
│   ├── eval/                      # Bộ đánh giá 90 câu chuẩn FLEURS
│   ├── diagnostics/               # Phân tích dải số học FP16 & chẩn đoán đồ thị
│   └── results/                   # Kết quả đo kiểm JSON và CSV từ chip NPU thật
└── zipformer/                     # Chuyên sâu Zipformer-150M-CR-CTC (Vi)
    ├── README.md                  # Báo cáo kỹ thuật toàn diện & kết quả đo kiểm trên chip silicon
    ├── deploy_zipformer_hub.py    # Script compile + profile + inference encoder trên AI Hub
    ├── submit_zipformer_to_aihub.py # Script submit full pipeline lên AI Hub & tính WER
    ├── step4_hardware/            # Bộ công cụ phẫu thuật ONNX, calibration & benchmark
    ├── vendored_zipformer/        # Định nghĩa kiến trúc lõi Zipformer
    └── results/                   # Kết quả đo kiểm chip thật từ Qualcomm AI Hub
```

---

## 2. Hướng dẫn Khởi chạy ASR Dashboard

Tệp [`asr_demo_server.py`](asr_demo_server.py) là máy chủ tích hợp phục vụ giao diện thu âm thời gian thực, tự động định tuyến ngôn ngữ và gửi tensor lên Qualcomm AI Hub để chạy trên chip NPU thật.

### Cách 1: Chạy Local (Nội bộ trên máy cá nhân)

Sử dụng khi bạn tự kiểm thử trực tiếp trên máy tính phát triển:

```powershell
python src/step4_quantization/asr/asr_demo_server.py
```

* Mở trình duyệt bất kỳ (Chrome, Edge, Firefox) và truy cập: **`http://localhost:8420`**
* Có thể tùy biến cổng hoặc host nếu cần:
  ```powershell
  python src/step4_quantization/asr/asr_demo_server.py --port 9000 --host 127.0.0.1
  ```

---

### Cách 2: Chạy Public ra ngoài Internet bằng Cloudflare Tunnel (Có HTTPS)

> [!IMPORTANT]
> **Tại sao bắt buộc phải dùng HTTPS khi chia sẻ ra ngoài mạng?**  
> Chính sách bảo mật của các trình duyệt web (Chrome, Edge, Safari trên cả iOS và Android) **chỉ cho phép ứng dụng truy cập Microphone khi kết nối thông qua `localhost` hoặc giao thức an toàn `HTTPS`**. Nếu bạn chia sẻ qua địa chỉ IP mạng cục bộ thường (`http://192.168.x.x:8420`), trình duyệt trên điện thoại/máy người khác sẽ lập tức **chặn quyền truy cập micro**.  
> Giải pháp tối ưu nhất là sử dụng **Cloudflare Tunnel** — hoàn toàn miễn phí, tự động cấp chứng chỉ SSL/HTTPS hợp lệ và không cần mở cổng modem (NAT/Port Forwarding).

#### Tùy chọn A: Tự động hoàn toàn (Khuyên dùng)
Chỉ cần thêm cờ `--share` vào lệnh khởi động:

```powershell
python src/step4_quantization/asr/asr_demo_server.py --share
```

* Script sẽ tự động phát hiện `cloudflared.exe` (hoặc tự động tải về nếu chưa có), khởi tạo đường hầm an toàn và in ra màn hình:
  ```text
  ====================================================================
  🌐 PUBLIC HTTPS URL (Dùng được Micro trên mọi thiết bị/điện thoại):
  👉 https://example-subdomain.trycloudflare.com
  ====================================================================
  ```
* Bạn chỉ cần gửi đường link `https://...` này cho Leader hoặc đồng đội mở trên điện thoại/máy tính để thu âm thử nghiệm ngay lập tức.

#### Tùy chọn B: Chạy thủ công bằng CLI Cloudflared
Nếu bạn muốn tự quản lý tiến trình đường hầm riêng:

1. **Terminal 1:** Khởi động ASR Server:
   ```powershell
   python src/step4_quantization/asr/asr_demo_server.py
   ```
2. **Terminal 2:** Chạy lệnh mở Cloudflare Tunnel (yêu cầu đã cài đặt `cloudflared`):
   ```powershell
   cloudflared tunnel --url http://127.0.0.1:8420
   ```
   *Sao chép URL có đuôi `.trycloudflare.com` từ terminal để chia sẻ.*

---

## 3. Cơ chế Vận hành của Dashboard

Dashboard được thiết kế bám sát các chỉ đạo kỹ thuật từ Leader:

1. **Định tuyến Ngôn ngữ Tự động (Dynamic Model Routing):**
   * Người dùng nhấn chọn ngôn ngữ trên giao diện:
     * 🇻🇳 **Tiếng Việt:** Hệ thống tự động chuyển luồng sang mô hình **Zipformer-150M-CR-CTC** (INT16 NPU, giới hạn tối đa 15 giây/đoạn).
     * 🇺🇸 **English** / 🇨🇳 **中文** / 🇰🇷 **한국어:** Hệ thống tự động chuyển sang **SenseVoice-Small** (W8A16 NPU, giới hạn tối đa 29 giây/đoạn).
2. **Thu âm Đa đoạn (Batch Processing — Tận dụng 1 lần nạp trọng số NPU):**
   * Người dùng có thể nhấn **"🔴 Ghi đoạn tiếp theo"** nhiều lần liên tiếp để thu thập nhiều câu nói khác nhau vào hàng đợi (Queue).
   * Khi nhấn **"📤 Gửi (N đoạn)"**, toàn bộ các đoạn âm thanh sẽ được đóng gói thành một batch tensor gửi lên Qualcomm AI Hub trong **duy nhất một Job suy luận**, giúp triệt tiêu chi phí nạp nguội (cold-load latency) của NPU cho từng mẫu đơn lẻ.
3. **Giám sát Tiến trình Thời gian thực:**
   * Hiển thị dòng thời gian trạng thái (Timeline) trực tiếp từ Qualcomm AI Hub: *Cấp phát thiết bị $\rightarrow$ 🟢 Đang chạy trên chip silicon NPU $\rightarrow$ Hoàn tất*.
   * Báo cáo chi tiết độ trễ thực thi phần cứng thuần (Hardware Latency) và văn bản kết quả cho từng đoạn ghi âm.
