# Module TTS — Tổng hợp Giọng nói Đa ngữ trên Qualcomm Hexagon NPU

Thư mục này quản lý toàn bộ mã nguồn tối ưu hóa đồ thị tĩnh, biên dịch NPU, vá lỗi vi kiến trúc phần cứng và ứng dụng Web Dashboard thời gian thực cho hệ thống chuyển văn bản thành giọng nói (**Text-to-Speech - TTS**) 4 ngôn ngữ trên nền tảng **Qualcomm Dragonwing IQ-9075 EVK (Hexagon NPU v73, 100 dense TOPS)**:

| Ngôn ngữ | Mô hình | Kiến trúc Graph NPU | Tần số mẫu | NPU Offload | Độ trễ NPU (Profile thật) | RAM đỉnh | Độ chính xác / Chất lượng |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🇻🇳 **Tiếng Việt** | **Piper-VITS** (`vi_VN-vais1000`) | Single Static DAG (`fullg8`) | 22.05 kHz | **100.0%** (2.077/2.077) | **480.00 ms** | 8.0 MB | Khớp espeak **100%** (361/370 câu FLEURS) |
| 🇺🇸 **English** | **Supertonic 3** (Voice M1) | Single Static DAG (`full_v3`) | 44.10 kHz | **100.0%** (4.476/4.476) | **748.30 ms** | 10.0 MB | **0.0% WER** (SenseVoice Roundtrip) |
| 🇰🇷 **한국어** | **Supertonic 3** (Voice M1) | Single Static DAG (`full_v3`) | 44.10 kHz | **100.0%** (4.476/4.476) | **748.30 ms** | 10.0 MB | **1.5% CER** (SenseVoice Roundtrip) |
| 🇨🇳 **中文** | **MeloTTS-ZH** (Mandarin) | Single Static DAG (`melo_one`) | 44.10 kHz | **100.0%** (3.451/3.451) | **720.00 ms** | 9.5 MB | **6.4% CER** (Ngang bằng CPU gốc 6.0%) |

---

## 1. Cấu trúc Thư mục

```text
src/step4_quantization/tts/
├── README.md                      # Hướng dẫn module TTS & cách chạy Dashboard (Tài liệu này)
├── tts_demo_server.py             # FastAPI Web Dashboard tích hợp 4 ngôn ngữ, hỗ trợ Cloudflare Tunnel
├── piper/                         # Chuyên sâu Piper-VITS tiếng Việt (Single Static DAG)
│   ├── README.md                  # Báo cáo kỹ thuật chi tiết & kiểm thử trên chip silicon
│   ├── piper_npu/                 # Mã nguồn G2P tĩnh, export đồ thị gộp & submit Qualcomm AI Hub
│   │   ├── g2p/                   # Xây dựng bảng tra UTF-8 -> phoneme (29.484 âm tiết)
│   │   ├── export_full.py         # Ghép toàn bộ G2P + VITS thành 1 graph duy nhất
│   │   ├── run_hw.py              # Profile và suy luận trên chip Dragonwing
│   │   └── eval_hw.py             # Đánh giá chất lượng sóng âm so với FP32 CPU
│   └── results/                   # File âm thanh WAV sinh ra trực tiếp từ chip silicon NPU
├── supertonic/                    # Chuyên sâu Supertonic 3 tiếng Anh & tiếng Hàn (Flow Matching)
│   ├── README.md                  # Báo cáo kỹ thuật toàn diện, phân tích lỗi vi kiến trúc HTP
│   ├── supertonic_npu/            # Pipeline tĩnh hóa, ghép graph, vá lỗi ReduceSum & submit Hub
│   │   ├── scripts/               # staticize.py, rot_fix.py, fuse.py, hub_run.py, eval_hw.py
│   │   ├── diagnostics/           # Chẩn đoán phân rã đồ thị tìm nguyên nhân lỗi phần cứng HTP
│   │   └── docs/                  # Tài liệu chi tiết bug_reducesum.md & deploy_log.md
│   └── results/                   # File âm thanh 44.1 kHz WAV đo kiểm trên chip silicon
└── melotts/                       # Chuyên sâu MeloTTS-ZH tiếng Trung (BERT + VITS Single DAG)
    ├── README.md                  # Báo cáo kỹ thuật, bảng tra tĩnh 7.335 token & link context binary
    ├── melotts_npu/               # Pipeline chưng cất front, BERT tĩnh, vocoder cửa sổ & AI Hub
    │   ├── melo_front.py          # Bảng tra tĩnh chưng cất thay thế triệt để jieba/pypinyin
    │   ├── melo_one.py            # Đồ thị gộp 100% NPU (Front + BERT 22 lớp + VITS)
    │   ├── step17_one_graph_linked.py # Link context binary nạp siêu tốc 0.9s & benchmark 40 câu
    │   └── docs/                  # Nhật ký chi tiết ket_qua_chip.md & các bước thử nghiệm
    └── results/                   # File âm thanh WAV 44.1 kHz tổng hợp từ chip thật NPU
```

---

## 2. Các Đột phá Kỹ thuật & Vá lỗi Vi kiến trúc NPU

Khác với các triển khai TTS thông thường vốn phụ thuộc nặng nề vào CPU máy chủ để tách từ, chuyển âm (G2P), dự đoán thời lượng và ghép cửa sổ vocoder, toàn bộ 3 mô hình TTS tại thư mục này đều được tái cấu trúc thành **Single Static DAG (Một đồ thị tính toán tĩnh duy nhất)**, chạy **100% thuần phần cứng trên Hexagon NPU**:

### 2.1. Piper TTS (Tiếng Việt) — Bảng tra G2P tĩnh 29.484 âm tiết
* **Vấn đề:** Công cụ `espeak-ng` không thể biên dịch lên NPU, khiến hầu hết các hệ thống phải giữ G2P trên CPU.
* **Giải pháp:** Tận dụng đặc tính âm tiết tiếng Việt có cách phát âm không đổi theo ngữ cảnh, xây dựng bảng tra tĩnh gồm **29.484 âm tiết** biểu diễn bằng phép toán ma trận (`MatMul`) và so sánh byte UTF-8 tĩnh ngay bên trong đồ thị ONNX.
* **Kết quả:** Khớp hoàn toàn 100% đầu ra của `espeak-ng` trên 361/370 câu test FLEURS tiếng Việt. Vocoder áp dụng cơ chế cửa sổ theo nhóm (`group=8`) giúp NPU xử lý trơn tru mà không bị tràn bộ nhớ (`MEM_ALLOC`), đạt độ trễ **480 ms** cho toàn bộ chuỗi từ byte văn bản ra âm thanh 22.05 kHz.

### 2.2. Supertonic 3 (Tiếng Anh & Hàn) — Vá lỗi phần cứng HTP `ReduceSum`
* **Vấn đề:** Khi biên dịch toàn bộ chuỗi Flow Matching (Euler ODE 5 bước + Vocoder) lên NPU, mô hình chạy thành công 100% NPU nhưng âm thanh sinh ra bị nhiễu loạn hoàn toàn (**WER 95.1%**).
* **Phát hiện:** Quá trình chẩn đoán phân rã đồ thị phát hiện toán tử `ReduceSum(axes=[1,2])` trên mặt nạ latent bị lỗi vi kiến trúc phần cứng trên HTP, trả về giá trị cố định là `1` thay vì độ dài thực tế là `62`. Điều này khiến góc quay Rotary Position Embedding (RoPE) trong 4 khối Cross-Attention bị sai lệch ~62 lần.
* **Giải pháp:** Thay thế toàn bộ cụm `ReduceSum` bằng phép tương đương toán học **`Reshape + MatMul` với vector toàn 1** (tương thích tuyệt đối với HTP compiler).
* **Kết quả:** Chất lượng âm thanh khôi phục hoàn hảo, đạt **WER 0.0%** (Tiếng Anh) và **CER 1.5%** (Tiếng Hàn) khi nhận dạng ngược bằng SenseVoice; thời gian thực thi chip cố định **748.3 ms** cho âm thanh 44.1 kHz chất lượng cao.

### 2.3. MeloTTS-ZH (Tiếng Trung) — Xóa bỏ CPU Host & Tách bit FP16
* **Vấn đề:** Tiếng Trung có cấu trúc chữ Hán phức tạp đòi hỏi bộ phân tích từ vựng `jieba`, chuyển âm `pypinyin` và xử lý biến điệu thanh điệu rất nặng trên CPU. Ngoài ra, việc token ID vượt quá 2048 gây lỗi tràn số nghiêm trọng trong tính toán nửa chính xác FP16 của NPU.
* **Giải pháp:**
  1. **Chưng cất Front tĩnh:** Rút trích luật phát âm từ 490.000 câu OPUS-100 thành bảng tra tĩnh 7.335 token và 69.000 quy tắc ghi đè ngữ cảnh, đưa thẳng vào đồ thị ONNX.
  2. **Tách dải số học Hi/Lo FP16:** Thay vì so sánh trực tiếp token ID lớn hơn 2048 (làm mất chính xác do độ phân giải bit của FP16), đồ thị phân tách thành 2 nhánh so sánh riêng biệt `hi` và `lo` (mỗi nhánh $< 2048$).
  3. **Context Binary Linking:** Sử dụng `submit_link_job` để liên kết trước nhị phân thực thi, giảm thời gian nạp nguội mô hình từ **514 giây xuống chỉ còn 0.9 giây**.
* **Kết quả:** Single Static DAG tích hợp cả Front tĩnh + BERT 22 lớp + VITS, đạt độ trễ **720 ms** trên NPU, CER đạt **6.4%** trên 40 câu thử nghiệm (tương đương bản chạy CPU gốc 6.0%).

---

## 3. Bảng Tra cứu Job ID Đo kiểm trên Chip Thật (Qualcomm AI Hub)

Toàn bộ số liệu công bố đều được đo kiểm trực tiếp trên thiết bị **Dragonwing IQ-9075 EVK** thông qua Qualcomm AI Hub:

| Mô hình | Ngôn ngữ | Compile Job ID | Profile Job ID | Inference Job ID | Ghi chú Trạng thái |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Piper TTS** | 🇻🇳 Vi | `fullg8` | `fullg8_v2_profile` | `fullg8_v2_infer` | 2.077 layers NPU, 480 ms, wav 22.05 kHz |
| **Supertonic 3** | 🇺🇸 En / 🇰🇷 Ko | [`jprxn9nep`](https://workbench.aihub.qualcomm.com/jobs/jprxn9nep/) | [`jpe6l2l0g`](https://workbench.aihub.qualcomm.com/jobs/jpe6l2l0g/) | [`jgzz4w46g`](https://workbench.aihub.qualcomm.com/jobs/jgzz4w46g/) | 4.476 layers NPU, 748.3 ms, 10 MB RAM |
| **MeloTTS-ZH** | 🇨🇳 Zh | [`jpy8mw17g`](https://workbench.aihub.qualcomm.com/jobs/jpy8mw17g/) | [`j5m9m9e9g`](https://workbench.aihub.qualcomm.com/jobs/j5m9m9e9g/) | [`jgn1n10qp`](https://workbench.aihub.qualcomm.com/jobs/jgn1n10qp/) | 3.451 layers NPU, 720 ms, link: `jprx0xv7p` |

---

## 4. Hướng dẫn Khởi chạy TTS Demo Dashboard

Tệp [`tts_demo_server.py`](tts_demo_server.py) là máy chủ web tích hợp giao diện điều khiển hiện đại (Glassmorphism Dark Theme), hỗ trợ nhập văn bản tùy ý và xuất file giọng nói đa ngữ với hai bộ máy thực thi:
* 🟢 **NPU Qualcomm Dragonwing (Qualcomm AI Hub):** Kết nối trực tiếp chip silicon NPU vật lý Hexagon v73 trên Qualcomm AI Hub. Người dùng có thể chạy thử nghiệm suy luận đám mây hoặc tải ngay các tệp WAV thực tế được tổng hợp từ phần cứng NPU (100% offload).
* ⚡ **Local CPU (Tổng hợp Trực tiếp trên Host):** Tự động tổng hợp và chuyển đổi bất kỳ câu văn bản nào người dùng nhập vào thành tệp âm thanh WAV chất lượng cao trong vài chục mili-giây mà không cần chờ hàng đợi đám mây.

### Cách 1: Chạy Local (Nội bộ trên máy cá nhân)

Sử dụng khi thử nghiệm trực tiếp trên máy trạm phát triển:

```powershell
python src/step4_quantization/tts/tts_demo_server.py
```

* Mở trình duyệt bất kỳ (Chrome, Edge, Firefox) và truy cập: **`http://localhost:8440`**
* Có thể tùy biến cổng hoặc host nếu cần:
  ```powershell
  python src/step4_quantization/tts/tts_demo_server.py --port 8440 --host 127.0.0.1
  ```

---

### Cách 2: Chạy Public ra ngoài Internet bằng Cloudflare Tunnel (Có HTTPS)

> [!IMPORTANT]
> **Tại sao bắt buộc phải dùng HTTPS khi chia sẻ ra ngoài mạng?**  
> Trình phát âm thanh Web Audio API hiện đại và các chính sách bảo mật trình duyệt di động (iOS Safari, Android Chrome) yêu cầu kết nối bảo mật **HTTPS** để đảm bảo khả năng giải mã âm thanh liên tục và tương thích với micro/loa ngoài. Nếu chia sẻ qua HTTP nội bộ (`http://192.168.x.x`), nhiều thiết bị di động sẽ chặn tự động phát âm thanh (Autoplay Policy).  
> **Cloudflare Tunnel** cung cấp đường hầm an toàn, tự động cấp chứng chỉ SSL/HTTPS hợp lệ và hoàn toàn miễn phí mà không cần mở cổng modem (NAT/Port Forwarding).

#### Tùy chọn A: Tự động hoàn toàn (Khuyên dùng)
Chỉ cần thêm cờ `--share` vào lệnh khởi động:

```powershell
python src/step4_quantization/tts/tts_demo_server.py --share
```

* Script sẽ tự động phát hiện `cloudflared.exe` (hoặc tự động tải về nếu chưa có), khởi tạo đường hầm an toàn và in ra màn hình:
  ```text
  ====================================================================
  🌐 PUBLIC HTTPS URL (Trình phát Audio hoạt động trên mọi thiết bị):
  👉 https://example-subdomain.trycloudflare.com
  ====================================================================
  ```
* Bạn chỉ cần gửi đường link `https://...` này cho Leader, Thầy Cô hoặc đồng đội mở trên điện thoại/laptop để trải nghiệm nghe thử giọng nói tổng hợp từ chip NPU ngay lập tức.

#### Tùy chọn B: Chạy thủ công bằng CLI Cloudflared
Nếu bạn muốn tự quản lý tiến trình đường hầm riêng:

1. **Terminal 1:** Khởi động TTS Server:
   ```powershell
   python src/step4_quantization/tts/tts_demo_server.py --port 8440
   ```
2. **Terminal 2:** Chạy lệnh mở Cloudflare Tunnel (yêu cầu đã cài đặt `cloudflared`):
   ```powershell
   cloudflared tunnel --url http://127.0.0.1:8440
   ```
   *Sao chép URL có đuôi `.trycloudflare.com` từ terminal để chia sẻ.*

---

## 5. Tính năng Nổi bật trên TTS Demo Dashboard

1. **Chuyển đổi 4 Ngôn ngữ Linh hoạt:** Click chọn nhanh 🇻🇳 Tiếng Việt, 🇺🇸 English, 🇰🇷 한국어, hoặc 🇨🇳 中文 với thông số phần cứng NPU hiển thị tương ứng.
2. **Bộ câu mẫu Tiêu chuẩn 1-chạm:** Cung cấp sẵn các mẫu câu giao tiếp đời sống, hội thoại du lịch và câu chuẩn FLEURS kèm bản dịch đối soát.
3. **Trình phát m thanh Trực quan:** Tích hợp bộ tạo sóng âm (Audio Waveform Visualizer), thanh đo âm lượng (VU Meter) và hỗ trợ tải trực tiếp tệp âm thanh `.wav` về máy.
4. **Giám sát Phần cứng Thời gian thực:** Báo cáo chi tiết số lượng lớp NPU (NPU Layer Offload), RAM đỉnh, độ trễ phần cứng thuần (Hardware Latency) và các mã Job ID chính thức từ Qualcomm AI Hub.
