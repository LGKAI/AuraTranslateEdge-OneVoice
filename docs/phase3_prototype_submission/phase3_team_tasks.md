# PHASE 3 — PHÂN CÔNG & KẾ HOẠCH 10 NGÀY (06/11 → 16/11/2026)

**Thiết bị mục tiêu:** HSPTEK AIBOX 8550 (QCS8550, Hexagon v73) tại VNG Campus.
**Hạn nộp:** hết ngày **16/11/2026** (UTC+7) — nên nộp sớm trước 22h để tránh lỗi form.

## 1. Yêu cầu bắt buộc từ BTC

- Báo cáo theo **Template prototype** của BTC (xuất PDF).
- Video demo **có cảnh ngắt mạng** (Airplane mode / rút LAN) trước khi demo → chứng minh 100% offline.
- GitHub **Private**; chỉ thêm thành viên đội + tài khoản `saigonaihub` (https://github.com/saigonaihub).
- Nộp qua Form: https://forms.gle/Qhdni9hdFm19ZJfF6 (có thể cập nhật thay đổi thành viên/định hướng kỹ thuật so với Vòng 2).
- Thiết bị VNG: dự kiến sẵn sàng **09–15/11**; BTC sẽ gửi lịch qua email → phải xác nhận lịch trước khi đến.
- Hỗ trợ: info@saigonaihub.com; chuyên gia Qualcomm: vankle@qti.qualcomm.com (CC info@saigonaihub.com).
- QDC: khi gửi yêu cầu ghi rõ *OneVoice AI Challenge 2026 – AuraTranslate Edge* để được ưu tiên.

## 2. Phân công

### Khanh — Technical Lead
- Ghép hoàn chỉnh pipeline E2E (ASR → MT → TTS) chạy offline; chạy trên AI Hub, QDC và sẵn sàng cho thiết bị thật.
- Quyết định kiến trúc giao diện/app (dựa trên đề xuất của Kin) **trước 07/11**.
- **Quản lý GitHub:** đặt Private, mời `saigonaihub`, kiểm tra quyền truy cập (repo của Khanh).
- Dẫn dắt phiên lab tại VNG; chịu trách nhiệm số liệu kỹ thuật.
- Rà soát tính chính xác báo cáo; **nộp form 16/11**.

### Khánh — BTC Liaison & Project Manager
- Theo dõi email/group BTC hằng ngày, **đăng ký slot VNG ngay khi mở**; xác nhận lịch cho cả đội.
- Chia việc nhỏ, theo dõi tiến độ theo mốc.
- Cùng viết báo cáo, dựng video; rà soát bài nộp rồi chuyển Khanh.

### Hiếu — App Developer (chính)
- Phát triển app theo quyết định của Khanh (chạy cục bộ, 1 chạm/PTT, hiển thị bản gốc + bản dịch + độ trễ + trạng thái OFFLINE).
- Cùng viết báo cáo (phần giao diện/UX); **diễn viên chính** video.

### Khoa — App Developer (hỗ trợ)
- Hỗ trợ Hiếu: tích hợp pipeline, xử lý audio in/out, đóng gói script chạy 1 lệnh + `REPRODUCTION_GUIDE.md`.
- Chạy benchmark dữ liệu mẫu BTC (WER/CER, BLEU, RTF) lấy số liệu cho báo cáo.
- Cùng viết báo cáo; **diễn viên chính** video.

### Kin — Hardware Research & Video Director
- Nghiên cứu HSPTEK AIBOX 8550: **OS (Android hay Linux), cổng audio, mic/loa, màn hình, kết nối, cách nạp file (SSH/adb), runtime QNN có sẵn** → báo Khanh **trước 07/11**.
- Viết kịch bản video, chuẩn bị bối cảnh; **cầm máy quay** tại VNG.
- Cùng viết báo cáo (phần phần cứng).

## 3. Mốc thời gian & việc cần xong

| Mốc | Ngày | Việc cần xong |
| :--- | :--- | :--- |
| **M1** | 06–08/11 | • Khanh: compile lại model cho **QCS8550 (Proxy)** trên AI Hub (cần compile lại vì context binary phụ thuộc SoC), test QDC.<br>• Khánh: gửi yêu cầu QDC.<br>• Kin: báo kết quả khảo sát AIBOX 8550 → Khanh chốt app.<br>• Hiếu, Khoa: app chạy được với pipeline offline trên laptop.<br>• Kin: kịch bản video + danh sách đạo cụ. |
| **M2** | **08/11 (tối)** | **Demo-ready:** gói cài đặt offline + app + đồ mang lên VNG (laptop, USB/SSD chứa model, mic, loa, cáp LAN/USB-C, sạc, máy quay + tripod). |
| **M3** | 09–15/11 | • Khánh đăng ký slot ngay khi BTC mở.<br>• Lab VNG: cài đặt, chạy pipeline, đo độ trễ/RAM/%NPU/nhiệt độ, chụp ảnh + lưu log (Khanh, Khoa).<br>• Quay video: **ngắt mạng → hội thoại 2 chiều VI↔EN/ZH/KO → hiện số liệu** (Kin quay; Hiếu, Khoa diễn). |
| **M4** | 12–15/11 | • Viết báo cáo theo template BTC (cả đội, theo phần được giao), xuất PDF.<br>• Dựng video có phụ đề, 3–5 phút, giữ nguyên cảnh ngắt mạng (Khánh, Kin).<br>• Repo: README + hướng dẫn tái lập (Khoa); upload Drive, kiểm tra quyền xem link. |
| **M5** | **15/11** | Báo cáo + video + repo hoàn chỉnh; rà soát chéo (số liệu khớp log, link mở được bằng tài khoản ẩn danh). |
| **M6** | **16/11** | Khánh chuyển bản cuối → Khanh rà soát → **nộp form (trước 22h)**, lưu ảnh xác nhận. |

## 4. Rủi ro cần lưu ý

| Rủi ro | Cách xử lý |
| :--- | :--- |
| Thiết bị bảo trì chậm, lịch lab ít | Đăng ký sớm nhất; chuẩn bị gói cài đặt hoàn chỉnh để tối đa hóa thời gian lab; nếu chưa có slot, quay bản offline trên máy cá nhân + số liệu AI Hub/QDC, bổ sung cảnh thiết bị thật sau |
| Model hiện mới chạy qua AI Hub cloud, chưa chạy tại chỗ trên thiết bị | Kin/Khanh xác định runtime trên AIBOX sớm (QNN/ONNX Runtime); kiểm thử trên QDC trước |
| Nạp model lớn rất lâu (NLLB ~866s lần đầu) | Dùng context binary đã link sẵn, nạp trước khi quay |
| Hồ sơ nộp sát hạn | Hoàn thành 15/11; nộp form trước 22h ngày 16/11 |
