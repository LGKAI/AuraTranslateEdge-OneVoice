# GHI CHÚ CHUẨN BỊ ON-DEVICE TEST (HSPTEK AIBOX 8550)
**Người phụ trách:** Lê Gia Khánh (OS & Hardware Lead — Laptop Fallback Lenovo LOQ)  
**Địa điểm & Thời gian:** VNG Campus (Saigon AI Hub), 15:00 – 18:00 (12/11 & 13/11/2026)

---

## 1. DỮ LIỆU & MÃ NGUỒN CẦN CHUYỂN LÊN HỘP (Thư mục đích: `/data/`)

Hộp AIBOX 8550 là một chiếc máy tính độc lập chạy Ubuntu 22.04 LTS. Cần đưa 4 thành phần sau từ laptop lên thư mục `/data/` của hộp:

1. **Thư mục code `s2s_edge/`:** Bộ não điều hành toàn bộ chuỗi (thu âm mic $\rightarrow$ cắt câu VAD $\rightarrow$ nạp mô hình vào NPU $\rightarrow$ phát loa $\rightarrow$ đo thời gian benchmark).
2. **Thư mục mô hình `models_qcs8550/` (~4.0 GB):** Chứa 9 file `.bin` NPU do Khanh xuất từ Qualcomm AI Hub (Zipformer ASR tiếng Việt, SenseVoice ASR đa ngữ, NLLB dịch máy, Piper/MeloTTS/Supertonic đọc tiếng). NPU chỉ chạy được các file `.bin` này.
3. **Thư mục dữ liệu mẫu `prototype_sample_dataset/` (~146 MB):** Dữ liệu đề thi của BTC (Vie-Eng, Vie-Kor, Vie-Man) để chạy chấm điểm.
4. **Thư mục `wheelhouse_cp310/` (~300 MB):** Chứa file cài đặt offline `qai_appbuilder-2.51.40` cho hộp khi phòng thi không có Internet.

---

## 2. CÔNG CỤ TRÊN LAPTOP (Checklist kiểm tra)

| Công cụ | Mục đích | Trạng thái trên máy bạn |
|---|---|:---:|
| **ADB (`adb`)** | Mở terminal điều khiển hộp (`adb shell`), đẩy file và lấy log qua cáp Type-C mà **không cần mạng**. *(Bắt buộc)* | **ĐÃ CÀI XONG & TEST THÀNH CÔNG** (`C:\platform-tools\adb.exe`) |
| **FFmpeg (`ffmpeg`)** | Xử lý các file âm thanh đề thi BTC cho script kiểm thử. *(Bắt buộc)* | **ĐÃ CÀI XONG** (Bản 9.0.2 Gyan.FFmpeg) |
| **Google USB Driver** | Driver nhận diện chip Qualcomm phòng khi Windows báo thiếu driver. | **ĐÃ TẢI SẴN** (Lưu tại `D:\usb_driver`) |
| **Môi trường Python** | Chạy kiểm tra cục bộ trên máy. | **SẴN SÀNG 100%** (Python 3.12, đủ thư viện) |

---

## 3. THIẾT BỊ CẦN ĐEM THEO VÀO BALO

| Vật dụng | Mục đích | Bắt buộc không? |
|---|---|:---:|
| **Laptop Lenovo LOQ + Củ sạc** | Máy trạm fallback điều khiển và cứu hộ nếu máy Khanh gặp sự cố. | **BẮT BUỘC** |
| **Cáp USB Type-C (đã test)** | Cắm trực tiếp laptop vào hộp AIBOX để chạy lệnh `adb`. | **BẮT BUỘC** *(Đã test thành công với Redmi)* |
| **Tai nghe có mic / USB Mic + Loa** | Cắm vào cổng USB 3.0 của hộp để test chế độ dịch trực tiếp (hộp không có mic/loa trong). | **BẮT BUỘC** *(Trong nhóm phải có)* |
| **USB $\ge 16\text{ GB}$** | Chép nhanh 4GB file `.bin` từ máy Khanh sang máy bạn; cắm trực tiếp vào hộp AIBOX nếu cần. | **RẤT NÊN CÓ** *(Trong nhóm)* |
| **Điện thoại Android** | Dự phòng phát 4G/Wi-Fi Hotspot cho laptop khi cần. | **NÊN MANG THEO** |
| *Cáp mạng LAN RJ45* | *Dự phòng kết nối SSH nếu cổng Type-C hỏng.* | **KHÔNG BẮT BUỘC** *(Đã có cáp Type-C làm hết)* |

---

## 4. QUY TRÌNH 4 BƯỚC THAO TÁC TRÊN HỘP TẠI BÀN THI

Cắm cáp Type-C từ laptop vào hộp AIBOX 8550, mở PowerShell gõ:

```powershell
# Bước 1: Kiểm tra nhận diện hộp
adb devices
# Kết quả hiện: <mã_thiết_bị>  device

# Bước 2: Bắn code và dữ liệu từ laptop lên hộp
adb push s2s_edge /data/
adb push data/prototype_sample_dataset /data/

# Bước 3: Mở terminal vào bên trong hộp
adb shell
cd /data/s2s_edge

# Bước 4: Chạy kiểm định tự động toàn diện Session 0
bash tools/session0.sh
```
*(Chạy xong, script sẽ tự động đóng gói toàn bộ log kiểm định vào file `logs/session0_<ngày_giờ>.tgz` để nộp cho BTC)*.
