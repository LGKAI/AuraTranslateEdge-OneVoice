# Bản Nháp Câu trả lời cho Mẫu Báo cáo Nộp bài Prototype Phase 3

Các ký hiệu trạng thái: **[đã có]** số liệu đã đo đạc / văn bản đã soạn thảo; **[thiết bị]** dữ liệu thực tế đo đạc trực tiếp trên thiết bị phần cứng; **[nhóm]** quyết định hoặc nội dung do cả đội ngũ thống nhất.
Mọi số liệu trình bày dưới đây đều dựa trên kết quả đo kiểm thực tế, không có số liệu suy đoán chưa được kiểm chứng.

---

## 1. Tóm tắt Nguyên mẫu (Prototype Summary)

| Mục thông tin | Nội dung câu trả lời |
|---|---|
| **Đội thi / Dự án** | **Gia Sư Đỉnh Cao** — Dự án **AuraTranslate Edge (OneVoice)** |
| **Bản dựng nộp bài (Build)** | `s2s_edge 0.1.0`, git branch `main`, commit hash mới nhất trên repo chính thức |
| **Các ngôn ngữ demo** | Tiếng Việt ⇄ Tiếng Anh / Tiếng Trung / Tiếng Hàn (trọn vẹn 6 chiều dịch), tích hợp chế độ đàm thoại hai chiều `--pair` và nhận diện tự động `--auto` |
| **Kịch bản kiểm thử** | Đàm thoại thực tế tại phân xưởng sản xuất nhà máy (~30 phút) giữa Quản lý FDI và Công nhân Việt Nam |
| **Điều kiện môi trường kiểm thử** | Thử nghiệm trong điều kiện nhiễu công xưởng (SNR từ 0 dB đến 20 dB), khoảng cách thu âm micro từ 0.5m đến 2m |
| **Thiết bị phần cứng demo thực tế** | **HSPTEK AIBOX 8550** (SoC Qualcomm Snapdragon QCS8550, Hexagon NPU v73, RAM 16 GB LPDDR5x, 128 GB UFS, Ubuntu 22.04 LTS, QAIRT 2.51) cùng trạm đo kiểm vật lý đối chứng **Qualcomm Dragonwing IQ-9075 EVK** |
| **Năng lực & Tiến độ kỹ thuật** | Toàn bộ chuỗi Pipeline: ASR (Zipformer-150M / SenseVoice) ➔ MT (NLLB-200-600M giải mã Host Loop trên đồ thị tĩnh NPU) ➔ TTS (Piper / MeloTTS / Supertonic); cả 9 đồ thị mô hình đều được compile và link thành QNN Context Binary cho Hexagon v73 và kiểm chứng 100% NPU offload trên chip silicon thật (`docs/chip_results.md`). Các đột phá mới: Single-graph TTS cho cả 3 engine, NLLB tách Encoder 64 token và 4-beam decoder step, tự động phục hồi dấu câu tiếng Việt (`punct_vi`), nhận diện ngôn ngữ theo từng câu nói (LID), và xử lý tên riêng ngoại lai (`names.py`). Đã chạy thực tế trên thiết bị phần cứng AIBOX 8550 tại VNG. |

---

## 2. Video Trình diễn Demo (Demo Video)
Ghi hình toàn màn hình:
1. Chạy lệnh `bash tools/session0.sh` kiểm tra golden test toàn bộ 9 đồ thị NPU.
2. Ngắt toàn bộ kết nối mạng Wi-Fi/LAN (`sudo nmcli networking off`, chứng minh `ping 8.8.8.8` thất bại) để khẳng định vận hành **100% Offline**.
3. Khởi chạy chế độ hội thoại liên tục: `python -m s2s_edge live --pair vi,ko --continuous` hoặc mở Web Dashboard `python s2s_web.py`.
4. Ghi nhận các mốc thời gian (timestamps) tương tác giọng nói thực tế.

---

## 3. Gói Giải pháp Triển khai (Solution Package)
* **Gói cài đặt Offline độc lập (`s2s_edge_pkg/`):** Chứa mã nguồn, tài nguyên (`assets/`), thư mục `models_qcs8550/` (các file nhị phân QNN Context Binary biên dịch với QAIRT 2.51), kho wheelhouse ngoại tuyến (`wheelhouse_cp310/` cho Ubuntu 22.04 và `wheelhouse/` cho Ubuntu 24.04 aarch64), script `install.sh` và `run_demo.sh` [đã có].
* **Kho mã nguồn chính thức:** Nhánh `main` của repo nhóm (`LGKAI/AuraTranslateEdge-OneVoice`).
* **Kiến trúc phân bổ NPU:** 
  - ASR Tiếng Việt: Zipformer-150M CTC (FP16, 15s window, 100% NPU).
  - ASR Ngoại ngữ: SenseVoice-Small (Frontend FP16 + Encoder W8A16, 29s window, 100% NPU).
  - MT: NLLB-200-600M (Encoder FP16 64 token + Decoder Step 4-beam W8A16, 100% NPU).
  - TTS Tiếng Việt: Piper vais1000 (VITS byte-level single graph, 100% NPU).
  - TTS Tiếng Trung: MeloTTS-ZH (Front tĩnh + BERT + VITS single graph, 100% NPU).
  - TTS Tiếng Anh / Hàn: Supertonic-3 (Flow-matching 5 ODE steps single graph, 100% NPU).
  - Host CPU: I/O âm thanh, VAD năng lượng thích ứng, fbank, SentencePiece, logic beam search, chuẩn hoá số sang chữ.
* **Môi trường Runtime:** QAI AppBuilder 2.51.40 (QAIRT 2.51) trên Ubuntu 22.04 / 24.04 aarch64, Python 3.10 / 3.12. Chạy hoàn toàn độc lập, không gửi bất kỳ request nào ra ngoài mạng.
* **Yêu cầu phần cứng tối thiểu:** Thiết bị SoC Qualcomm có NPU Hexagon v73 (**HSPTEK AIBOX 8550** hoặc dòng Snapdragon 8 Gen 2 / IQ-9075); RAM $\ge 8$ GB (tập mô hình chiếm ~2.2–3.5 GB RAM); bộ nhớ trống $\ge 8$ GB; Micro + Loa USB ngoài (ALSA `snd-usb-audio`).

---

## 4. Kết quả Tự Đo kiểm & Đánh giá (Self-recorded Test Results)

| Tiêu chí đánh giá | Kết quả thực tế & Phương pháp đo | Tệp tin minh chứng |
|---|---|---|
| **Chất lượng Dịch máy (MT Quality)** | FLORES-200 devtest (văn bản chuẩn, 25 câu): VI→EN chrF **57.0** / BLEU **28.3**; EN→VI chrF **56.4** / BLEU **36.5**; ZH→VI chrF **47.7** / BLEU **25.4**; KO→VI chrF **46.8** / BLEU **25.1** (chế độ Beam 4). Tái tạo 99.8% chất lượng bản gốc HuggingFace FP32. | `logs/eval_mt_beam.json` |
| **Độ trễ Suy luận trên Chip (Inference Time)** | Thời gian thực thi chip thuần túy trên QCS8550: Zipformer **163.4 ms**, SenseVoice **294.8 ms**, NLLB Encoder **7.5 ms**, NLLB Decoder **15.2 ms/bước** (Greedy) và **30.8 ms/bước** (Beam 4), Piper **520.1 ms**, MeloTTS **740.6 ms**, Supertonic **701.2 ms**. | `s2s_edge/docs/chip_results.md` |
| **Chất lượng trên Dữ liệu mẫu BTC (OneVoice BTC Dataset)** | Đánh giá thực tế trên phần cứng AIBOX 8550 với tập BTC: Chiều EN→VI đạt chrF **50.13**, TTS lỗi chỉ **3.76%**; Chiều KO→VI đạt ASR lỗi CER chỉ **1.56%** (87.8% câu lỗi 0.0%), MT chrF tập sạch đạt **38.57**, TTS lỗi **7.29%**. | `evaluation_output.md`, `ko-vi.md`, `evaluation.txt` |
| **Dung lượng Mô hình (Model Footprint)** | Tổng dung lượng 9 đồ thị context binary: ~4.0 GB. Khi triển khai 1 chiều dịch nạp vào RAM chỉ mất **~2.2 GB**, nằm an toàn trong 16 GB RAM của AIBOX 8550. | `models_qcs8550/manifest.json` |
| **Độ trễ Toàn trình Thiết bị (End-to-End Latency)** | Đo đạc trên AIBOX 8550: Chiều KO→VI trung bình **3.54 s** cho câu 2.75s (Graph NPU chiếm 93.4%); Chiều EN→VI trung bình **7.19 s** cho câu dài 5.75s (Beam 4). Chế độ Greedy giảm xuống còn ~1.2s – 2.3s. | `ko-vi/ko-vi/BTC_results.csv`, `evaluation.txt` |

---

## 5. Giới hạn Đã biết & Kế hoạch Phát triển Tiếp theo
* **Giới hạn đã biết:** Xem chi tiết tại mục "Các giới hạn đã biết" trong `s2s_edge/README.md` và `docs/conversation.md` (tên riêng ngoại lai, nói chồng âm nhiều người, từ mượn tiếng Anh trong tiếng Việt).
* **Kế hoạch tiếp theo:**
  1. Kích hoạt Greedy Search cho các câu ngắn dưới 12 token để giảm 50% độ trễ MT.
  2. Áp dụng kỹ thuật Pipelining chồng chập (chạy ASR câu sau trong khi đang phát TTS câu trước) để tối ưu trải nghiệm đàm thoại liên tục.
  3. Bổ sung từ điển âm vị G2P chuyên biệt cho thuật ngữ kỹ thuật nhà máy sản xuất.
