# Đánh giá repo TTS của đồng đội (`pckhoa206/onevoice_AI_challeng-TTS-`)

Đọc README, docs 00/02/03 và các script chính (`deploy_dragonwing_fresh_full_pipeline.py`,
`deploy_tasks_5_6_7_to_aihub.py`, `deploy_true_full_dragonwing_npu.py`, `build_end_to_end_fused_tts_graph.py`,
và các script tokenizer/fuse). **Chưa đọc hết 167 file**, nên có thể sót. Không có mã nào của repo đó được sao chép
vào nhánh này.

## Họ đã làm gì

- Supertonic 3 tách 4 submodel (duration predictor, text encoder, vector estimator, vocoder), quantize W8A16 trên AI Hub,
  compile cho Dragonwing IQ-9075 và Galaxy S24, có đủ job ID quantize/compile/profile/inference.
  Thời gian chip họ báo trên Dragonwing: 5.4 / 7.2 / 72.3 (mỗi bước ODE) / 43.3 ms.
- Nhiều tối ưu hợp lý: One-Hot GEMM thay Gather, mặt nạ tạo trong graph, nhiễu tĩnh nhúng sẵn, vocoder gộp FIR 16 kHz + PCM16,
  tokenizer ký tự dạng ONNX, bản ODE 5 bước dùng chung trọng số, tokenizer C++ theo chuẩn QNN OpPackage.

## Điều đã kiểm chứng trong mã

| Nội dung | Phát hiện |
|---|---|
| Audio "Dragonwing" / S24 họ nộp | Trong `deploy_dragonwing_fresh_full_pipeline.py` (bước 4B) và `deploy_tasks_5_6_7_to_aihub.py`, **duration predictor, text encoder và vector estimator chạy trên CPU** bằng `onnxruntime` + `CPUExecutionProvider`; **chỉ vocoder** được gửi lên chip |
| Bản "true full" (`deploy_true_full_dragonwing_npu.py`) | Cả 4 submodel gửi lên chip nhưng chỉ **3 bước ODE** (README ghi 5), nhiễu khởi tạo sinh bằng `np.random.randn` ở máy chủ, mỗi bước một job AI Hub. Tôi không thấy audio do bản này tạo trong repo |
| Graph gộp (`build_end_to_end_fused_tts_graph.py`) | Có script ghép, nhưng README/báo cáo JSON không có job nào cho nó; đầu vào `char_codes [1,54]` giới hạn 54 ký tự (18 chữ Hangul). Các file `*_static.onnx` mà nó đọc không do script nào trong repo tạo ra |
| Tokenizer C++ QNN | chạy trên CPU của máy chủ (README ghi "CPU Host / DSP HVX"), không phải trên NPU |
| Token API | `deploy_dragonwing_fresh_full_pipeline.py` có hằng `DEFAULT_TOKEN` chứa một chuỗi trông như token AI Hub **trong repo công khai**. Nên thu hồi/tạo lại token và xoá khỏi lịch sử git. Giá trị đó không được sao chép vào repo này |

## Chất lượng audio họ nộp (tôi chạy ASR SenseVoice fp32)

| File | Kết quả |
|---|---|
| `live_dragonwing_fresh_english_complete.wav` | WER 28%, CER 17% |
| `live_dragonwing_fresh_korean_complete.wav` | CER 27% |

Nghe được nhưng chưa tốt. Chỉ hai file được chấm, chưa chấm các file S24.

## Chỗ báo cáo mô tả nhiều hơn thực tế

- README: "Pure NPU", "0% CPU fallback", "offload ~95% FLOPs" — nhưng audio nộp chỉ có vocoder trên NPU.
- Chất lượng đo bằng tỉ lệ năng lượng trong dải giọng nói (78–97%): cho biết có tín hiệu giọng nói, không cho biết đúng chữ.
- "WER 0.00%": không rõ audio nào, từ pipeline nào, và chỉ đạt "sau chuẩn hoá văn bản" (7.93% khi chưa chuẩn hoá).
- Cosine 0.996 của vector estimator trên chip: chỉ đo một bước ODE. Trong thí nghiệm của chúng tôi, bước 0 cho 0.998 ngay cả
  khi graph đang hỏng hoàn toàn, nên con số này không phát hiện được lỗi (xem `bug_reducesum.md`).

## Chưa kiểm chứng

- Bản vector estimator NPU của họ có dính lỗi `ReduceSum` của HTP không. Cùng cấu trúc model nên nhiều khả năng có,
  nhưng chúng tôi chưa chạy thử trên các graph họ đã compile.
- Tiếng Việt/tiếng Trung và số liệu độ trễ end-to-end của họ.
