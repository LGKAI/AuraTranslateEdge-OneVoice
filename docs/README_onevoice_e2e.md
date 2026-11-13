# OneVoice — Hệ Thống Dịch Giọng Nói Sang Giọng Nói (Speech-to-Speech E2E)
### Tích hợp Runtime Trọn Gói, Module TTS Hybrid Format Nâng Cấp & Bộ Báo Cáo Thực Nghiệm

---

## 1. Giới thiệu tổng quan

Thư mục này là **bản đóng gói hoàn chỉnh End-to-End (E2E)** của hệ thống OneVoice Speech-to-Speech (S2S), hỗ trợ 4 ngôn ngữ (**Tiếng Việt `vi`, Tiếng Anh `en`, Tiếng Hàn `ko`, Tiếng Trung `zh`**). 

Hệ thống đã được tích hợp mặc định bộ xử lý **TTS Hybrid Format** (`S2S_TTS_FORMAT=hybrid`), giải quyết triệt để các lỗi đọc sai định dạng số, giờ giấc, ngày tháng, mã bưu chính (ZIP), số điện thoại và mã định danh mà không làm biến đổi hay ảo giác (hallucination) dữ liệu gốc.

---

## 2. Kiến trúc luồng xử lý End-to-End (E2E Pipeline)

```
[Audio Đầu Vào]
       │
       ▼
[Audio Segmenter & VAD]
       │
       ▼
[ASR - Nhận dạng giọng nói]
  ├── Tiếng Việt: Zipformer-Small (CTC static 1500)
  └── EN / KO / ZH: SenseVoice-Small (2-graph ONNX)
       │
       ▼
[Khôi phục dấu câu & Viết hoa (Punctuation Restoration)]
  └── viBERT-CAPU (int8 ONNX, torch-free)
       │
       ▼
[MT - Dịch máy đa ngữ]
  └── NLLB-600M (Encoder + Unrolled Decoder Beam 4, Vocabulary Pruned)
       │
       ▼
[★ NÂNG CẤP: TTS Hybrid Format Text Normalization]
  ├── Parser có kiểu nhận diện: giờ, ngày, tiền tệ, số âm, thập phân, %, đơn vị
  ├── Bộ quy tắc & CRF bảo vệ mã: ZIP, phone, flight code, alnum
  └── Adaptive Chunking: Tính ngân sách graph chuẩn xác (gồm cả NFKD Jamo tiếng Hàn)
       │
       ▼
[TTS - Tổng hợp giọng nói]
  ├── Tiếng Việt: Piper VITS (piper_vi_full_g8.onnx)
  ├── EN / KO: Supertonic 3 (fused text-to-wav single graph)
  └── Tiếng Trung: MeloTTS-ZH (fused UTF-8 -> 44.1 kHz audio)
       │
       ▼
[Audio Đầu Ra (WAV)]
```

---

## 3. Chi tiết Nâng cấp TTS Hybrid Format

### 3.1. Cơ chế hoạt động
* **Bảo toàn dữ liệu tuyệt đối:** Sử dụng parser grammar xác định loại dữ liệu và chuyển đổi lời đọc theo quy tắc chuẩn ngôn ngữ đích, **không để model tự do sinh lại số hay chữ**.
* **Độ trễ siêu thấp:** Thời gian xử lý chuẩn hóa CPU trung bình chỉ **0,23 ms** (median 0,17 ms, p95 0,47 ms), không làm ảnh hưởng đến độ trễ thời gian thực.
* **Ngân sách chia đoạn an toàn:** Tự động chia nhỏ câu theo ranh giới từ và byte UTF-8; với tiếng Hàn, tự động phân tích và tính toán chiều dài sau khi tách âm tiết thành Jamo (NFKD).

### 3.2. Khắc phục các lỗi định dạng thực tế
| Định dạng | Ví dụ đầu vào | Trước nâng cấp (Lỗi) | Sau nâng cấp (TTS Hybrid) |
|---|---|---|---|
| **Giờ phút** | `5:30 sáng` | "năm : ba mươi" | **"năm giờ ba mươi phút sáng"** |
| **Giờ chẵn** | `08:00` | "tám : không" | **"tám giờ"** |
| **Mã bưu chính** | `ZIP 67297` | "sáu mươi bảy nghìn..." | **"ZIP sáu bảy hai chín bảy"** (đọc từng số) |
| **Số nhà** | `Số nhà 734` | "năm ba mươi bốn" (ByT5 sai) | **"Số nhà bảy trăm ba mươi bốn"** |
| **Số điện thoại** | `0908 246 135` | "chín trăm linh tám..." | **"không chín không tám hai bốn sáu..."** |
| **Mã chuyến bay** | `VN247` | "vơn hai trăm..." | **"vê en hai bốn bảy"** |
| **Tiền tệ** | `$18.25` | "$ mười tám chấm..." | **"mười tám đô la hai mươi lăm xen"** (EN: *eighteen dollars twenty-five cents*) |
| **Tiếng Hàn** | `3:30` | Lỗi chia Jamo / cắt âm | **`세 시 삼십 분`** (bảo toàn ngân sách Jamo) |
| **Tiếng Trung** | Năm `2027` | Đọc thành số đếm | **`二零二七`** (sửa cả thang 10^12 万亿) |

---

## 4. Danh mục Báo cáo & Tài liệu Thực nghiệm (Thư mục `reports/`)

Toàn bộ các tài liệu đánh giá và báo cáo định dạng TTS được lưu trữ tại `reports/`:

1. **[bao-cao-format-tts-hybrid.pdf](reports/bao-cao-format-tts-hybrid.pdf)** *(57 KB)*:
   * **Báo cáo chính thức:** *"Format TTS hybrid — Đã làm được & việc tiếp theo"* (Bản cập nhật ngày 12/11/2026).
   * Phân tích thực nghiệm trên 4 ngôn ngữ (VI/EN/KO/ZH), 55 ví dụ ngữ nghĩa, 1.600 ca sinh cấu trúc và đối chiếu ASR Whisper.
2. **[bao-cao-tts-anh-viet-3-phuong-phap.pdf](reports/bao-cao-tts-anh-viet-3-phuong-phap.pdf)** *(148 KB)*:
   * Báo cáo nghiên cứu so sánh 3 phương pháp: **M1 (WFST)**, **M2 (CRF + WFST)** và **M3 (WFST + DistilBERT)** trên 96 ca thử nghiệm.
3. **[bao-cao-fix-normalize-anh-viet.pdf](reports/bao-cao-fix-normalize-anh-viet.pdf)** *(61 KB)*:
   * Báo cáo về các cải tiến chuẩn hóa văn bản trước MT và sau MT.
4. **[bao_cao_danh_gia_audio_en_vi.pdf](reports/bao_cao_danh_gia_audio_en_vi.pdf)** *(804 KB, kèm bản HTML [bao_cao_danh_gia_audio_en_vi.html](reports/bao_cao_danh_gia_audio_en_vi.html))*:
   * Báo cáo phân tích đối chiếu lỗi và thẩm định chất lượng Speech-to-Speech chiều Anh - Việt.
5. **[hybrid_format_plan.md](reports/hybrid_format_plan.md)**:
   * Kế hoạch triển khai 7 bước và tiêu chí nghiệm thu không lỗi giá trị.
6. **Thư mục [results_hybrid_format/](reports/results_hybrid_format/)**:
   * Tập hợp các kết quả máy đọc được: `evaluation.json`, `semantic_examples.json`, `cardinal_crosscheck.json`, `audio_crosscheck.json`.

---

## 5. Cấu trúc thư mục

```
OneVoice_E2E_HybridTTS/
├── README.md                          # Tài liệu tổng quan này
├── requirements.txt                   # Danh sách thư viện Python
├── s2s_edge/                          # Core Runtime Package (ASR, MT, Hybrid TTS, Punct)
│   ├── s2s_edge/                      # Các module Python chính
│   │   ├── hybrid_textnorm.py         # Module chuẩn hóa Hybrid chính
│   │   ├── format_locales.py          # Quy tắc đọc số/ngày/giờ/đơn vị 4 ngôn ngữ
│   │   ├── tts_format.py              # Chunking & điều phối TTS format
│   │   ├── identifier_tagger.py       # Nhận diện chuỗi mã (CRF)
│   │   ├── identifiers.py             # Bộ quy tắc bảo vệ thực thể
│   │   ├── tts.py                     # Wrapper TTS thống nhất (Piper/Supertonic/Melo)
│   │   ├── asr.py, mt.py, punct_vi.py # Các module ASR, MT, Dấu câu
│   │   └── pipeline.py                # Translator E2E
│   └── assets/                        # Dữ liệu từ điển phát âm & style
├── components/                        # Mã nguồn exporter NPU của các nhánh
├── models/                            # Cấu hình nạp graph (models.cpu.json)
├── build/                             # Các đồ thị ONNX đã được xuất và tối ưu
├── reports/                           # Các file Báo cáo PDF & tài liệu thực nghiệm
│   ├── bao-cao-format-tts-hybrid.pdf
│   ├── bao-cao-tts-anh-viet-3-phuong-phap.pdf
│   ├── bao-cao-fix-normalize-anh-viet.pdf
│   ├── bao_cao_danh_gia_audio_en_vi.pdf
│   └── results_hybrid_format/
├── research/                          # Bộ công cụ benchmark FST, CRF & ReportLab
│   └── tts_format_benchmark/
├── tests/                             # Bộ kiểm thử tự động
│   ├── test_hybrid_format.py          # 10 test suites kiểm thử hybrid format
│   └── tests_identifiers.py           # 5 test suites kiểm thử identifiers
├── tools/                             # Công cụ chạy E2E, web server & sinh báo cáo
│   ├── run_e2e.py                     # CLI chạy dịch audio E2E
│   ├── s2s_web.py                     # Web interface chạy cục bộ
│   └── create_hybrid_tts_report.py    # Script tái sinh báo cáo PDF
└── web/                               # Giao diện web
```

---

## 6. Hướng dẫn sử dụng & Chạy thử nghiệm

### 6.1. Chạy Unit Tests kiểm thử định dạng TTS
Đảm bảo môi trường hoạt động chính xác bằng cách chạy 2 bộ kiểm thử độc lập:
```bash
# Kiểm tra hybrid text normalization (10 test suites)
PYTHONPATH=s2s_edge python3 -m unittest tests/test_hybrid_format.py

# Kiểm tra quy tắc bảo vệ thực thể identifiers (5 test suites)
PYTHONPATH=s2s_edge python3 -m unittest tests/tests_identifiers.py
```

### 6.2. Chạy dịch Audio End-to-End (CLI)
Để thực hiện dịch từ file âm thanh đầu vào ra file âm thanh đầu ra:
```bash
# Cú pháp: python3 tools/run_e2e.py <ngôn_ngữ_nguồn> <ngôn_ngữ_đích> [đường_dẫn_file_wav]

# Ví dụ 1: Dịch từ Tiếng Anh sang Tiếng Việt
python3 tools/run_e2e.py en vi path/to/input_en.wav

# Ví dụ 2: Dịch từ Tiếng Việt sang Tiếng Anh
python3 tools/run_e2e.py vi en path/to/input_vi.wav

# Ví dụ 3: Dịch từ Tiếng Việt sang Tiếng Hàn
python3 tools/run_e2e.py vi ko path/to/input_vi.wav
```
Kết quả audio đầu ra sẽ được lưu tại `results/e2e/audio/<src>_to_<tgt>.wav` kèm file JSON trace đầy đủ thông tin thời gian xử lý từng chặng.

### 6.3. Khởi chạy Giao diện Web Trực quan (Localhost)
Hệ thống tích hợp sẵn web server để test âm thanh trực tiếp:
```bash
python3 tools/s2s_web.py
```
Mở trình duyệt truy cập `http://localhost:8766` (hoặc cổng hiển thị trong log) để tải file âm thanh hoặc thu âm trực tiếp và quan sát kết quả dịch tức thì.

### 6.4. Tái tạo Báo cáo PDF Bất kỳ lúc nào
Nếu muốn cập nhật số liệu hoặc tạo lại file PDF báo cáo:
```bash
python3 tools/create_hybrid_tts_report.py
```
File PDF sẽ được xuất thẳng vào thư mục `reports/bao-cao-format-tts-hybrid.pdf`.

## Tích hợp asr-improvement (13/11/2026)

Đã fetch và tích hợp chọn lọc VAD, hậu xử lý dấu câu/viết hoa VI và ORT fallback từ commit `1e52318`; không thay weights ASR hoặc TTS hybrid. 22 nhóm test và 6 chiều S2S trước/sau đã chạy. Còn lỗi số liệu ASR/MT và tradeoff latency; xem [báo cáo tích hợp](reports/asr_integration/REVIEW.md). Với `S2S_NEURAL_TN=0`, đường VI là CAPU + vi_punct nâng cấp; chế độ neural nguồn cũ vẫn được giữ riêng.
