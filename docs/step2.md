# Step 2 — MT (Machine Translation, Text → Text)

**Status (2026-08-09):** Code-tested thật (NLLB-600M vs Qwen3-0.6B vs Qwen3-1.7B trên FLORES-200, 6 chiều) + research full-text 5/7 paper cho streaming policy, low-resource, code-switch. Kiến trúc **CHỐT**: NLLB-200-distilled-600M + AlignAtt.

---

## Part A — Drop-in cho Technical Proposal §4.2 "Module-by-Module Design"

| Module | Model / Framework | Size (est.) | Latency Target | Key Technique |
|---|---|---|---|---|
| NMT (core) | NLLB-200-distilled-600M | ~600M params (~1.2GB fp16, int8 quantizable to ~600MB) | RTF <0.1 non-streaming (measured: 0.4–0.9s/sentence on FLORES-200) | Fine-tune per pair; self-converted ONNX→QNN (not on Qualcomm AI Hub) |
| Streaming policy (MVP) | AlignAtt (Papi et al., 2023) | 0 extra params | AL ≈ 2s (reference number from source paper) | Cross-attention threshold, zero retraining, applied directly on NLLB |
| Streaming policy (upgrade path) | AliBaStr-style read/write policy network | +~1–3M params | 32–37% latency reduction vs non-streaming (reference: comparable on-device system, Meta AI arXiv 2508.13358) | Supervised policy net trained on attention pseudo-labels |
| Low-resource boost (Vi↔Zh, Vi↔Ko) | mBART-50 fine-tune + domain-filtered back-translation | — | — | TF-IDF monolingual selection + back-translation (NOT naive BT) |
| Code-switch handling | Copy-Through augmentation (identity pairs in fine-tune data) | — | — | Cheaper than full LLM-based augmentation pipeline |
| Synthetic-data teacher (offline only, dev machine) | Hunyuan-MT-7B (WMT2025 #1 in 30/31 pairs) | 7B, never deployed to device | — | Generates higher-quality synthetic Vi↔Zh/Ko pairs than raw back-translation |

**⚠️ Note for Hardware section (§5):** NLLB-600M is **not** listed on Qualcomm AI Hub (unlike Qwen3-0.6B/1.7B, which are pre-quantized for Snapdragon) — this was tested head-to-head (see Part B §1) and NLLB still wins decisively on quality and speed. Plan: convert ONNX→QNN ourselves; flag this as a known extra engineering task in the timeline (§7). **Also note: all BLEU/speed numbers above were measured on the dev machine's NVIDIA GPU (CUDA), not on real Snapdragon hardware** — same caveat applies to step1.md's ASR numbers. Real on-device latency is still unmeasured; plan to use Qualcomm AI Hub's free remote-profiling service (`qai-hub` package) once the ONNX→QNN conversion is done.

**Quality target:** streaming BLEU within ~2–3 points of non-streaming baseline (reference gap from AliBaStr-MT's own measurement: 43.89 vs 45.56 BLEU on a comparable ES pair).

---

## Part B — Phân tích đầy đủ, có số liệu chọn/loại từng candidate (Vietnamese)

### 1. Chọn model dịch: NLLB-600M vs Qwen3-0.6B vs Qwen3-1.7B (code-test thật, FLORES-200, 30 câu × 6 chiều)

**Vì sao test lại từ đầu**: NLLB-600M ban đầu chỉ thừa kế từ kế hoạch gốc, chưa từng đo thật. Đề bài khuyến khích dùng model có sẵn trên Qualcomm AI Hub — **NLLB không có ở đó, Qwen3-0.6B/1.7B thì có** (đã quantize sẵn cho Snapdragon). Cần số thật để biết lợi thế "có sẵn" có đáng đánh đổi chất lượng không.

| Chiều | NLLB-600M | Qwen3-0.6B | Qwen3-1.7B |
|---|---|---|---|
| vi→en | **33.81** | 18.44 | 26.85 |
| en→vi | **29.67** | 16.19 | 26.01 |
| vi→zh | 20.45 | 17.74 | **23.41** |
| zh→vi | **21.07** | 10.39 | 17.58 |
| vi→ko | **8.05** | 3.18 | 5.75 |
| ko→vi | **18.60** | 5.63 | 13.97 |
| Tốc độ (giây/câu) | **0.4–0.9** | 2.7–6.9 | 1.6–10.1 |

**✅ CHỌN: NLLB-200-distilled-600M.** Thắng 5/6 chiều và thắng tốc độ ở **mọi** chiều (nhanh hơn Qwen3 3–15 lần tuỳ chiều). Dù không có sẵn trên AI Hub (phải tự convert ONNX→QNN, tốn thêm công), số liệu thật cho thấy đây vẫn là lựa chọn đúng — lợi thế "pre-optimized" không bù được khoảng cách chất lượng/tốc độ.

**❌ LOẠI: Qwen3-0.6B.** Cùng cỡ tham số với NLLB, có sẵn trên AI Hub, nhưng **thua rất xa ở mọi chiều** (VD vi→en: 18.44 vs 33.81 — kém gần 2 lần) và chậm hơn 3–8 lần. Không có lý do chọn ngoài việc "có sẵn trên AI Hub" — không đủ bù chất lượng.

**❌ LOẠI: Qwen3-1.7B làm nhánh chính.** Chỉ thắng đúng 1/6 chiều (vi→zh: 23.41 vs 20.45) — đáng chú ý nhưng không đủ để chọn làm model chính khi thua 5/6 chiều còn lại và chậm nhất trong 3 candidate (có chiều lên tới 10.1s/câu). **Giữ lại làm ứng viên phụ cho riêng chiều vi→zh nếu cần tối ưu sâu hơn sau này**, hoặc cho "accurate mode" không streaming.

**Bài học rút ra**: việc AI Hub liệt kê sẵn không đồng nghĩa là lựa chọn tốt nhất cho MỌI module — cần đo thật từng trường hợp, không suy đoán theo "có sẵn = nên dùng".

### 2. Streaming policy: AlignAtt (MVP) — có phương án nâng cấp AliBaStr-MT

Đề bài liệt kê wait-k / AlignAtt / AliBaStr — đọc full-text phát hiện đây là **2 paper riêng biệt dùng tên gần giống nhau**, dễ nhầm:

| | **AlignAtt** (SimulSeamless, IWSLT 2024) | **AliBaStr-MT** (Meta AI, on-device, 2025) |
|---|---|---|
| Cơ chế | Không cần train lại — dùng cross-attention model offline có sẵn, dừng phát từ nếu "nhìn" vào f khung gần nhất | Cần train thêm policy network nhỏ, học từ pseudo-label rút từ attention model offline |
| Model gốc bài báo dùng | SeamlessM4T-medium — 1.2B tham số, quá nặng cho edge | Encoder/decoder riêng — chỉ ~103M, đúng tầm on-device |
| Hyperparameter | `f` (số khung gần nhất) — chỉnh riêng theo từng cặp ngôn ngữ | `δ` (ngưỡng suy luận) — chỉnh lúc infer, không cần train lại |
| Số đo thật từ paper gốc | en-zh: BLEU 20.56, AL 1.94s (paper tự nhận "struggles" ở cặp này — đúng 1 cặp OneVoice cần!) | ES: BLEU 43.89/45.30 vs non-streaming 45.56/46.60 — chỉ thua ~1.5 điểm; AL giảm 4.39s→2.74s (**37% nhanh hơn**) |

**✅ CHỌN cho MVP: AlignAtt trên NLLB-600M.** Lý do: zero retraining, triển khai nhanh, không cần data huấn luyện chính sách — đúng ưu tiên giai đoạn đầu.

**Ghi nhận nâng cấp: AliBaStr-MT phù hợp hơn về kiến trúc** (model ~103M đúng tầm edge thay vì 1.2B, tự báo cáo trên kịch bản on-device thật chứ không phải academic benchmark, latency đo được thấp nhất trong mọi phương án so sánh kể cả wait-k/EMMA) nhưng cần train thêm policy network nhỏ — **không chọn cho MVP vì tốn thêm công train, để làm bước 2 nếu còn thời gian.**

### 3. Vi-Zh low-resource: SỬA claim "beat Google Translate" (phát hiện lật ngược quan trọng)

Đọc full-text arXiv 2501.19314 (Samsung SDS, VLSP 2022), bảng số thật:

| Chiều | Google Translate | UET Engine | mBART-50 + back-translation (paper) |
|---|---|---|---|
| Vi→Zh | **41.6** | 39.2 | 38.97 |
| Zh→Vi | 38.1 | **40.6** | 38.90 |

Paper tự kết luận: *"Google Translate is the best for Vietnamese-to-Chinese translation while the UET engine is the best for Chinese-to-Vietnamese translation"* — **không thắng ở cả 2 chiều**, chỉ nhỉnh hơn Google Translate 0.8 điểm ở MỘT chiều.

**Điều back-translation thực sự làm được**: nâng from-scratch baseline lên đáng kể — Zh→Vi +3.19 BLEU (35.58→38.90), Vi→Zh +0.75 BLEU (khiêm tốn hơn). Kỹ thuật tốt để cải thiện hệ thống yếu, **không phải công thức thắng Google Translate**.

**✅ Áp dụng cho OneVoice**: giữ back-translation domain-filtered (TF-IDF chọn câu monolingual liên quan domain) làm kỹ thuật cải thiện Vi↔Zh/Ko, nhưng **không đặt "moat" (lợi thế cạnh tranh) vào việc thắng Google Translate về BLEU tổng quát** — dữ liệu thật không ủng hộ. Lợi thế thật: (1) hoạt động offline (Google Translate cần mạng), (2) chuyên biệt thuật ngữ nhà máy/an toàn lao động (Google Translate không có lợi thế ở từ vựng hẹp). DoD nên đặt "đúng thuật ngữ chuyên ngành > 95%" làm tiêu chí chính, không phải "BLEU ≥ Google Translate".

### 4. Code-switching: back-translation thất bại, dùng Copy-Through

VietMix (arXiv 2505.24472, UMD) đo trực tiếp: back-translation trên text code-switch Việt-Anh chỉ cải thiện **+0.55 xCOMET so với zero-shot** (77.72→78.27) — gần vô dụng, vì model có xu hướng "sửa" code-switch về câu đơn ngữ sạch, xoá mất đúng hiện tượng cần giữ.

**❌ LOẠI: pipeline VietMix đầy đủ** (LLM 7-9B + lọc nhiều tầng) — hiệu quả nhưng quá nặng cho NLLB-600M edge.

**✅ CHỌN: kỹ thuật "Copy-Through" từ paper AliBaStr-MT** — thêm cặp dữ liệu "giữ nguyên" (En→En, X→X) vào tập fine-tune, dạy model không dịch phần đã đúng ngôn ngữ đích. Nhẹ hơn nhiều, không cần LLM lớn sinh data.

**Dùng VietMix để làm gì**: chỉ mượn bộ test 1,002 câu gold-standard (Vi→En, license CC BY-NC-SA — đủ điều kiện cho giai đoạn thi phi thương mại) làm **benchmark nội bộ** đánh giá khả năng xử lý code-switch, không replicate pipeline nặng của họ.

### 5. Hunyuan-MT-7B làm teacher sinh dữ liệu — khả thi, chỉ chạy offline

Đọc full-text arXiv 2509.05209 (Tencent): #1 WMT2025 trong 30/31 cặp, hỗ trợ tiếng Việt (trong danh sách 33 ngôn ngữ). Không có số Vi-Zh/Vi-Ko riêng (chỉ báo cáo nhóm ZH↔XX/EN↔XX gộp), nhưng khoảng cách với Google Translate ở nhóm này rất lớn (ZH⇒XX: 0.876 vs Google 0.762 XCOMET-XXL) — đủ tin cậy làm **teacher sinh synthetic data Vi↔Zh/Ko chất lượng cao hơn back-translation thô**. Model 7B chỉ chạy trên máy dev để sinh data, không deploy lên thiết bị — không ảnh hưởng ngân sách runtime edge.

### 6. vi→ko BLEU thấp bất thường (8.05 vs ko→vi 18.60) — soi mẫu dịch thật để chẩn đoán

Kiểm tra 5 câu vi→ko thật của NLLB: **phần lớn ngữ nghĩa đúng** (1 lỗi thật: "tuần lộc"/reindeer bị dịch nhầm thành "오징어"/mực). BLEU thấp chủ yếu do **tiếng Hàn là ngôn ngữ chắp dính** — cùng nghĩa nhưng đuôi từ/trợ từ khác nhau (VD quá khứ "않았습니다" vs hiện tại "않습니다", đều đúng ngữ pháp) khiến BLEU tính n-gram bề mặt bị đánh giá thấp giả tạo. Giải thích khớp với việc ko→vi (đích là tiếng Việt, không chắp dính) không bị ảnh hưởng. Khác bug CER khoảng trắng tiếng Trung ở Step 1 (sửa được bằng chuẩn hoá text), lỗi này cần metric ngữ nghĩa (COMET/xCOMET) thay vì BLEU bề mặt mới đánh giá công bằng — **chưa nên kết luận NLLB "kém 2 lần" ở vi↔ko chỉ dựa vào BLEU thô**.

### 7. Bug môi trường gặp khi code-test (đáng nhớ)

- **Nguồn FLORES-200**: `Muennighoff/flores200` (HF Datasets) dùng loading script không còn được `datasets` hỗ trợ; `facebook/flores` và `openlanguagedata/flores_plus` đều gated (cần đăng nhập HF). Giải pháp: tải thẳng tarball chính thức Meta (`dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz`), không qua thư viện `datasets`.
- **`test_mt_qwen3.py` chạy nền tưởng như "treo"** (>30 phút không tiến triển dù GPU bận thật) — nguyên nhân: dùng nhầm tham số `torch_dtype` đã deprecated (transformers dev `5.15.0.dev0`) thay vì `dtype`. Sửa xong + thêm progress log theo từng câu.

### 8. Kiến trúc tổng hợp cuối

- **Live mode**: NLLB-600M fine-tune per pair + AlignAtt.
- **Vi↔En**: fine-tune trên PhoMT (chất lượng cao sẵn, không cần back-translation).
- **Vi↔Zh, Vi↔Ko**: fine-tune + back-translation domain-filtered (TF-IDF) + tuỳ chọn bổ sung synthetic data từ Hunyuan-MT-7B teacher. Kỳ vọng: cải thiện rõ so với from-scratch, KHÔNG kỳ vọng vượt Google Translate.
- **Code-switch**: Copy-Through augmentation, đánh giá bằng VietMix test set.
- **Nâng cấp nếu còn thời gian**: policy network kiểu AliBaStr-MT thay AlignAtt; Qwen3-1.7B riêng cho vi→zh nếu cần.

### 9. Kế hoạch test tiếp theo

1. Fine-tune NLLB-600M riêng từng cặp: Vi-En (PhoMT), Vi-Zh (VLSP 2022 + back-translation theo công thức đã kiểm chứng ở §3), Vi-Ko (AI Hub Korea + back-translation/pivot).
2. So trước/sau fine-tune trên FLORES-200 — kỳ vọng cải thiện rõ so với baseline hiện tại (§1), không kỳ vọng thắng Google Translate.
3. Đo lại vi↔ko bằng COMET/xCOMET, không chỉ BLEU, để tách bạch "NLLB yếu thật" khỏi "BLEU đánh giá sai do đặc tính tiếng Hàn" (§6).
4. Test streaming thật: SimulEval đo AL/LAAL, quét δ (AlignAtt), đối chiếu con số 32–37% latency reduction tham khảo từ AliBaStr-MT.
5. Test thuật ngữ an toàn lao động — DoD chính, không phải BLEU tổng quát.
6. Test code-switch bằng VietMix test set (1,002 câu Vi→En) sau khi áp dụng Copy-Through.

---

**Document version:** 2026-08-09 — code-tested NLLB-600M vs Qwen3-0.6B/1.7B trên FLORES-200 thật (6 chiều), kiến trúc chốt, Part A khớp format §4.2 Technical Proposal chính thức.
**Bước tiếp theo:** Fine-tune NLLB theo từng cặp; đo lại vi↔ko bằng COMET/xCOMET; xác nhận kế hoạch tự quantize ONNX→QNN trong Hardware section (§5).
