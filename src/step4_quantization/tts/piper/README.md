# Piper (tiếng Việt) trên NPU Dragonwing IQ-9075 qua Qualcomm AI Hub

Giọng `vi_VN-vais1000-medium` (Piper/VITS). Mục tiêu: chữ → âm thanh, toàn bộ phần mạng chạy trên NPU.

## Kết quả (đo thật trên AI Hub, device "Dragonwing IQ-9075 EVK")
| Thử nghiệm | Kết quả |
|---|---|
| 5 graph riêng (encoder, sdp, aligner, flow, decoder) | 5/5 chạy, 100% layer trên NPU (403/619/19/310/117). Cosine từng tầng so fp32 ≥ 0.9999 trên cùng đầu vào |
| Chữ→phoneme trên NPU (`g2p_only`) | Giống hệt espeak ở 3/3 câu, 168/168 layer NPU, 12.5 ms |
| **Một graph duy nhất: byte UTF-8 → âm thanh** (`fullg8`) | Chạy 1 lần inference, **2077/2077 layer NPU**, 480 ms trên chip, ra audio 22.05 kHz |

Chưa kiểm: độ dễ hiểu bằng ASR/nghe thật (chỉ so sóng âm; cosine sóng thấp vì lệch 1–3 frame độ dài do fp16 ở SDP, không có nghĩa là sai).

## Cách hoạt động của graph gộp
- G2P bằng bảng tra 29.484 âm tiết (espeak-ng cho phoneme không phụ thuộc ngữ cảnh: 0 xung đột/2070 âm tiết), viết bằng matmul/so sánh tĩnh trên byte UTF-8.
- Trên 370 câu tiếng Việt thuần của FLEURS test: 361 câu khớp espeak 100%.
- Decoder gộp 39 cửa sổ một lần làm chip báo `MEM_ALLOC`; bản cuối gọi decoder theo nhóm 8 (`group=8`). Gọi từng cửa sổ một thì quá thời gian chuẩn bị graph.

## Giới hạn
- Từ ngoài bảng (tên riêng nước ngoài, số, ký hiệu) bị bỏ qua. Một câu ≤ ~40 âm tiết (254 phoneme), ≤ ~18.6 s, đầu ra 22.05 kHz.
- Máy chủ vẫn: đổi chuỗi thành mảng byte, cắt audio theo `y_len`, ghi wav. AI Hub chỉ chạy từng model, không chạy được code tuỳ ý; cần board thật nếu muốn bỏ hẳn.
- Decoder luôn tính đủ 40 cửa sổ dù câu ngắn.
- **Bẫy:** AI Hub trả đầu ra tên `output_N` theo thứ tự ONNX nhưng dict có thể xáo thứ tự; luôn sắp theo N.
- Code trong `src/step4_npu` lấy từ https://github.com/24122040-kin/text-to-speech. Nhận xét: `zero_cpu_driver.py` của họ chạy ONNX Runtime CPU (không phải NPU); `byte_text_encoder` không nối vào chuỗi thật.

## Cách chạy (từ thư mục `piper_npu/`)
```bash
pip install -r requirements.txt            # cần cấu hình qai-hub (API token)
cd g2p
python fetch_data.py                       # tải giọng, FLEURS-vi, vocab PhoBERT
python build_tables.py                     # dựng bảng tra + đo độ khớp espeak
python test_full_cpu.py                    # graph gộp (CPU) so với tham chiếu
python export_full.py 8                    # -> build/piper_vi_full_g8.onnx
python submit_compile.py fullg8 build/piper_vi_full_g8.onnx
python run_hw.py fullg8                    # profile + inference trên NPU
python eval_hw.py fullg8                   # so với fp32, ghi wav vào ../results/full_e2e_npu
```
Pipeline 5 graph cũ: `baseline_5graph/` (chạy từ `piper_npu/`; cần `outputs/piper_vi_npu/components/*.onnx` từ `src/step4_npu/export_piper_components.py`).

`results/` chứa wav NPU và fp32 để nghe.
