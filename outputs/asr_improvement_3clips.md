# Kết quả Kiểm thử Cải tiến ASR Tiếng Việt (`s2s_edge` CPU Reference)

Kiểm tra 3 file âm thanh gốc tiếng Việt trong `data/phase3_sample_dataset/` qua pipeline `s2s_edge`: `segment()` (VAD) → `AsrVi` (Zipformer-150M CTC) → `restore_vi()` (`vi_punct` nâng cấp) → `split_sentences()` → `Nllb` (`nllb_S32_T16.onnx`).

| Mã clip | Thời lượng | Đoạn VAD | Gold Tiếng Việt | ASR Raw (Zipformer) | ASR Sau Khôi phục & Ngắt câu | Bản dịch NLLB Đích | Bản dịch NLLB Tham chiếu EN |
|---|---:|:---:|---|---|---|---|---|
| `VE-0371` | 3.4s | 1 | Hạn cuối là thứ Sáu đúng không? | `hẹn cuối là thứ sáu đúng không` | **Hẹn cuối là thứ Sáu đúng không?** | (EN) The last one's Friday, right? | The last one's Friday, right? |
| `VK-0755` | 2.74s | 2 | Tôi rất hài lòng. Bao nhiêu tiền? | `tôi rất hài lòng bao nhiêu tiền` | **Tôi rất hài lòng. Bao nhiêu tiền?** | (KO) 정말 만족합니다 얼마나 많은 돈을? | I'm very pleased. How much money? |
| `VM-4870` | 7.04s | 3 | Ngày mai là Chủ nhật, người đi xe chắc chắn sẽ rất đông, chúng ta đi sớm chút đi. Bẩy rưỡi đi thế nào? | `ngày mai là chủ nhật người đi xe chắc chắn sẽ rất đông chúng ta đi sớm chút đi bảy rưỡi đi thế nào` | **Ngày mai là Chủ nhật người đi xe chắc chắn sẽ rất đông. Chúng ta đi sớm chút đi. Bảy rưỡi đi thế nào?** | (ZH) 明天是星期日,车辆肯定会很大. 我们要早点走. 七半怎么样? | Tomorrow is Sunday. It's definitely going to be a busy drive. Let's go a little early. How's the seven and a half? |
