Vietnamese punctuation + capitalisation model for the ASR output (used by s2s_edge/punct_vi.py, S2S_VI_PUNCT=model).
Source: https://huggingface.co/dragonSwing/vibert-capu (viBERT-base token tagger), licence CC-BY-SA-4.0.
vibert_capu_int8.onnx : exported to ONNX (opset 17) and dynamically quantised to int8 (onnxruntime quantize_dynamic, per-channel), 111 MiB.
vocab.txt             : WordPiece vocabulary of FPTAI/vibert-base-cased ("$START" is appended at id 38168 by the code).
verb-form-vocab.txt   : the model's number-word -> digit table, copied unchanged.
The .onnx file is larger than GitHub's 100 MB file limit: it ships in the offline package / Drive, not in git.
