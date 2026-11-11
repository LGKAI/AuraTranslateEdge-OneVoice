# Measured on the Dragonwing IQ-9075 EVK through Qualcomm AI Hub (development measurements)

These numbers come from AI Hub `profile` / `inference` jobs on a hosted IQ-9075 EVK: chip execution time per call, **excluding** queueing and model loading.
They are *not* on-device end-to-end measurements; those are produced by `python -m s2s_edge.bench` on the device (see README).

| Graph | Role | Job (compile / link QAIRT 2.51) | Layers on NPU | Chip time per call | Quality check |
|---|---|---|---|---|---|
| `asr_vi` Zipformer-150M CTC, fp16 | vi ASR, 15 s window | `jpv226qjg` / `jg9ood6wg`; profile `j57ooy99g`, infer `jp4eel31g` | 3092 / 3092 | **174.3 ms** | WER on 15 clips: clean 7.5 %, 5 dB 9.5 %, 0 dB 6.2 % |
| `asr_xx_*` SenseVoice-Small (frontend + w8a16 encoder) | en/zh/ko ASR, 29 s window | frontend `j57ez99vp`, encoder (inference model `jglynxxj5`) / `jp1oo6r75`, `jgd662jzp` | all NPU | frontend **62.5 ms** (`jp2oow1rg`), encoder **268.0 ms** (`jpy88xl8g`) per 29 s window | earlier 30-clip eval: en CER 4.8 %, zh CER 8.3 %, ko CER 7.3 % (CPU fp32 reference: 4.0 / 8.7 / 7.6 %) |
| `mt_enc` NLLB encoder, fp16, 32-token source (superseded) | MT | `jp4ee3yvg` / `jp1oo6r85`; profile `jp4ee3evg` | 463 / 463 | 7.1 ms | cosine 1.0 vs CPU |
| `mt_enc` NLLB encoder, 64-token source (**shipped**) | MT | `jgd663krp` / `j5q44y8og`; profile `jpy893z8g`, infer `jg9oom18g` | all NPU | **7.4 ms** | cosine ≥ 0.999 vs CPU, 8 sentences |
| `mt_dec` NLLB decoder step, 96 slots, greedy (**shipped, fast mode**) | MT | `j56oo7x65` / `j56oo7jy5`; profile `jgd6j1qzp`, infer `j5q44ykog` | all NPU | **16.1 ms per step** | 8 / 8 tokens equal to CPU |
| `mt_decB` NLLB decoder step, 4 beams (**shipped, default**) | MT | `j5wyjeqmg` / `jgd6jxz6p`; profile `jp4ezl92g`, infer `jp1or46n5` | 617 / 617 | **31.0 ms per step** | 32 / 32 argmax equal, logit cosine ≥ 0.99999 |
| `mt_dec` 48-slot version (superseded) | MT, one greedy step with static KV cache | `jpe66y71g` / `jgd662jrp`; profile `jgd6624rp`, infer `j5q44v1ng` | 667 / 667 | **14.8 ms per step** | 42 / 42 teacher-forced next tokens equal to CPU; cache outputs cosine 1.0 |
| `tts_vi` Piper vais1000 (byte text → audio) | vi TTS | `j5q46w3ng` / `jgodd87q5` | 2077 / 2077 | 480 ms (fixed, 40 windows) | not ASR-verified |
| `tts_zh` MeloTTS-ZH + static front + BERT | zh TTS | `jpy8mw17g` / `j5wyy4jmg`; profile `j5m9m9e9g`, infer `jgn1n10qp` | 3451 / 3451 | **720.2 ms** (fixed, 384 frames) | ASR CER 6.4 % on 40 texts (CPU: 6.0 %) |
| `tts_enko` Supertonic-3 fused | en/ko TTS | `jprxn9nep` / `jg9ood68g`; profile `jpe6l2l0g`, infer `jgzz4w46g` | 4476 / 4476 | 748.3 ms (fixed, 150 latent frames) | round trip WER 0.0 % (en), CER 1.5 % (ko), 6 sentences each |

## Same graphs re-linked for QCS8550 (HSPTEK AIBOX 8550), profiled on the hosted "QCS8550 (Proxy)" (09/11/2026)

`tools/relink_device.py` (link jobs in the package `models_qcs8550/link_jobs.json`, QAIRT 2.51) + `tools/profile_links.py`. All 9 context binaries load and run on QCS8550.

| Graph | QCS8550 chip time per call | IQ-9075 | Profile job |
|---|---|---|---|
| asr_vi | 163.4 ms | 174.3 ms | `jgl8z82ep` |
| asr_xx frontend / encoder | 58.3 / 236.5 ms | 62.5 / 268.0 ms | `jp37371xg` / `jpvmomr7p` |
| mt_enc | 7.5 ms | 7.4 ms | `jgjymy27g` |
| mt_dec / mt_decB per step | 15.2 / 30.8 ms | 16.1 / 31.0 ms | `jpex1xw7g` / `jp11l1wkp` |
| tts_vi / tts_zh / tts_enko | 520.1 / 740.6 / 701.2 ms | 480 / 720.2 / 748.3 ms | `j57nwnlqp` / `jp44o4dqp` / `jpxrjr6jp` |

## Estimated chip time per utterance (sum of the profiles above; an *estimate*, not an end-to-end measurement)

MT = encoder 7 ms + ~25 decoder steps: beam search 25 × 31 ≈ 0.8 s (default), greedy 25 × 16 ≈ 0.4 s (`models_fast.json`).

| Direction | ASR | MT (beam / greedy) | TTS | Sum (beam / greedy) |
|---|---|---|---|---|
| vi → en / ko | 174 ms | 0.78 / 0.41 s | ~750 ms per chunk | ~1.7 / ~1.3 s |
| vi → zh | 174 ms | 0.78 / 0.41 s | ~720 ms per chunk | ~1.7 / ~1.3 s |
| en / zh / ko → vi | 62 + 268 = 330 ms (SenseVoice, + 174 ms Zipformer when language identification runs) | 0.78 / 0.41 s | ~480 ms per chunk | ~1.6 / ~1.2 s (+ 0.17 s) |

Long sentences need several TTS chunks / translation pieces, so their time grows in steps.

The TTS graphs always compute their full static window, so their time is constant per chunk. Host work (fbank, SentencePiece, number
normalisation, I/O) comes on top and is measured by the benchmark harness.

## Translation quality (CPU reference, FLORES-200 devtest, gold source text; see `logs/eval_mt_split.json`)

`tools/eval_mt.py`, 25 FLORES sentences per direction, chrF / BLEU (greedy S=32 chunking → greedy S=64 → beam 4 S=64 → Hugging Face fp32 reference, greedy / beam 4):

| Direction | S=32 greedy | S=64 greedy | S=64 beam 4 | HF fp32 greedy | HF fp32 beam 4 |
|---|---|---|---|---|---|
| vi→en | 51.0 / 20.8 | 54.7 / 27.3 | **57.0 / 28.3** | 54.5 / 27.0 | 57.2 / 28.4 |
| en→vi | 54.2 / 32.5 | 55.9 / 35.5 | **56.4 / 36.5** | 55.8 / 35.6 | – |
| zh→vi | 40.4 / 15.3 | 46.5 / 23.7 | **47.7 / 25.4** | 46.8 / 24.1 | – |
| ko→vi | 45.3 / 23.1 | 46.1 / 24.9 | – | – | – |
| vi→zh / vi→ko | 17.0 / 18.3, 21.5 | 17.3 / 18.3, 22.2 | – | – | – |

The shipped graphs reproduce the Hugging Face model (beam 4: 57.0 vs 57.2 chrF on vi→en).

## Speech → speech → ASR (`tools/eval_e2e.py`, CPU reference, 8 FLEURS clips per direction, beam 4)

| Direction | ASR-chrF | ASR-BLEU | source ASR error | note |
|---|---|---|---|---|
| vi→en | 47.8 | 18.8 | WER 8.4 % | |
| en→vi | 46.9 | 21.0 | WER 11.3 % | |
| zh→vi | 36.3 | 12.5 | CER 13.0 % | |
| ko→vi | 43.7 | 14.6 | CER 7.1 % | |
| vi→zh | 16.2 | 17.4 | WER 8.4 % | chrF on Chinese characters is low by construction |
| vi→ko | 13.6 | 1.2 | WER 8.4 % | one reference only; spoken output vs its own MT text = 82.7 chrF, so the loss is translation, not TTS |

Bugs this evaluation found and fixed: the voice-activity threshold ignored quiet microphones (RMS 5e-4) and a loud file hid quiet speakers (now relative to the local level);
TTS audio was cut at the static window of Supertonic / Melo / Piper (now detected and the text re-split); runaway repetition in MT (n-gram blocking + length guard).

## Speech enhancement (GTCRN) — measured, off by default

Source WER at 5 dB SNR, 8 clips (`tools/eval_denoise.py`): vi clean 17.7 % → with GTCRN 17.3 %; vi babble 20.5 → 22.7; vi factory 18.7 → 19.1; en clean 11.7 → 13.1;
en babble 19.9 → 20.9; en factory 17.4 → 20.2. It does not help these two ASR models, so `--denoise` is optional and not used by default.

## Findings worth keeping

* The 16-step unrolled NLLB graph compiles to an **8.8 GB DLC** (the link job dedupes it to a 1.3 GB context binary) and can only produce 15 target tokens.
  The split encoder + decoder-step version is 0.64 + 0.93 GB as context binaries, has no 15-token limit and runs the greedy loop on the host.
* Context binaries are bound to the QAIRT version: AI Hub link jobs are pinned with `--qairt_version 2.51`, matching the `qai_appbuilder 2.51.40` wheel.
* Token ids above 2048 are not exact in fp16 on the HTP when compared as a single number; compare `(hi, lo)` halves instead (this is how the Melo front works).
  `int32` Cast / Mul / Add / Gather on ids up to 256 k are exact on the chip (384 / 384 in a probe).
