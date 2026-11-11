# s2s_edge — offline speech-to-speech translation (vi ⇄ en / zh / ko) for Qualcomm Dragonwing IQ-9075

OneVoice AI Challenge 2026, Phase 3 prototype. Audio in → translated audio out, **all inference on the device, no network**.
Six directions: vi→en, vi→zh, vi→ko, en→vi, zh→vi, ko→vi (one side is always Vietnamese).

> **Status (be honest):** the pipeline code runs end to end on a CPU reference backend (ONNX Runtime). Every model graph has been compiled for
> the IQ-9075 NPU through Qualcomm AI Hub and measured there (see `docs/`), but the **NPU backend on a real device is not completed yet** — it is
> finished in the first device session (`tools/recon.sh` tells us which runtime is installed). Nothing in this README claims on-device numbers
> that have not been measured.

## Pipeline

```
mic / wav ──► energy VAD + segmentation (host, numpy) ──► ASR ──► NLLB-200-600M MT ──► number normalisation (host) ──► TTS ──► speaker / wav
                                                           │          │                                               │
        vi : Zipformer-150M CTC (1 graph, fp16)            │          encoder graph + decoder-step graph             vi : Piper vais1000 (1 graph, byte text → audio)
        en/zh/ko : SenseVoice-Small (frontend + encoder)   │          (host decode loop, KV cache as tensors)        zh : MeloTTS-ZH (1 graph incl. static-table text front + BERT)
                                                                                                                      en/ko : Supertonic-3 (1 graph incl. 5 ODE steps + vocoder)
```

Graphs run on the Hexagon NPU (`--backend npu`) or ONNX Runtime CPU (`--backend cpu`, reference / development). The host does only
cheap work: audio I/O, VAD, fbank, CTC collapse, SentencePiece, the NLLB greedy loop, number → words, wav joining.

| Component | Model | Graph(s) | Notes |
|---|---|---|---|
| ASR vi | Zipformer-150M CR-CTC-RNNT-6000h, CTC head | `asr_vi` | 15 s window, fp16 (WER on 15 clips: 7.5 / 9.5 / 6.2 % clean / 5 dB / 0 dB, measured on chip via AI Hub) |
| ASR en/zh/ko | SenseVoice-Small | `asr_xx_frontend`, `asr_xx_encoder` | 29 s window, w8a16 encoder |
| MT | NLLB-200-distilled-600M, 139k-token vocabulary | `mt_enc` (64-token source), `mt_decB` (4-beam decoder step) or `mt_dec` (greedy step) | host decode loop; beam search +0.5…2.4 chrF over greedy, 31 vs 16 ms per step on chip; `mt` = old 16-step unrolled graph (15-token limit) |
| TTS vi | Piper vi_VN-vais1000-medium | `tts_vi` | ≤ 26 words / chunk; out-of-vocabulary words are counted (`n_oov`) |
| TTS zh | MeloTTS-ZH (+ distilled g2p tables) | `tts_zh` | ≤ 20 characters / chunk |
| TTS en/ko | Supertonic-3 | `tts_enko` | ≤ 82 characters / chunk |

## Quick start (development machine, CPU reference)

```bash
pip install -r requirements.txt
cp models.dev.json models/models.json          # edit the paths to your ONNX files
python -m s2s_edge file --src vi --tgt en --in samples/vi_0.wav --out out.wav --backend cpu --models models/models.json
python -m s2s_edge live --src vi --tgt zh --backend cpu --models models/models.json     # press Enter, speak, hear the translation
```

## Multilingual conversations

```bash
python -m s2s_edge live --pair vi,ko --continuous --models models/models.json   # Vietnamese <-> Korean, language of each utterance detected, no key press
python -m s2s_edge live --auto --partner en --models models/models.json         # Vietnamese <-> whichever of en / zh / ko is spoken
```
Proper names (Latin names the Vietnamese / Chinese / Korean TTS cannot read), the glossary and the limits of mixed-language speech: `docs/conversation.md`.

## On the device

```bash
bash tools/session0.sh      # recon + install + golden check of every graph + smoke tests + quick benchmark, all logged and packed (do this first)
```

```bash
bash tools/recon.sh                                            # environment survey -> recon_<host>_<date>.txt
python tools/golden.py check --models models/models.json --backend npu      # every graph vs the CPU reference (seconds)
python -m s2s_edge live --src vi --tgt en --models models/models.json --backend npu
python -m s2s_edge.bench --src vi --tgt en --manifest eval/manifest_vi.json --runs 3 --logs logs/     # raw evidence for the benchmark section
```

`models/models.json` maps component names to files (`ctx` = QNN context binary produced by AI Hub link job, `onnx` = CPU reference).

## Files

| Path | Purpose |
|---|---|
| `s2s_edge/` | package: `audio` (I/O, VAD), `asr`, `mt`, `tts`, `textnorm`, `backends` (ONNX Runtime, QNN), `registry`, `pipeline`, `bench`, `__main__` (CLI) |
| `tools/recon.sh` | one-shot device survey (OS, SoC, QNN libs, audio devices, network) |
| `tools/session0.sh` | everything to run first in a device session, logs packed to a .tgz |
| `tools/golden.py` | make / check golden input-output pairs for every graph |
| `tools/eval_e2e.py` | speech → speech → ASR of the output vs FLORES reference (ASR-chrF / BLEU) for the six directions |
| `tools/lid_features.py` | features used to fit the vi / not-vi language identification |
| `tools/eval_mt.py` | FLORES-200 translation quality (chrF / BLEU) for all six directions |
| `assets/` | SentencePiece models, NLLB language ids, Supertonic unicode table + voice styles |
| `models.dev*.json` | development model maps (CPU ONNX paths) |

## Known limits

* Chinese TTS front is a static table: digits are spoken through `textnorm`, Latin letters and rare characters are dropped.
* Piper drops out-of-vocabulary words (counted in `n_oov`).
* NLLB source chunks are ≤ 30 SentencePiece tokens; longer sentences are split at punctuation / midpoint and translated piecewise.
* One utterance segment is ≤ 14.5 s (Zipformer window); longer audio is split at the quietest frame.
* The voice-activity detector is energy based and relative to the recording level (works with quiet microphones); it is not a neural VAD.
* Several speakers talking at once, and Vietnamese speech containing English words, are not handled (see `docs/conversation.md`).
* Models are compiled for Hexagon v73 (IQ-9075). They do **not** run on QCS6490 / RB3 Gen 2 (v68).
