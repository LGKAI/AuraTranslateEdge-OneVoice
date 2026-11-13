# Draft answers for the Phase 3 "Prototype Submission" template

Status markers: **[have]** measured / written; **[device]** must be produced on the device (script given); **[team]** a decision or text only the team can provide.
Nothing below claims an on-device number that has not been measured.

## 1. Prototype summary

| Field | Answer |
|---|---|
| Team / project | [team] — project name, e.g. "AuraTranslate Edge" |
| Submitted build | s2s_edge 0.1.0, git branch `s2s` commit [team: final commit id], package date [team] |
| Demo languages | Vietnamese ⇄ English / Chinese / Korean, six directions, plus bidirectional `--pair` and `--auto` conversation modes [have: runs on CPU reference; device: to confirm] |
| Test scenario | [team] ~30 min factory floor conversation (supervisor ⇄ worker); script goes to `eval/` |
| Test conditions | [device] duration, background noise, microphone distance |
| Actual demo device | Dragonwing IQ-9075 (Hexagon v73, 36 GB LPDDR5 per the EVK spec) [device: OS version, microphone and speaker] |
| Capability & progress | Pipeline vi/en/zh/ko ASR → NLLB-200-600M (host beam search over NPU graphs) → Piper / Melo / Supertonic TTS; every graph compiled for the IQ-9075 and checked on a hosted IQ-9075 through AI Hub (`docs/chip_results.md`). Since Phase 2: single-graph TTS (Supertonic, MeloTTS, Piper), split NLLB with 64-token windows and 4-beam search, language identification and per-utterance routing, name handling. Main limit: on-device end-to-end run not yet measured [device]. |

## 2. Demo video [device]
Record `bash tools/session0.sh` output scrolling, then Wi-Fi off, then `python -m s2s_edge live --pair vi,ko --continuous`. Key timestamps [device].

## 3. Solution package
* Offline package: `s2s_edge_pkg/` (code, assets, `models/` = QNN context binaries built with QAIRT 2.51, `wheelhouse/` for Python 3.12 aarch64, `install.sh`, `run_demo.sh`) [have]. Download URL [team].
* Public repository: branch `s2s` of the team repository [team: push].
* Architecture: README pipeline diagram; ASR vi Zipformer-150M CTC (fp16), ASR en/zh/ko SenseVoice-Small (frontend fp16 + w8a16 encoder), MT NLLB-200-distilled-600M (encoder + 4-beam decoder step, fp16), TTS Piper vi, MeloTTS-ZH (static-table front + BERT + VITS in one graph), Supertonic-3 (one graph). Host CPU: audio I/O, VAD, fbank, tokenisation, beam bookkeeping, number normalisation.
* Runtime: QAI AppBuilder 2.51.40 (QAIRT 2.51) on Ubuntu 24.04 aarch64, Python 3.12. AI Hub / Device Cloud are development tools only; the demo runs without any network call.
* Minimum requirements [device: tested]: IQ-9075-class SoC with Hexagon v73 (models do not run on v68 parts such as QCS6490); RAM ≥ 8 GB [estimate: models ≈ 3.5 GB as context binaries; measure]; free storage ≈ 5 GB; microphone + speaker (ALSA).
* Install and run offline: `bash install.sh` → `bash tools/session0.sh` → `bash run_demo.sh vi en`.

## 4. Self-recorded test results (fill from `logs/`)
| Area | Result and method | Evidence |
|---|---|---|
| Translation quality | CPU reference: FLORES-200 devtest, gold source text, 25 sentences: vi→en chrF 57.0 / BLEU 28.3, en→vi 56.4 / 36.5, zh→vi 47.7 / 25.4 (beam 4). Speech → speech → ASR: `tools/eval_e2e.py` [have for CPU; device: repeat] | `logs/eval_mt_beam.json`, `logs/e2e2_*.json` |
| Inference time | AI Hub chip times per call: Zipformer 174 ms, NLLB encoder 7 ms, decoder step 16 ms (greedy) / 31 ms (4 beams), Melo 720 ms, Supertonic 748 ms, Piper 480 ms (`docs/chip_results.md`). End-to-end on device [device: `s2s_edge.bench`] | `logs/bench_*.json` |
| Model size | context binaries: Zipformer 351 MB, SenseVoice 9 + 272 MB, NLLB enc 640 + dec 927 + decB 927 MB, Piper 126 MB, Melo 704 MB, Supertonic 223 MB (loaded set per direction ≈ 2.4–3.1 GB) [have] | `s2s_pkg/models_q251/manifest.json` |
| Peak memory | [device] `bench` reports peak RSS | `logs/bench_*.json` |
| Device operation & UX | [device] | video timestamps |

**Known limitations:** see README "Known limits" and `docs/conversation.md` (names, code-switching, overlapping speakers).
**Next steps:** [team] up to three: (1) on-device latency tuning (batch / overlap ASR and TTS), (2) fine-tuned or bilingual ASR for code-switched factory speech, (3) denoising front end measured on real factory audio.
