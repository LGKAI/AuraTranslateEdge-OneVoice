# README_EVAL

- Team name: Gia Su Dinh Cao (`DONE` team token: `GiaSuDinhCao`)
- Registered language pair: `vi-en`, `vi-ko`, `vi-zh` (all 6 directions: `vi2en`, `en2vi`, `vi2ko`, `ko2vi`, `vi2zh`, `zh2vi`)
- Packaging format: `cli`
- Device and operating system used in the demo video: HSPTEK AIBOX 8550 (Qualcomm Snapdragon QCS8550, Hexagon NPU v73, 16 GB LPDDR5x RAM, 128 GB UFS) / Ubuntu 22.04 LTS (`aarch64`), QAIRT 2.51 (`qai_appbuilder 2.51.40`)
- Total time to process the full observe dataset (minutes), for judging schedule planning:
  - Default demo routing (`models_qcs8550/models.json`: NLLB-600M for `vi<->en`, NLLB-3.3B for `vi<->ko` & `vi<->zh`): ~140–180 minutes for all 658 clips across 6 directions (~18–20 min for `vi<->en` 200 clips; ~60–80 min per 360-clip `vi<->ko` or `vi<->zh` pair, based on measured on-device turn latency).
  - Optional NLLB-600M-only mode (`S2S_MT=600m`): ~55–65 minutes for all 658 clips across 6 directions (measured on-device mean 3.5–7.4 s/turn).

## For apps only
- Package ID or Bundle ID: N/A
- Main activity name (Android): N/A
- Is models.zip included? If so, which path should it be copied to: N/A
- Is input.zip supported (iOS): N/A
- Xcode and iOS versions used to build and test (iOS): N/A
- Have you received the judging iPhone's UDID (iOS): N/A

## For containers only
- Image name: N/A
- Architecture: N/A
- User running the process: N/A
- Additional docker run arguments required: N/A
- Image compression format: N/A

## For CLI packages only
- Required device and operating system: HSPTEK AIBOX 8550 (Qualcomm QCS8550, Hexagon NPU v73, 16 GB RAM), Ubuntu 22.04 LTS `aarch64`, Python 3.10, with `/dev/adsprpc-smd` NPU access.
- Is setup required, and does it require administrator privileges: Yes. Run `./setup.sh` once inside `s2s_edge_pkg/` to create `./.venv` completely offline from `wheelhouse_cp310/` (also runs automatically on the first `./run_eval.sh` invocation if `./.venv` does not exist). Root or `system` user privileges are required on the AIBOX 8550 to access `/dev/adsprpc-smd`; when invoked as `root`, both `setup.sh` and `run_eval.sh` automatically re-exec as user `system` (matching the submitted `start.sh`). Do **not** place the package under `/root` because user `system` cannot traverse `/root` (use `/tmp/onevoice/s2s_edge_pkg` or `/opt/onevoice/s2s_edge_pkg`).

## Additional notes

### 1. How to assemble & run Eval Mode offline on HSPTEK AIBOX 8550
1. Download and extract the 3 submitted offline package parts into a single `s2s_edge_pkg/` folder (exact submitted Phase 3 build with all 13 QNN context binaries, `assets/`, and `wheelhouse_cp310/`):
   - `s2s_edge_pkg_AIBOX8550_c4fad20.part1.zip`
   - `s2s_edge_pkg_AIBOX8550_c4fad20.part2.zip`
   - `s2s_edge_pkg_AIBOX8550_c4fad20.part3.zip`
2. Overlay the eval-mode scripts from branch `eval-mode` (`https://github.com/Khanhhh239/AuraTranslateEdge/tree/eval-mode`) into `s2s_edge_pkg/` (`s2s_edge/eval_mode.py`, `s2s_edge/__main__.py`, `setup.sh`, `run_eval.sh`, `README_EVAL.md`). **Zero model files or pipeline weights/logic are modified.**
3. On the AIBOX 8550 (network disabled), run:
   ```bash
   chmod +x setup.sh run_eval.sh
   ./setup.sh
   ./run_eval.sh "$PWD/input" "$PWD/output"
   echo "exit code: $?"
   ```

### 2. Compliance with Phase 3 Eval Mode rules
- **Exact demo pipeline & models:** Uses the exact submitted `s2s_edge` pipeline (`Translator(reg, src, tgt).translate(wav)` with `models_qcs8550/models.json`, `vi_capu`, and `ko_punct`) and all 13 QNN `.bin` files unchanged.
- **Offline & zero context carry-over:** 100% offline; each manifest row is translated independently with no cross-sentence state.
- **Model pre-loading & 60s timeout watchdog:** All required graphs are loaded and warmed up once in a persistent worker subprocess before the loop. Each sentence has a strict 60-second watchdog (`OVAC_TIMEOUT_S=60`); any hung sentence is terminated, recorded as `timeout` (with no WAV file), and the worker is respawned cleanly for subsequent rows.
- **Output format & audio normalization:** Output WAVs are PCM 16-bit mono (`>= 16 kHz`: 22.05 kHz VI, 44.1 kHz EN/KO/ZH), normalized toward `-23 LUFS` with a strict `-1 dBFS` peak ceiling and zero clipping. `result.csv` is UTF-8 without BOM, LF line endings, single-line cells, `message <= 200` chars, flushed + `fsync()` after every row.
- **Resume safety:** Checks `output/STARTED`; clears only contents inside `output/` on a fresh run (never deleting `output/` itself), and resumes from the first missing `id` on restart while appending to `output/log.txt`.

### 3. Self-test verification note
- Automated unit & integration tests (`python s2s_edge/tests/test_eval_mode.py`) verified every checklist requirement (fresh run, `STARTED`/resume with truncated CSV recovery, corrupt WAV -> `error`, 60s subprocess timeout kill+respawn, `empty`/`unsupported`, `-1 dBFS` peak ceiling + loudness boost, CSV UTF-8 no-BOM LF, paths with spaces, `DONE` single-line format).
- Because the HSPTEK AIBOX 8550 hardware is housed at VNG's lab and is not physically accessible at home before the 28/11 deadline, `selftest_output/` could not be freshly re-generated on the physical QCS8550 NPU after adding `eval_mode.py`. Existing real on-device NPU batch outputs on the observe dataset (658 clips across all 6 directions) are included in `Evidence_Folder/onevoice_btc_half1_20261112_1610.zip` and `onevoice_btc_half2_20261112_1610_partial.zip`.
