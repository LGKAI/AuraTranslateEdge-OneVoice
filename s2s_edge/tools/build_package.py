# -*- coding: utf-8 -*-
"""Assemble the offline installation package (a folder, zip it afterwards).
  python tools/build_package.py --models-dir <dir with <name>/<name>.bin + manifest.json> --wheelhouse <dir> --out dist/s2s_edge_pkg
Result: code + assets + tools + golden + samples + models/ (QNN context binaries + models.json) + wheelhouse/ + install.sh + run_demo.sh."""
import argparse, json, os, shutil, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# component -> manifest entry name
MAP = {"asr_vi": "asr_vi_zip150_ctc_fp16", "asr_xx_frontend": "asr_xx_sensevoice_frontend_v3", "asr_xx_encoder": "asr_xx_sensevoice_encoder_q",
       "mt_enc": "mt_enc_nllb_s64", "mt_dec": "mt_dec_nllb_s64", "mt_decB": "mt_decB_nllb_s64", "tts_vi": "tts_vi_piper_fullg8", "tts_zh": "tts_zh_melo_one", "tts_enko": "tts_enko_supertonic_full_v3"}

INSTALL = """#!/usr/bin/env bash
# Offline install (no network needed), aarch64 Linux with a Hexagon v73 NPU on HSPTEK AIBOX 8550 (QCS8550).
#   wheelhouse/        Python 3.12, glibc >= 2.39 (Ubuntu 24.04)
#   wheelhouse_cp310/  Python 3.10, glibc >= 2.35 (Ubuntu 22.04)
# Both carry qai_appbuilder 2.51.40 = the QAIRT 2.51 runtime the context binaries in models/ were linked for.
set -e
cd "$(dirname "$0")"
PYV=$(python3 -c 'import sys; print("%d%d" % sys.version_info[:2])')
GLIBC=$(ldd --version 2>/dev/null | head -1 | grep -oE '[0-9]+\\.[0-9]+$' || echo 0)
echo "python3 = $PYV, glibc = $GLIBC, arch = $(uname -m)"
case "$PYV" in
  312) WH=wheelhouse ;;
  310) WH=wheelhouse_cp310 ;;
  *)   echo "No wheelhouse for Python $PYV. Options: install python3.10/3.12 from the OS packages, or run in a container (tools/docker_run.sh)."; exit 2 ;;
esac
[ -d "$WH" ] || { echo "missing $WH/"; exit 2; }
python3 -m venv .venv || { echo "python3 -m venv failed: install the python3-venv package (offline: .deb from the OS image)"; exit 2; }
. .venv/bin/activate
pip install --no-index --find-links "$WH" -r requirements.txt openpyxl
# optional extras: missing wheels only disable a feature (python-crfsuite -> identifier CRF of the hybrid TTS normalizer)
for opt in python-crfsuite; do pip install --no-index --find-links "$WH" "$opt" || echo "optional $opt not in $WH: feature disabled"; done
pip install --no-index --find-links "$WH" qai_appbuilder || { echo "qai_appbuilder wheel rejected (glibc $GLIBC too old?). See tools/docker_run.sh"; exit 3; }
python3 -c "import qai_appbuilder, os; print('qai_appbuilder', os.path.dirname(qai_appbuilder.__file__))"
echo "Before running anything: . .venv/bin/activate; . tools/env.sh   (puts the bundled QAIRT 2.51 libs ahead of any QAIRT installed on the box)"
echo "Next: bash tools/session0.sh   (survey + golden check + smoke test + benchmark, logs packed as .tgz)"
"""

RUN = """#!/usr/bin/env bash
# usage: bash run_demo.sh <src> <tgt> [extra live-mode args]      e.g. S2S_MIC=plughw:1,0 S2S_SPK=plughw:1,0 bash run_demo.sh vi en --continuous
# Model set: S2S_MODELS, else S2S_MT=13b -> NLLB-1.3B set (models_qcs8550_13b), else the set tools/session0.sh validated, else models_qcs8550 (NLLB-600M).
cd "$(dirname "$0")"; . .venv/bin/activate; . tools/env.sh
case "${S2S_MT:-600m}" in 13b|1.3b|B) DEF=models_qcs8550_13b/models.json ;; *) DEF=$(cat logs/models_ok.txt 2>/dev/null || echo models_qcs8550/models.json) ;; esac   # A = NLLB-600M, B = NLLB-1.3B
M=${S2S_MODELS:-$DEF}
python -m s2s_edge live --src "${1:-vi}" --tgt "${2:-en}" --models "$M" --backend npu "${@:3}"
"""


def merge_asr(cfg, path, models_out):
    """Merge an ASR models.json fragment (bucket windows / RNN-T head / host SenseVoice features) into cfg, copying its files."""
    frag = json.load(open(path, encoding="utf-8")); base = os.path.dirname(os.path.abspath(path))
    asr = {k: v for k, v in frag.items() if k.startswith(("asr_", "_sv_"))}
    if any(k == "asr_vi" or k.startswith("asr_vi@") for k in asr): [cfg.pop(k) for k in list(cfg) if k == "asr_vi" or k.startswith("asr_vi@")]
    if any(k.startswith("asr_xx_encoder") for k in asr): [cfg.pop(k) for k in list(cfg) if k.startswith("asr_xx_encoder")]
    for k, v in asr.items():
        if not isinstance(v, dict): cfg[k] = v; continue
        v = dict(v)
        for f in ("ctx", "onnx"):
            if f in v:
                src = v[f] if os.path.isabs(v[f]) else os.path.join(base, v[f]); dst = os.path.join(models_out, os.path.basename(src))
                if not os.path.exists(dst): shutil.copy(src, dst)
                if os.path.exists(src + ".data") and not os.path.exists(dst + ".data"): shutil.copy(src + ".data", dst + ".data")
                v[f] = os.path.basename(src)
        v.setdefault("name", k.replace("@", "_")); cfg[k] = v
    print("asr:", sorted(asr))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--models-dir", required=True); ap.add_argument("--wheelhouse", default=""); ap.add_argument("--wheelhouse310", default="");ap.add_argument("--out", default="dist/s2s_edge_pkg"); ap.add_argument("--no-models", action="store_true"); ap.add_argument("--eval-dir", default=""); ap.add_argument("--eval-n", type=int, default=10)
    ap.add_argument("--asr-config", default="", help="models.json fragment of the ASR speed/accuracy work (ctx + graph_index entries printed by "
                    "tools/build_asr_buckets_aihub.py, asr_vi_decoder / asr_vi_joiner onnx, \"_sv_frontend\": \"host\"). Files are relative to it; "
                    "its keys replace the single-window ASR entries. See docs/asr_optimization.md.")
    a = ap.parse_args(); out = a.out
    if os.path.exists(out): shutil.rmtree(out)
    os.makedirs(out)
    for d in ("s2s_edge", "assets", "tools", "golden", "samples"):
        if os.path.isdir(os.path.join(HERE, d)): shutil.copytree(os.path.join(HERE, d), os.path.join(out, d), ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for f in ("README.md", "requirements.txt", "glossary_enterprise.json"): shutil.copy(os.path.join(HERE, f), out)
    for f, tmpl in (("install.sh", INSTALL), ("run_demo.sh", RUN)):          # the tracked scripts (tested on the box) win over the old templates
        src = os.path.join(HERE, f); txt = open(src, encoding="utf-8").read() if os.path.isfile(src) else tmpl
        open(os.path.join(out, f), "w", newline="\n").write(txt)
    for d in ("docs", "tests", "eval", "models_qcs8550"):                     # docs/tests, eval sets, NPU model-set configs (.bin files: --models-dir or tools/sync_package.py)
        if os.path.isdir(os.path.join(HERE, d)): shutil.copytree(os.path.join(HERE, d), os.path.join(out, d), dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.bin"))
    if a.wheelhouse and os.path.isdir(a.wheelhouse): shutil.copytree(a.wheelhouse, os.path.join(out, "wheelhouse"))
    if a.wheelhouse310 and os.path.isdir(a.wheelhouse310): shutil.copytree(a.wheelhouse310, os.path.join(out, "wheelhouse_cp310"))
    if a.eval_dir:                                              # small evaluation subset (FLEURS test audio + FLORES references) for judges
        os.makedirs(os.path.join(out, "eval"), exist_ok=True)
        for l in ("vi", "en", "zh", "ko"):
            man = json.load(open(os.path.join(a.eval_dir, f"manifest_{l}.json"), encoding="utf-8"))[: a.eval_n]; os.makedirs(os.path.join(out, "eval", l), exist_ok=True)
            for it in man: shutil.copy(os.path.join(a.eval_dir, it["path"]), os.path.join(out, "eval", l, os.path.basename(it["path"]))); it["path"] = f"{l}/" + os.path.basename(it["path"])
            json.dump(man, open(os.path.join(out, "eval", f"manifest_{l}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    if not a.no_models:
        man = json.load(open(os.path.join(a.models_dir, "manifest.json"))); cfg = {}; os.makedirs(os.path.join(out, "models"))
        for comp, name in MAP.items():
            if name not in man: print("MISSING", comp, name); continue
            src = os.path.join(a.models_dir, man[name]["file"]); dst = os.path.join("models", os.path.basename(src)); shutil.copy(src, os.path.join(out, dst))
            cfg[comp] = {"ctx": os.path.basename(src), "name": comp, "job": man[name]["job"]}
        cfg["mt_enc"]["S"] = 64; cfg["mt_dec"]["Tm"] = 96; cfg["mt_decB"].update(Tm=96, B=4)
        if a.asr_config: merge_asr(cfg, a.asr_config, os.path.join(out, "models"))
        fast = {k: v for k, v in cfg.items() if k != "mt_decB"}                      # greedy decoding (16 ms / step on chip) ; models.json = beam search (31 ms / step, +2.4 chrF)
        json.dump(cfg, open(os.path.join(out, "models", "models.json"), "w"), indent=1); json.dump(fast, open(os.path.join(out, "models", "models_fast.json"), "w"), indent=1)
        print("models:", {k: v.get("ctx", v.get("onnx")) for k, v in cfg.items() if isinstance(v, dict)})
    crf = os.path.join(os.path.dirname(HERE), "models", "normalization", "identifier.crfsuite")   # trained by tools/train_identifier_tagger.py
    if os.path.isfile(crf): shutil.copy(crf, os.path.join(out, "assets", "identifier.crfsuite")); print("assets: identifier.crfsuite")
    tot = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(out) for f in fs); print(f"package {out}: {tot / 1e9:.2f} GB")


if __name__ == "__main__": main()
