# -*- coding: utf-8 -*-
"""Link the shipped graphs for HSPTEK AIBOX 8550 (Qualcomm QCS8550, Hexagon v73) with pinned QAIRT version.
  python tools/relink_device.py --device "QCS8550 (Proxy)" --qairt 2.51 --dry_run
  python tools/relink_device.py --device "QCS8550 (Proxy)" --qairt 2.51 --out models_qcs8550 [--dlc]
Job ids are the ones in docs/chip_results.md (shipped graphs)."""
import argparse, json, os, sys, time
import qai_hub as hub

SHIPPED = {                                   # package file name -> job that holds the compiled model
    "asr_vi_zip150_ctc_fp16": "jpv226qjg",
    "asr_xx_sensevoice_frontend_v3": "j57ez99vp",
    "asr_xx_sensevoice_encoder_q": "jglynxxj5",
    "mt_enc_nllb_s64": "jgd663krp",
    "mt_dec_nllb_s64": "j56oo7x65",
    "mt_decB_nllb_s64": "j5wyjeqmg",
    "tts_vi_piper_fullg8": "j5q46w3ng",
    "tts_zh_melo_one": "jpy8mw17g",
    "tts_enko_supertonic_full_v3": "jprxn9nep",
}


def compiled_model(job_id):
    j = hub.get_job(job_id)
    m = j.get_target_model() if isinstance(j, hub.CompileJob) else j.model
    return j, m


ap = argparse.ArgumentParser(); ap.add_argument("--device", default="QCS8550 (Proxy)"); ap.add_argument("--qairt", default="2.51")
ap.add_argument("--out", default="models_relinked"); ap.add_argument("--only", default=""); ap.add_argument("--dlc", action="store_true"); ap.add_argument("--dry_run", action="store_true")
a = ap.parse_args(); sys.stdout.reconfigure(encoding="utf-8"); dev = hub.Device(a.device); os.makedirs(a.out, exist_ok=True)
names = [n for n in SHIPPED if not a.only or n in a.only.split(",")]; links = {}
for n in names:
    j, m = compiled_model(SHIPPED[n]); print(f"{n:32s} {type(j).__name__:14s} {SHIPPED[n]}  model={m.model_id if m else None}", flush=True)
    if a.dry_run or m is None: continue
    if a.dlc: m.download(os.path.join(a.out, n + ".dlc"))
    links[n] = hub.submit_link_job(models=[m], device=dev, options=f"--qairt_version {a.qairt}", name=f"s2s_{a.qairt}_{n}")
    print("   link job", links[n].job_id, flush=True)
json.dump({n: l.job_id for n, l in links.items()}, open(os.path.join(a.out, "link_jobs.json"), "w"), indent=1)
for n, l in links.items():
    st = l.wait()
    if not st.success: print("FAILED", n, st.message); continue
    l.get_target_model().download(os.path.join(a.out, n + ".bin")); print("ok", n, flush=True)
