# -*- coding: utf-8 -*-
"""LAPTOP fallback when the web download does not work: pull every log and result from the box into ONE zip file.
  python tools/laptop_pull.py [--src /root/onevoice/s2s_edge_pkg/logs]       -> onevoice_box_logs_<date>.zip in the current folder"""
import argparse, os, shutil, subprocess, sys, tempfile, time
ADB = shutil.which("adb") or r"C:\platform-tools\adb.exe"
ap = argparse.ArgumentParser(); ap.add_argument("--src", default=""); a = ap.parse_args()
src = a.src or subprocess.run([ADB, "shell", "echo $HOME"], capture_output=True, text=True).stdout.strip() + "/onevoice/s2s_edge_pkg/logs"
tmp = tempfile.mkdtemp(); print("pulling", src)
p = subprocess.run([ADB, "pull", src, tmp], capture_output=True, text=True); print((p.stdout or p.stderr).strip().splitlines()[-1] if (p.stdout or p.stderr).strip() else "")
if p.returncode: sys.exit("adb pull failed")
out = shutil.make_archive(f"onevoice_box_logs_{time.strftime('%Y%m%d_%H%M%S')}", "zip", tmp); shutil.rmtree(tmp, ignore_errors=True); print("saved", os.path.abspath(out))
