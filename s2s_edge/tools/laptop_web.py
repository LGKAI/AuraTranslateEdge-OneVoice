# -*- coding: utf-8 -*-
"""LAPTOP side of the web UI: forwards the port over the USB cable (adb) and opens the page served by the box.
  python tools/laptop_web.py            box web UI  (on the box: bash start.sh box,  port 8770)
  python tools/laptop_web.py --demo     web demo    (on the box: bash start.sh demo, port 8000)
The page, the models and the data are on the box; this laptop only displays it (and records the microphone for the mic test)."""
import argparse, shutil, subprocess, sys, time, urllib.request, webbrowser
ADB = shutil.which("adb") or r"C:\platform-tools\adb.exe"

ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=0); ap.add_argument("--demo", action="store_true"); a = ap.parse_args()
a.port = a.port or (8000 if a.demo else 8770); HEALTH = "/api/health" if a.demo else "/health"
try: print(subprocess.run([ADB, "forward", f"tcp:{a.port}", f"tcp:{a.port}"], capture_output=True, text=True, check=True).stdout or "adb forward ok")
except Exception as e: sys.exit(f"adb forward failed ({e!r}): is the box plugged in (adb devices)?")
for i in range(150):
    try: urllib.request.urlopen(f"http://127.0.0.1:{a.port}{HEALTH}", timeout=3).read(); break
    except Exception: print("waiting for the box server (" + ("bash start.sh demo" if a.demo else "bash start.sh box") + " on the box; the first start loads every model, 1-2 min)..."); time.sleep(2)
else: sys.exit("box server not reachable")
webbrowser.open(f"http://127.0.0.1:{a.port}/"); print(f"opened http://127.0.0.1:{a.port}/")
