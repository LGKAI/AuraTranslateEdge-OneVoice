# -*- coding: utf-8 -*-
"""Check the LAPTOP side before going to VNG: adb, python audio libraries, microphone, loudspeaker, the package folder and its checksums.
  python tools/laptop_check.py [--pkg D:\\onevoice_deploy]
Prints OK / FIX lines; every FIX line says what to do."""
import argparse, hashlib, os, shutil, subprocess, sys

ok = lambda m: print("OK  ", m)
fix = lambda m: print("FIX ", m)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pkg", default=""); a = ap.parse_args(); sys.stdout.reconfigure(encoding="utf-8")
    adb = shutil.which("adb")
    if adb:
        ok(f"adb = {adb}"); out = subprocess.run(["adb", "devices"], capture_output=True, text=True).stdout.strip().splitlines()[1:]
        (ok if any("\tdevice" in l for l in out) else fix)("adb devices: " + (", ".join(out) if out else "no device (plug in the USB-C cable; Google USB Driver in Device Manager if it shows a yellow mark)"))
    else: fix("adb not found: download Android platform-tools, unzip to C:\\platform-tools and add it to PATH (docs/HUONG_DAN.md, section 2)")
    for m in ("numpy", "soundfile", "sounddevice"):
        try: __import__(m); ok(f"python module {m}")
        except Exception: fix(f"pip install {m}")
    try:
        import sounddevice as sd
        d = sd.query_devices(); ins = [i for i, x in enumerate(d) if x["max_input_channels"] > 0]; outs = [i for i, x in enumerate(d) if x["max_output_channels"] > 0]
        (ok if ins else fix)(f"microphones: {[d[i]['name'] for i in ins][:4]} (default input {sd.default.device[0]})"); (ok if outs else fix)(f"loudspeakers: {len(outs)}")
        if ins:
            import numpy as np; x = sd.rec(16000, samplerate=16000, channels=1, dtype="int16"); sd.wait(); lv = int(np.abs(x).max())
            (ok if lv > 300 else fix)(f"1 s of microphone, peak level {lv}" + ("" if lv > 300 else " (silent: Windows Settings > Privacy > Microphone, allow desktop apps; pick the right default device)"))
    except Exception as e: fix(f"sounddevice failed: {e!r}")
    if a.pkg:
        sums = os.path.join(os.path.dirname(os.path.abspath(a.pkg)), "SHA256SUMS"); (ok if os.path.isdir(a.pkg) else fix)(f"package folder {a.pkg}")
        if os.path.exists(sums) and os.path.isdir(a.pkg):
            bad = 0; n = 0
            for line in open(sums, encoding="utf-8"):
                h, p = line.rstrip("\n").split("  ", 1); f = os.path.join(a.pkg, p)
                n += 1
                if not os.path.exists(f): bad += 1; print("   missing", p); continue
                s = hashlib.sha256()
                with open(f, "rb") as fh:
                    for blk in iter(lambda: fh.read(1 << 22), b""): s.update(blk)
                if s.hexdigest() != h: bad += 1; print("   differs", p)
            (ok if not bad else fix)(f"checksums: {n - bad}/{n} files match")
        else: fix(f"no SHA256SUMS next to the package ({sums})")


if __name__ == "__main__":
    main()
