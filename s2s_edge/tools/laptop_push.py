# -*- coding: utf-8 -*-
"""LAPTOP: copy everything the box needs over the USB cable (adb), then verify checksums ON THE BOX. One command, no network.
  python tools/laptop_push.py                 # pushes to $HOME/onevoice on the box (usually /root/onevoice)
  python tools/laptop_push.py --all           # also the IQ-9075 fallback models and the Python 3.12 wheels (+4 GB, normally not needed)
  python tools/laptop_push.py --dest /data/onevoice
Pushed by default: the package without models/ (IQ-9075) and wheelhouse/ (Python 3.12), the BTC sample set, the ADV noise set.
Run it from the package folder or anywhere (it finds the deploy folder from its own location)."""
import argparse, os, shutil, subprocess, sys, time

ADB = shutil.which("adb") or r"C:\platform-tools\adb.exe"
PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # .../onevoice_deploy/s2s_edge_pkg
DEPLOY = os.path.dirname(PKG)
SKIP = {"models", "wheelhouse", ".venv", "logs", "__pycache__"}


def adb(*a, check=True, quiet=False):
    p = subprocess.run([ADB, *a], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if check and p.returncode: sys.exit(f"adb {' '.join(a)} failed:\n{p.stdout}{p.stderr}")
    if not quiet and p.stdout.strip(): print("   " + p.stdout.strip().splitlines()[-1])
    return p.stdout


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dest", default=""); ap.add_argument("--all", action="store_true"); a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if not os.path.exists(ADB) and not shutil.which("adb"): sys.exit("adb not found: install Android platform-tools to C:\\platform-tools")
    devs = [l for l in adb("devices", quiet=True).splitlines()[1:] if l.strip()]
    if not any(l.endswith("\tdevice") for l in devs): sys.exit(f"no box in `adb devices` ({devs or 'empty'}): check the USB-C data cable / Google USB Driver; 'unauthorized' = accept on the box")
    home = adb("shell", "echo $HOME", quiet=True).strip() or "/root"
    # adb shell may log in as an unprivileged user (e.g. "adb", HOME=/home/adb which does not exist / is not writable): use the first writable place
    dest = a.dest
    if not dest:
        for cand in (f"{home}/onevoice", "/tmp/onevoice"):
            if "OK" in adb("shell", f"mkdir -p {cand} 2>/dev/null && echo OK", check=False, quiet=True): dest = cand; break
        else: sys.exit("no writable folder found on the box for user " + adb("shell", "whoami", quiet=True).strip() + ": pass --dest <folder you can write>")
    else: adb("shell", f"mkdir -p {dest}", quiet=True)
    print(f"box: {devs[0].split()[0]} | user {adb('shell', 'whoami', quiet=True).strip()} | destination {dest}")
    print(adb("shell", f"mkdir -p {dest} && df -h {dest} | tail -1", quiet=True).strip())
    skip = set() if a.all else SKIP; skip |= {".venv", "logs", "__pycache__"}
    jobs = [(os.path.join(PKG, n), f"{dest}/s2s_edge_pkg/") for n in sorted(os.listdir(PKG)) if n not in skip]
    jobs += [(os.path.join(DEPLOY, n), f"{dest}/") for n in ("onevoice_btc_data", "onevoice_btc_adv") if os.path.isdir(os.path.join(DEPLOY, n))]
    adb("shell", f"mkdir -p {dest}/s2s_edge_pkg", quiet=True); t0 = time.time()
    for src, dst in jobs:
        print(f"-> {os.path.relpath(src, DEPLOY)}"); adb("push", src, dst)
    # checksums of exactly what was pushed (skipped folders left out), verified on the box
    sums = os.path.join(DEPLOY, "SHA256SUMS"); lines = []
    if os.path.exists(sums):
        for l in open(sums, encoding="utf-8"):
            p = l.split("  ", 1)[1].strip()[2:] if "  " in l else ""
            if p and p.split("/")[0] not in skip: lines.append(l)
        tmp = os.path.join(DEPLOY, "SHA256SUMS.box"); open(tmp, "w", encoding="utf-8", newline="\n").writelines(lines); adb("push", tmp, f"{dest}/SHA256SUMS.box", quiet=True)
        out = adb("shell", f"cd {dest}/s2s_edge_pkg && sha256sum -c ../SHA256SUMS.box --quiet && echo CHECKSUM_OK", check=False, quiet=True)
        print(f"checksum ({len(lines)} files): " + ("OK" if "CHECKSUM_OK" in out else "FAILED\n" + out[-1500:]))
    print(f"done in {time.time() - t0:.0f} s. Next:  adb shell  ->  log in to the account you were given (su - <account>)  ->  cd {dest}/s2s_edge_pkg && bash start.sh demo   (or: bash start.sh box). Guide: HUONG_DAN_CHAY_BOX.md")


if __name__ == "__main__":
    main()
