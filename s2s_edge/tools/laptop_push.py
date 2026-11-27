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
    ap = argparse.ArgumentParser(); ap.add_argument("--dest", default=""); ap.add_argument("--all", action="store_true")
    ap.add_argument("--sync", action="store_true", help="push only files that are missing or differ on the box (checksums), then verify")
    ap.add_argument("--dry-run", action="store_true", help="with --sync: only list what would be pushed"); a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if not os.path.exists(ADB) and not shutil.which("adb"): sys.exit("adb not found: install Android platform-tools to C:\\platform-tools")
    devs = [l for l in adb("devices", quiet=True).splitlines()[1:] if l.strip()]
    if not any(l.endswith("\tdevice") for l in devs): sys.exit(f"no box in `adb devices` ({devs or 'empty'}): check the USB-C data cable / Google USB Driver; 'unauthorized' = accept on the box")
    home = adb("shell", "echo $HOME", quiet=True).strip() or "/root"
    # adb shell may log in as an unprivileged user (e.g. "adb", HOME=/home/adb which does not exist / is not writable): use the first writable place
    dest = a.dest
    if not dest:                                                            # an existing copy first (sync updates it in place)
        for cand in ("/tmp/onevoice", "/opt/onevoice", f"{home}/onevoice"):
            if "YES" in adb("shell", f"[ -f {cand}/s2s_edge_pkg/start.sh ] && echo YES", check=False, quiet=True): dest = cand; print(f"existing package found at {cand}"); break
    if not dest:
        # never under /root: user "system" (owner of the NPU) cannot read it
        for cand in [c for c in (f"{home}/onevoice", "/tmp/onevoice") if not c.startswith("/root")]:
            if "OK" in adb("shell", f"mkdir -p {cand} 2>/dev/null && echo OK", check=False, quiet=True): dest = cand; break
        else: sys.exit("no writable folder found on the box for user " + adb("shell", "whoami", quiet=True).strip() + ": pass --dest <folder you can write>")
    else: adb("shell", f"mkdir -p {dest}", quiet=True)
    print(f"box: {devs[0].split()[0]} | user {adb('shell', 'whoami', quiet=True).strip()} | destination {dest}")
    print(adb("shell", f"mkdir -p {dest} && df -h {dest} | tail -1", quiet=True).strip())
    skip = set() if a.all else SKIP; skip |= {".venv", "logs", "__pycache__"}
    if a.sync: return sync(dest, skip, a.dry_run)
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


def box_sums(skip):
    """SHA256SUMS lines of the package files that are pushed (skipped folders left out), written to SHA256SUMS.box."""
    lines = []
    for l in open(os.path.join(DEPLOY, "SHA256SUMS"), encoding="utf-8"):
        p = l.split("  ", 1)[1].strip()[2:] if "  " in l else ""
        if p and p.split("/")[0] not in skip: lines.append(l)
    tmp = os.path.join(DEPLOY, "SHA256SUMS.box"); open(tmp, "w", encoding="utf-8", newline="\n").writelines(lines)
    return tmp, lines


def ok_paths(sha_output):
    """Paths that `sha256sum -c` confirmed ("./a/b.py: OK" -> "a/b.py"). Anything not confirmed (FAILED, missing, no output at all) is pushed."""
    out = set()
    for l in sha_output.splitlines():
        if l.rstrip().endswith(": OK"):
            p = l.rstrip()[:-4].strip(); out.add(p[2:] if p.startswith("./") else p)
    return out


def local_count(d):
    return sum(len(fs) for _, _, fs in os.walk(d))


def sync(dest, skip, dry):
    """Bring the box copy up to date with the fewest bytes: checksum every package file ON THE BOX, push only the missing / different ones,
    push a data set only if its file count differs, then verify everything again. Works on an empty box too (then it pushes everything)."""
    t0 = time.time(); pkg = f"{dest}/s2s_edge_pkg"
    tmp, lines = box_sums(skip); adb("shell", f"mkdir -p {pkg}", quiet=True); adb("push", tmp, f"{dest}/SHA256SUMS.box", quiet=True)
    rels = [l.split("  ", 1)[1].strip()[2:] for l in lines]
    good = ok_paths(adb("shell", f"cd {pkg} && sha256sum -c ../SHA256SUMS.box 2>/dev/null", check=False, quiet=True))
    todo = [p for p in rels if p not in good]
    if not good and rels: print("!! the box confirmed no file (empty box, or sha256sum/cd failed): everything will be pushed")
    size = sum(os.path.getsize(os.path.join(PKG, p)) for p in todo if os.path.exists(os.path.join(PKG, p)))
    print(f"package: {len(lines) - len(todo)}/{len(lines)} files already identical on the box; to push: {len(todo)} files, {size / 1e6:.1f} MB")
    for p in todo[:40]: print("   " + p)
    if len(todo) > 40: print(f"   ... and {len(todo) - 40} more")
    data = []
    for n in ("onevoice_btc_data", "onevoice_btc_adv"):
        src = os.path.join(DEPLOY, n)
        if not os.path.isdir(src): continue
        have = adb("shell", f"[ -d {dest}/{n} ] && find {dest}/{n} -type f | wc -l || echo 0", check=False, quiet=True).strip().splitlines()
        have = int(have[-1]) if have and have[-1].strip().isdigit() else 0; want = local_count(src)
        print(f"{n}: {have}/{want} files on the box" + ("" if have == want else "  -> push"))
        if have != want: data.append(n)
    if dry: print("dry run: nothing pushed"); return
    dirs = sorted({os.path.dirname(p) for p in todo if os.path.dirname(p)})
    for i in range(0, len(dirs), 50):                                       # create the folders first, a few shell calls instead of one per file
        adb("shell", "mkdir -p " + " ".join(f"'{pkg}/{d}'" for d in dirs[i:i + 50]), quiet=True)
    for k, p in enumerate(todo, 1):
        print(f"-> [{k}/{len(todo)}] {p}", flush=True); adb("push", os.path.join(PKG, p), f"{pkg}/{p}", quiet=True)
    for n in data:
        print(f"-> {n}/", flush=True); adb("push", os.path.join(DEPLOY, n), f"{dest}/", quiet=True)
    out = adb("shell", f"cd {pkg} && sha256sum -c ../SHA256SUMS.box --quiet && echo CHECKSUM_OK", check=False, quiet=True)
    ok = "CHECKSUM_OK" in out
    print(f"checksum ({len(lines)} files): " + ("OK" if ok else "FAILED\n" + out[-1500:]))
    print(f"done in {time.time() - t0:.0f} s. Next:  adb shell  ->  su -  ->  cd {pkg} && bash start.sh check   then   bash start.sh demo")
    if not ok: sys.exit(1)


if __name__ == "__main__":
    main()
