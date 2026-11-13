"""Start or safely reuse the user-facing local OneVoice web server."""
from pathlib import Path
import argparse, json, os, signal, subprocess, sys, tempfile, time, urllib.request

root = Path(__file__).resolve().parents[1]
if str(root / "s2s_edge") not in sys.path:
    sys.path.insert(0, str(root / "s2s_edge"))

from s2s_edge.registry import resolve_runtime_profile, runtime_code_fingerprint

PROFILE_KEYS = ("source_mode", "vad_profile", "ort_policy", "tts_format")


class LauncherError(RuntimeError):
    """Explicit launcher failure carrying a deterministic non-zero exit code."""

    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = int(code)


def _norm_env_arg(val):
    if val in (None, ""):
        return None
    return str(val)


def build_launch_env(
    base_env=None,
    vad_profile=None,
    ort_policy=None,
    source_mode=None,
    tts_format=None,
):
    """Build subprocess env honoring caller S2S_NEURAL_TN/VAD/ORT/TTS settings without forcing 1 over 0."""
    env = dict(os.environ if base_env is None else base_env)
    if vad_profile is not None:
        env["S2S_VAD_PROFILE"] = str(vad_profile)
    if ort_policy is not None:
        env["S2S_ORT_POLICY"] = str(ort_policy)
    if source_mode is not None:
        env["S2S_NEURAL_TN"] = str(source_mode)
    if tts_format is not None:
        env["S2S_TTS_FORMAT"] = str(tts_format)
    env.setdefault("S2S_NEURAL_TN", "1")
    env.setdefault("S2S_TTS_FORMAT", "hybrid")
    env.setdefault("S2S_VAD_PROFILE", "legacy")
    env.setdefault("S2S_ORT_POLICY", "auto")
    env.setdefault("HF_HOME", os.path.join(tempfile.gettempdir(), "s2s-hf"))   # /private/tmp exists only on macOS
    requested_profile_from_env(env)
    return env


def requested_profile_from_env(env):
    """Resolve and validate the effective runtime profile purely from the supplied mapping."""
    return resolve_runtime_profile(
        vad_profile=_norm_env_arg(env.get("S2S_VAD_PROFILE")),
        ort_policy=_norm_env_arg(env.get("S2S_ORT_POLICY")),
        source_mode=_norm_env_arg(env.get("S2S_NEURAL_TN")),
        tts_format=_norm_env_arg(env.get("S2S_TTS_FORMAT")),
    )


def disk_code_fingerprint():
    """Return current on-disk fingerprint covering s2s_edge modules, tools/s2s_web.py, and git HEAD."""
    return runtime_code_fingerprint(extra_files=[str(root / "tools/s2s_web.py")])


def fetch_health(url, timeout=2):
    """Fetch /api/health JSON payload when server responds healthy, else return None."""
    try:
        with urllib.request.urlopen(url + "/api/health", timeout=timeout) as r:
            data = json.load(r)
        if isinstance(data, dict) and data.get("ok") is True:
            return data
        return None
    except Exception:
        return None


def ready(url):
    return fetch_health(url) is not None


def compare_server_state(health, requested_profile, disk_fp):
    """Verify server /api/health matches requested profile and current on-disk code fingerprint."""
    reasons = []
    if not isinstance(health, dict) or health.get("ok") is not True:
        return False, ["health_not_ok"]

    eff = health.get("effective_profile")
    if not isinstance(eff, dict):
        reasons.append("missing_effective_profile")

    for key in PROFILE_KEYS:
        want = requested_profile.get(key)
        got_top = health.get(key)
        if got_top != want:
            reasons.append(f"profile_mismatch:{key}:want={want!r},got={got_top!r}")
        if isinstance(eff, dict):
            got_eff = eff.get(key)
            if got_eff != want:
                reasons.append(f"effective_profile_mismatch:{key}:want={want!r},got={got_eff!r}")

    loaded_fp = health.get("loaded_code_fingerprint")
    if not isinstance(loaded_fp, dict):
        reasons.append("missing_loaded_code_fingerprint")
    else:
        for fp_key in ("s2s_edge_sha256", "server_files_sha256", "git_head"):
            if fp_key not in loaded_fp:
                reasons.append(f"missing_fingerprint_field:{fp_key}")
            elif loaded_fp.get(fp_key) != disk_fp.get(fp_key):
                reasons.append(
                    f"stale_code:{fp_key}:loaded={loaded_fp.get(fp_key)!r},disk={disk_fp.get(fp_key)!r}"
                )

    return len(reasons) == 0, reasons


def launch_or_reuse(
    args,
    *,
    base_env=None,
    health_fn=None,
    fingerprint_fn=None,
    popen_fn=None,
    kill_fn=None,
    sleep_fn=None,
):
    """Validate requested config, safely reuse matching server or reload idle server cleanly."""
    health_fn = fetch_health if health_fn is None else health_fn
    fingerprint_fn = disk_code_fingerprint if fingerprint_fn is None else fingerprint_fn
    popen_fn = subprocess.Popen if popen_fn is None else popen_fn
    kill_fn = os.kill if kill_fn is None else kill_fn
    sleep_fn = time.sleep if sleep_fn is None else sleep_fn

    env = build_launch_env(
        base_env=base_env,
        vad_profile=getattr(args, "vad_profile", None),
        ort_policy=getattr(args, "ort_policy", None),
        source_mode=getattr(args, "source_mode", None),
        tts_format=getattr(args, "tts_format", None),
    )
    req_profile = requested_profile_from_env(env)
    url = f"http://localhost:{args.port}"

    health = health_fn(url)
    was_healthy = health is not None
    if was_healthy:
        disk_fp = fingerprint_fn()
        matches, reasons = compare_server_state(health, req_profile, disk_fp)
        if matches and not getattr(args, "reload", False):
            return {"action": "reused", "url": url, "profile": req_profile, "reasons": []}
        if not getattr(args, "reload", False):
            raise LauncherError(
                2,
                f"Existing server at {url} does not match requested profile or disk code "
                f"(refusing unsafe reuse; pass --reload to replace when idle): {'; '.join(reasons)}",
            )

        pending = health.get("pending")
        if not isinstance(pending, int) or isinstance(pending, bool) or pending != 0:
            raise LauncherError(
                3,
                f"Refusing --reload for {url}: server reports pending={pending!r} (must be 0).",
            )
        pid = health.get("pid")
        if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 1:
            raise LauncherError(
                3,
                f"Refusing --reload for {url}: missing or invalid server pid={pid!r}.",
            )

        kill_fn(pid, signal.SIGTERM)
        stopped = False
        for _ in range(40):
            if health_fn(url) is None:
                stopped = True
                break
            sleep_fn(0.25)
        if not stopped:
            raise LauncherError(4, f"Server pid={pid} at {url} did not stop after SIGTERM.")

    cmd = [
        sys.executable,
        "-u",
        str(root / "tools/s2s_web.py"),
        "--port",
        str(args.port),
        "--vad-profile",
        req_profile["vad_profile"],
        "--ort-policy",
        req_profile["ort_policy"],
        "--source-mode",
        req_profile["source_mode"],
        "--tts-format",
        req_profile["tts_format"],
    ]
    if args.out_dir:
        cmd.extend(["--out-dir", str(args.out_dir)])
    log_path = Path(args.log_file) if args.log_file else (root / "logs/s2s_web.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as f:
        proc = popen_fn(
            cmd,
            cwd=str(root),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    if args.out_dir:
        pid_path = Path(args.out_dir) / "server.pid"
        pid_path.parent.mkdir(parents=True, exist_ok=True)
        pid_path.write_text(str(proc.pid))
    elif args.port == 8766:
        pid_path = root / "results/mic_web/server.pid"
        pid_path.parent.mkdir(parents=True, exist_ok=True)
        pid_path.write_text(str(proc.pid))

    started = False
    for _ in range(40):
        if health_fn(url) is not None:
            started = True
            break
        if proc.poll() is not None:
            raise RuntimeError(f"Web server exited; inspect {log_path}")
        sleep_fn(0.25)
    if not started:
        raise RuntimeError("Web server did not become ready")
    return {
        "action": "reloaded" if was_healthy else "started",
        "url": url,
        "profile": req_profile,
        "pid": proc.pid,
    }


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Start the OneVoice web server.")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--log-file", default=None)
    ap.add_argument("--vad-profile", default=None, choices=["hybrid", "legacy", "upstream"])
    ap.add_argument("--ort-policy", default=None, choices=["auto", "basic", "disable"])
    ap.add_argument("--source-mode", default=None, choices=["0", "1"])
    ap.add_argument("--tts-format", default=None, choices=["hybrid", "legacy"])
    ap.add_argument(
        "--reload",
        action="store_true",
        help="Safely SIGTERM a matching-PID idle (pending==0) server on mismatch and restart.",
    )
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        res = launch_or_reuse(args)
    except LauncherError as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(exc.code)
    except ValueError as exc:
        print(f"ERROR: invalid runtime configuration: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2)
    print(res["url"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
