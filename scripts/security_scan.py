# SPDX-License-Identifier: Apache-2.0
"""
Merge-time security & privacy scanner for oMLX.

Looks for malware indicators, telemetry/analytics, data-exfiltration patterns,
obfuscated code and unexpected network endpoints in the source tree and,
optionally, in a built package (wheel, sdist, zip, or an unpacked .app dir).

Design: *baseline + allowlist*. Everything currently reviewed lives in
``scripts/security_baseline.json``. The scan FAILS on anything new, so every
merge that introduces a new host, a new dangerous call, a new telemetry SDK,
a new binary, ... forces a human to look at it and consciously update the
baseline (``--update-baseline``) in the same PR.

Usage:
    PYTHONPATH=. python scripts/security_scan.py                # source tree
    PYTHONPATH=. python scripts/security_scan.py --dist dist/*.whl
    PYTHONPATH=. python scripts/security_scan.py --dist oMLX.app
    PYTHONPATH=. python scripts/security_scan.py --update-baseline
    PYTHONPATH=. python scripts/security_scan.py --pip-audit    # + CVE scan

Exit code: 0 = clean, 1 = findings (or hard failure).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import subprocess
import sys
import tarfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASELINE_PATH = ROOT / "scripts" / "security_baseline.json"

TEXT_EXT = {
    ".py", ".swift", ".js", ".mjs", ".ts", ".html", ".htm", ".sh", ".bash",
    ".zsh", ".yml", ".yaml", ".toml", ".json", ".cfg", ".ini", ".txt", ".md",
    ".plist", ".pth", ".rb", ".command", ".xcconfig", ".entitlements", "",
}
SKIP_DIRS = {
    ".git", ".venv", "venv", "node_modules", "__pycache__", ".mypy_cache",
    ".ruff_cache", ".pytest_cache", "build", "DerivedData", ".build", "_export",
}
# Files that legitimately describe the patterns we search for.
SELF_FILES = {"scripts/security_scan.py", "scripts/compliance_check.py",
              "SECURITY_AUDIT.md", "QWEN.md", "scripts/security_baseline.json"}
MAX_BYTES = 5_000_000
# Pure data / minified vendor bundles: free text and third-party libs make the
# pattern rules meaningless. They are still checked for hosts, binaries and
# conflict markers, and any *code* file is always scanned in full.
DATA_GLOBS = ("omlx/eval/data/", "omlx/admin/bench_corpora/",
              "omlx/oq_calibration_data.json", "omlx/oqe_calibration_data.json",
              "apps/omlx-web/omlx_web/i18n/")
MINIFIED = (".min.js", ".min.css")

# ── Rules ────────────────────────────────────────────────────────────────────
# Counted per (rule, file) and compared with the baseline count.
DANGEROUS = {
    "dyn-exec": r"(?<![\w.])(?:eval|exec)\(|\bcompile\(.{0,80}[\"']exec[\"']",
    "pickle-load": r"\b(?:pickle|cPickle|dill|cloudpickle|marshal|shelve)\.(?:loads?|Unpickler)\b",
    "os-system": r"\bos\.(?:system|popen|exec[lv]p?e?)\(|\bcommands\.getoutput",
    "shell-true": r"shell\s*=\s*True",
    "pipe-to-shell": r"(?:curl|wget)[^\n|]{0,200}\|\s*(?:sudo\s+)?(?:ba|z)?sh\b",
    "raw-socket": r"\bsocket\.socket\(|\bsocket\.create_connection\(",
    "ctypes-load": r"\bctypes\.(?:CDLL|cdll|WinDLL|windll)|\bdlopen\(",
    "dynamic-import": r"\b__import__\(|importlib\.import_module\(\s*[^\"'\s]",
    "b64-exec": r"b64decode[^\n]{0,200}\b(?:exec|eval|compile|loads)\b",
    "subprocess-net-tool": r"[\"'](?:curl|wget|nc|ncat|netcat|scp|sftp|ftp|telnet)[\"']",
    "swift-process": r"\bProcess\(\)|NSTask|\bsystem\(",
    "swift-url-session": r"URLSession\.(?:shared|init)|URLSession\(|NWConnection|URLRequest\(",
}
# Any hit is a failure unless baselined.
EXFIL = {
    "ssh-key-read": r"\.ssh/(?:id_|authorized_keys|known_hosts)|id_(?:rsa|ed25519|ecdsa)\b",
    "browser-data": r"(?:Cookies|Login Data|Web Data|key4\.db|logins\.json|cookies\.sqlite|Chrome/Default|Firefox/Profiles)",
    "keychain-dump": r"security\s+(?:find-|dump-)\w*(?:password|keychain)|SecItemCopyMatching",
    "dotenv-read": r"[\"']\.env[\"']|load_dotenv\(",
    "cloud-creds": r"\.aws/credentials|\.config/gcloud|\.azure/|\.kube/config|\.docker/config\.json|\.npmrc|\.pypirc|\.netrc",
    "exfil-sink": r"(?:discord(?:app)?\.com/api/webhooks|api\.telegram\.org/bot|hooks\.slack\.com|pastebin\.com|transfer\.sh|requestbin|webhook\.site|ngrok\.(?:io|app)|pipedream\.net|interact\.sh|\boast\.(?:pro|live|fun|online|site|me)\b|burpcollaborator|canarytokens)",
    "ip-literal-url": r"https?://(?!127\.0\.0\.1|0\.0\.0\.0|10\.|192\.168\.|localhost)\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",
    "clipboard-read": r"NSPasteboard\.general\.(?:string|data)|pyperclip\.paste|pbpaste",
    "screen-capture": r"CGWindowListCreateImage|CGDisplayCreateImage|ScreenCaptureKit|ImageGrab\.grab|pyautogui\.screenshot",
    "keylogger": r"CGEventTap|pynput|keyboard\.(?:hook|on_press)|NSEvent\.addGlobalMonitor",
    "persistence": r"LaunchAgents|LaunchDaemons|crontab\s+-|/etc/cron|schtasks|HKEY_.*\\Run\b|loginitems|SMAppService",
}
TELEMETRY = {
    "telemetry-sdk": r"\b(?:sentry_sdk|sentry-sdk|posthog|mixpanel|amplitude|segment\.analytics|analytics-python|datadog|ddtrace|newrelic|rollbar|bugsnag|honeybadger|logrocket|fullstory|hotjar|heap-analytics|appcenter|crashlytics|TelemetryDeck|Firebase|FirebaseAnalytics|GoogleAnalytics|Countly|Aptabase|PostHog|SentrySwift|Sentry|logfire|wandb|mlflow|opentelemetry-exporter|otlp)\b",
    "analytics-url": r"https?://[\w.-]*(?:google-analytics\.com|googletagmanager\.com|doubleclick\.net|segment\.(?:io|com)|mixpanel\.com|posthog\.com|plausible\.io|matomo|clarity\.ms|sentry\.io|datadoghq\.|amplitude\.com|bugsnag\.com|rollbar\.com|hotjar\.com|fullstory\.com|logrocket\.com|statcounter\.com|crashlytics\.com)",
    "phone-home": r"\b(?:phone[_ -]?home|send_telemetry|track_event|capture_event|report_usage|anonymous_id|machine_id|hardware_id|device_id|uuid\.getnode|IOPlatformUUID|IOPlatformSerialNumber)\b",
    "updater-framework": r"\b(?:Sparkle|SUFeedURL|SPUStandardUpdaterController|autoupdate|appcast)\b",
}
OBFUSCATION = {
    "long-b64-blob": r"[A-Za-z0-9+/]{400,}={0,2}",
    "hex-escape-blob": r"(?:\\x[0-9a-fA-F]{2}){40,}",
    "chr-join": r"(?:chr\(\d+\)\s*\+\s*){8,}|join\(\s*\[?\s*chr\(",
    "zlib-b64-exec": r"(?:zlib|bz2|lzma|gzip)\.decompress\(\s*(?:base64|b64)",
    "rot13-codecs": r"codecs\.decode\([^)]*rot",
}
RULESETS = {"danger": DANGEROUS, "exfil": EXFIL, "telemetry": TELEMETRY,
            "obfuscation": OBFUSCATION}
COMPILED = {
    (cat, name): re.compile(rx)
    for cat, rules in RULESETS.items()
    for name, rx in rules.items()
}

URL_RE = re.compile(r"""\b(?:https?|wss?|ftp)://([A-Za-z0-9._-]+(?:\.[A-Za-z]{2,})|localhost)""")
# Host patterns to ignore entirely (schemas, namespaces, loopback, placeholders).
IGNORE_HOST = re.compile(
    r"^(?:localhost|127\.0\.0\.1|.*\.local|example\.(?:com|org|net)|"
    r"(?:www\.)?w3\.org|json-schema\.org|schemas\.\w+\.\w+|ns\.adobe\.com|"
    r"purl\.org|xmlns\.\w+|www\.apple\.com|developer\.apple\.com)$"
)
ALLOWED_BINARY_EXT = {".png", ".jpg", ".jpeg", ".gif", ".ico", ".icns", ".svg",
                      ".woff", ".woff2", ".ttf", ".otf", ".pdf", ".webp",
                      ".car", ".mov", ".mp4", ".wav", ".mp3", ".flac",
                      ".xcstrings", ".nib", ".pyi", ".typed"}
SUSPICIOUS_BINARY_EXT = {".so", ".dylib", ".dll", ".exe", ".bin", ".o", ".a",
                         ".pyc", ".pyd", ".jar", ".class", ".apk", ".msi",
                         ".pkl", ".pickle", ".dmg", ".pkg", ".whl", ".egg",
                         ".node", ".wasm", ".sh", ".command", ".pth"}

# ── Helpers ──────────────────────────────────────────────────────────────────


def _git_files() -> list[Path]:
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True
        ).stdout.decode("utf-8", "replace")
        files = [ROOT / p for p in sorted(set(out.split("\0"))) if p]
        return [f for f in files if f.exists()]
    except Exception:
        return [p for p in ROOT.rglob("*") if p.is_file()]


def _rel(p: Path) -> str:
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def _skip(rel: str) -> bool:
    parts = rel.split("/")
    return any(d in SKIP_DIRS for d in parts[:-1]) or rel in SELF_FILES


def _is_text(name: str, data: bytes) -> bool:
    """Content-based: no NUL bytes and mostly valid UTF-8."""
    head = data[:8192]
    if b"\0" in head:
        return False
    try:
        head.decode("utf-8")
    except UnicodeDecodeError as exc:
        # tolerate a multi-byte char cut at the 8 KiB boundary
        if exc.start < len(head) - 4:
            return False
    return True


def scan_text(rel: str, text: str, sink: "Collector") -> None:
    """Run every regex rule over one text blob."""
    is_data = rel.startswith(DATA_GLOBS) or rel.endswith(MINIFIED)
    for (cat, name), rx in COMPILED.items():
        if is_data and cat != "obfuscation":
            continue
        # URL-ish rules would be hopelessly noisy in the committed docs.
        if rel.endswith((".md", ".xcstrings")) and cat != "exfil":
            continue
        n = len(rx.findall(text))
        if n:
            sink.count[f"{cat}:{name}"][rel] += n
    for m in URL_RE.finditer(text):
        host = m.group(1).lower()
        if not IGNORE_HOST.match(host):
            sink.hosts[host].add(rel)
    if re.search(r"^(?:<{7} |>{7} )", text, re.M):
        sink.hard.append(f"merge-conflict markers in {rel}")


class Collector:
    def __init__(self) -> None:
        self.count: dict[str, Counter] = defaultdict(Counter)
        self.hosts: dict[str, set[str]] = defaultdict(set)
        self.binaries: set[str] = set()
        self.hard: list[str] = []
        self.warn: list[str] = []
        self.hashes: dict[str, str] = {}

    def snapshot(self) -> dict:
        return {
            "patterns": {
                k: dict(sorted(v.items())) for k, v in sorted(self.count.items())
            },
            "hosts": sorted(self.hosts),
            "binaries": sorted(self.binaries),
        }


def scan_source(sink: Collector) -> None:
    for f in _git_files():
        rel = _rel(f)
        if _skip(rel) or f.stat().st_size > MAX_BYTES * 4:
            continue
        data = f.read_bytes()
        ext = f.suffix.lower()
        if _is_text(f.name, data):
            if ext == ".pth":
                sink.binaries.add(rel)
            scan_text(rel, data.decode("utf-8", "replace"), sink)
        elif ext not in ALLOWED_BINARY_EXT:
            sink.binaries.add(rel)
        if ext in SUSPICIOUS_BINARY_EXT - {".sh", ".command"}:
            sink.binaries.add(rel)
    scan_workflows_and_deps(sink)


def scan_workflows_and_deps(sink: Collector) -> None:
    for wf in (ROOT / ".github" / "workflows").glob("*.y*ml"):
        t = wf.read_text("utf-8", "replace")
        if "pull_request_target" in t and re.search(
            r"github\.event\.pull_request\.head\.(?:sha|ref)", t
        ):
            sink.hard.append(
                f"{_rel(wf)}: pull_request_target checks out untrusted PR code"
            )
        for m in re.finditer(r"uses:\s*([\w./-]+)@([\w./-]+)", t):
            if not re.fullmatch(r"[0-9a-f]{40}", m.group(2)) and not m.group(
                1
            ).startswith(("actions/", "./")):
                sink.warn.append(
                    f"{_rel(wf)}: third-party action {m.group(1)}@{m.group(2)} not pinned to a SHA"
                )
    pp = ROOT / "pyproject.toml"
    if pp.exists():
        t = pp.read_text("utf-8")
        for m in re.finditer(r'"([^"\n]*(?:git\+|https?://|file:)[^"\n]*)"', t):
            if "@" in m.group(1) or "git+" in m.group(1):
                sink.count["deps:url-or-git-dependency"][m.group(1)[:120]] += 1
        for tool in ("setup.py", "setup.cfg"):
            if (ROOT / tool).exists():
                sink.warn.append(f"{tool} present: review install-time code execution")


TELEMETRY_PKG = re.compile(
    r"sentry[-_]sdk|posthog|mixpanel|amplitude.*|analytics[-_]python|datadog|"
    r"ddtrace|newrelic|rollbar|bugsnag|honeybadger|logfire|opentelemetry[-_]exporter.*|"
    r"segment[-_]analytics.*|telemetry.*|appcenter.*"
)


def _project_path(name: str) -> str | None:
    """Map an archive member to a repo-relative path if it is project code."""
    n = re.sub(r"^[^/]*?\d+\.\d+[^/]*/", "", name) if "/" in name else name
    for cand in (n, name):
        for root in ("omlx/", "apps/", "omlx_web/", "scripts/"):
            i = cand.find(root)
            if i == 0 or (i > 0 and cand[i - 1] == "/" and "site-packages" not in cand):
                path = cand[i:]
                if "!/" in path:
                    path = path.split("!/")[-1]
                return path if path.startswith("omlx/") or (ROOT / path).exists() else (
                    "omlx/" + path if root == "omlx_web/" else path)
    return None


def scan_archive(path: Path, sink: Collector) -> None:
    """Scan a wheel/zip/sdist/.app dir with the same rules (prefix 'dist:')."""
    def feed(name: str, data: bytes) -> None:
        ext = Path(name).suffix.lower()
        proj = _project_path(name)
        rel = proj or f"dist:{name}"
        if name.endswith(".pth") and not name.endswith("distutils-precedence.pth"):
            sink.hard.append(f"dist:{name}: .pth file runs code at interpreter start")
        m = re.search(r"([\w.]+)-[\d.]+[^/]*\.(?:dist|egg)-info/METADATA$", name)
        if m and TELEMETRY_PKG.fullmatch(m.group(1).lower()):
            sink.hard.append(f"dist:{name}: telemetry/analytics package bundled")
        if _is_text(name, data):
            text = data[:MAX_BYTES].decode("utf-8", "replace")
            if proj:
                scan_text(rel, text, sink)
            else:  # third-party code: only high-signal exfil/obfuscation sinks
                for key in ("exfil:exfil-sink", "exfil:ip-literal-url",
                            "obfuscation:zlib-b64-exec"):
                    n = len(COMPILED[tuple(key.split(":"))].findall(text))
                    if n:
                        sink.hard.append(f"[{key}] {rel}: {n} hit(s) in third-party file")
        elif ext in SUSPICIOUS_BINARY_EXT or ext not in ALLOWED_BINARY_EXT:
            sink.binaries.add(rel)
            sink.hashes[rel] = hashlib.sha256(data).hexdigest()
        if ext in {".whl", ".zip", ".egg"} and len(data) < 200_000_000:
            try:
                with zipfile.ZipFile(io.BytesIO(data)) as z:
                    for i in z.infolist():
                        if not i.is_dir():
                            feed(f"{name}!/{i.filename}", z.read(i))
            except zipfile.BadZipFile:
                pass

    if path.is_dir():
        for f in path.rglob("*"):
            if f.is_file() and not f.is_symlink() and f.stat().st_size < 200_000_000:
                feed(f.relative_to(path).as_posix(), f.read_bytes())
    elif zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            for i in z.infolist():
                if not i.is_dir():
                    feed(i.filename, z.read(i))
    elif tarfile.is_tarfile(path):
        with tarfile.open(path) as t:
            for m in t.getmembers():
                if m.isfile():
                    feed(m.name, t.extractfile(m).read())  # type: ignore[union-attr]
    else:
        sink.hard.append(f"cannot read dist artifact {path}")


# ── Comparison with baseline ─────────────────────────────────────────────────


def compare(snap: dict, base: dict, is_dist: bool) -> list[str]:
    problems: list[str] = []
    bp = base.get("patterns", {})
    for key, files in snap["patterns"].items():
        cat = key.split(":", 1)[0]
        for rel, n in files.items():
            allowed = bp.get(key, {}).get(rel, 0)
            if n > allowed:
                problems.append(
                    f"[{key}] {rel}: {n} hit(s), baseline allows {allowed}"
                )
    allowed_hosts = set(base.get("hosts", []))
    for h in snap["hosts"]:
        if h not in allowed_hosts and not any(
            h.endswith("." + a) for a in allowed_hosts
        ):
            problems.append(f"[new-network-host] {h}")
    allowed_bin = set(base.get("dist_binaries" if is_dist else "binaries", []))
    for b in snap["binaries"]:
        if b not in allowed_bin:
            problems.append(f"[unexpected-binary/executable] {b}")
    return problems


def run_pip_audit() -> list[str]:
    print("\n--- pip-audit (known CVEs in installed dependencies) ---")
    try:
        r = subprocess.run(
            [sys.executable, "-m", "pip_audit", "--progress-spinner", "off"],
            capture_output=True, text=True, timeout=600,
        )
    except FileNotFoundError:
        return ["pip-audit not installed (pip install pip-audit)"]
    except subprocess.TimeoutExpired:
        return ["pip-audit timed out"]
    print(r.stdout[-3000:] + r.stderr[-1000:])
    if "No module named pip_audit" in r.stderr:
        return ["pip-audit not installed (pip install pip-audit)"]
    return [] if r.returncode == 0 else ["pip-audit reported vulnerabilities"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dist", nargs="*", help="built artifacts to scan (wheel, sdist, .app dir)")
    ap.add_argument("--update-baseline", action="store_true",
                    help="record the CURRENT state as reviewed (source tree only)")
    ap.add_argument("--pip-audit", action="store_true")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    base = json.loads(BASELINE_PATH.read_text()) if BASELINE_PATH.exists() else {}
    sink = Collector()
    if args.dist:
        for d in args.dist:
            scan_archive(Path(d), sink)
    else:
        scan_source(sink)
    snap = sink.snapshot()

    if args.update_baseline and args.dist:
        base["dist_binaries"] = sorted(snap["binaries"])
        BASELINE_PATH.write_text(json.dumps(base, indent=1, sort_keys=True) + "\n")
        print(f"Recorded {len(snap['binaries'])} reviewed native binaries (dist_binaries).")
        return 0
    if args.update_baseline:
        snap["dist_binaries"] = base.get("dist_binaries", [])
        BASELINE_PATH.write_text(json.dumps(snap, indent=1, sort_keys=True) + "\n")
        print(f"Baseline written: {_rel(BASELINE_PATH)}")
        print("REVIEW `git diff scripts/security_baseline.json` before committing.")
        return 0

    if not base:
        print("[FAIL] no baseline; run --update-baseline after a manual review")
        return 1

    problems = sink.hard + compare(snap, base, bool(args.dist))
    extra = run_pip_audit() if args.pip_audit else []

    print(f"Scanned {'artifacts' if args.dist else 'source tree'}: "
          f"{sum(sum(c.values()) for c in sink.count.values())} pattern hits, "
          f"{len(snap['hosts'])} hosts, {len(snap['binaries'])} binaries")
    for rel, h in sorted(sink.hashes.items()):
        print(f"[BIN]  {h}  {rel}")
    for w in sink.warn:
        print(f"[WARN] {w}")
    for p in problems + extra:
        print(f"[FAIL] {p}")
    if problems or extra:
        print(f"\nSECURITY SCAN FAILED: {len(problems) + len(extra)} finding(s). "
              "Review each; if legitimate, run --update-baseline and justify in the PR.")
        return 1
    print("SECURITY SCAN PASSED: nothing new vs. reviewed baseline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
