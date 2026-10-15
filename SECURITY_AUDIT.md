# oMLX Security Audit & Compliance Guide

**Version:** 1.3
**Last Updated:** October 15, 2026
**Project:** oMLX (Open MLX Inference Server)

---

## ⚡ Auditor Quick-Start — what to run on every merge

Run from the repo root (`source .venv/bin/activate`, `export PYTHONPATH=.`). **Any failure blocks the merge.**

| # | Check | Command | Expected | Covers |
|---|-------|---------|----------|--------|
| 1 | No conflict leftovers | `git diff --name-only --diff-filter=U` and `grep -rlE "^(<<<<<<<\|>>>>>>>) " omlx/ tests/ scripts/ apps/ *.md` | both empty | unresolved merges |
| 2 | Compliance | `python scripts/compliance_check.py` | `ALL CHECKS PASSED` | secure defaults, README privacy, license headers |
| 3 | **Malware / telemetry / privacy scan** | `python scripts/security_scan.py` | `SECURITY SCAN PASSED` | new hosts, telemetry SDKs, exec/pickle, exfiltration, obfuscation, binaries, git deps |
| 4 | Same scan on the **built package** | `python scripts/security_scan.py --dist dist/*.whl` (or unpacked `.app`) | `SECURITY SCAN PASSED` | what actually ships (native binaries, bundled telemetry pkgs, `.pth`) |
| 5 | Known CVEs *(release builds; also in `build_tahoe.sh`)* | `python scripts/security_scan.py --pip-audit` | no vulnerabilities | dependencies |
| 6 | Fast tests | `pytest -m "not slow" -q` | 0 failed / 0 errors | behaviour (see `QWEN.md` for triage) |

**If check 3 or 4 fails:** do not baseline blindly. Follow the decision table in §5.6.2.

### Runtime spot-checks (macOS host with a real install — the scripts cannot verify these)

```bash
lsof -i :8000                                            # bound to 127.0.0.1 only (unless remote access is intended)
stat -f "%Lp" ~/.omlx/settings.json                      # must print 600   (Linux: stat -c %a)
grep skip_api_key_verification ~/.omlx/settings.json     # must be false
grep -r "Incoming POST" ~/.omlx/logs                     # empty = OK; content = log level too high (TRACE leaks prompts)
```

---

## 1. Privacy Policy & Data Handling

oMLX is **local-first by design.** No user data leaves the device unless the user explicitly enables an opt-in feature.

| Data Category | Policy | Storage |
| :--- | :--- | :--- |
| **User Prompts** | Processed in GPU/RAM only; never transmitted | Local logs only if `log_level = TRACE` (discouraged) |
| **Model Weights** | Loaded into Unified Memory; not transmitted | `~/.omlx/models` |
| **API Keys** | Plaintext; protected by OS file permissions | `~/.omlx/settings.json` (must be `0600`) |
| **Telemetry / Analytics** | **None.** Zero "phone home" by default | N/A |
| **Update Checks** | **Disabled by default.** Anonymous `GET` to GitHub only if user enables `check_updates` | N/A |
| **Benchmark Results** | **Opt-in only** (`allow_upload: false` default). User must check "Share results" in UI | omlx.ai (only when explicitly enabled) |
| **Hardware Fingerprint** | Only included in benchmark uploads; only when `allow_upload: true` | omlx.ai (only when explicitly enabled) |
| **Update Actions** | User-initiated only; no auto-install | N/A |

### Privacy Compliance Statement

oMLX complies with **privacy-by-design** principles:

1. **Local Processing:** Inference never leaves the device.
2. **Zero Telemetry by Default:** No analytics, no usage tracking, no opt-out needed — it is simply off.
3. **Explicit Opt-In for Sharing:** Benchmark results and hardware info are transmitted only when the user checks "Share results" (`allow_upload: true`). The default is `false`.
4. **Data Sovereignty:** The user owns the weights, the logs, and the API keys.

---

## 2. Security Architecture

oMLX's security model assumes it runs on a **trusted host (macOS)** accessed locally or via a secured network.

- **Framework:** FastAPI (Python 3.10+)
- **API Auth:** Bearer Token (OpenAI/Anthropic compatible) or `x-api-key` header
- **Admin Auth:** Signed session cookies (`itsdangerous` URLSafeTimedSerializer)
- **Default Binding:** `127.0.0.1:8000` (loopback only)
- **Encryption at rest:** None for settings; TLS via reverse proxy (Nginx/Caddy) if network-exposed

---

## 3. Security Audit Checklist

### 3.1 Privacy & Outbound Traffic (Audit First)

- [ ] **Telemetry disabled:** `security_scan.py` reports no new `telemetry:*` hit and no new host. Only the opt-in benchmark upload (`omlx/admin/benchmark.py`, `accuracy_upload.py`) may contact omlx.ai.
- [ ] **Accepted outbound features** (user-initiated, reviewed; their hosts live in the baseline): model download (huggingface.co / hf-mirror / modelscope), web search (DuckDuckGo default, Brave, SearXNG), GitHub release check (off by default), cluster peers.
- [ ] **Benchmark upload gated:** Verify `BenchmarkRequest.allow_upload` defaults to `False` and the upload is only called when `run.request.allow_upload is True`.
- [ ] **Update check off:** Verify `server.check_updates` is `false` in `settings.json` (default).
- [ ] **StatusKit check off:** Verify `server.check_statuskit` is `false` (default; probes a system setting on macOS Tahoe).
- [ ] **Log level safe:** Verify `server.log_level` is `info` or `warning`. `TRACE` (level 5) logs full prompt text.

### 3.2 Network & Connectivity

- [ ] **Binding:** Server binds to `127.0.0.1` unless remote access is explicitly required.
- [ ] **CORS:** default is `["http://localhost", "http://127.0.0.1"]`; `cors_origins` must not be `["*"]` if the server is internet-exposed.
- [ ] **Cluster** (`omlx/cluster/`): pairing token kept secret (it authenticates the pickle control channel, OMLX-06); SSH key files `0600`/`0700`; peers on trusted network only.
- [ ] **TLS:** A reverse proxy (Nginx/Caddy) provides HTTPS if accessed over a network.
- [ ] **MCP:** `mcp.example.json` contains no hardcoded secrets and no overly permissive tool access.

### 3.3 Authentication & Authorization

- [ ] **Bypass off:** `auth.skip_api_key_verification` is `false`.
- [ ] **API Key strength:** `api_key` is a high-entropy string (not `"1234"`).
- [ ] **Session secret:** `auth.secret_key` is unique and was server-generated (default behavior).
- [ ] **Sub-key scope:** Sub-keys only access `/v1/` routes; they cannot reach `/admin/`.

### 3.4 Filesystem & Permissions

- [ ] **Config permissions:** `~/.omlx/settings.json` has mode `0600`.
- [ ] **Path traversal:** Model directories are validated against `base_path` using `Path.resolve()`.
- [ ] **Log retention:** `logging.retention_days` is configured to prevent disk exhaustion.

### 3.5 Dependency & Supply Chain Security (Vulnerabilities & Malware Auditing)

- [ ] **Vulnerability Auditing:** `pip-audit` blocks macOS builds (`build_tahoe.sh`); run `security_scan.py --pip-audit` on demand.
- [ ] **Malicious Payload & Stealer Scanning:** **Automated by `scripts/security_scan.py` (§5.6)** — it covers the indicators below on every merge and on built packages:
  - Check for dynamic/obfuscated code execution (e.g. `eval()`, `exec()`, or base64 decoding to execute payloads).
  - Verify that no raw outbound calls exist outside of official API routes (e.g., raw TCP/UDP sockets, unsanctioned `urllib` or `requests` calls).
  - Inspect files for exfiltration behavior (such as scraping SSH keys, browser cookies, `.env` files, or routing data to unverified webhooks/IPs).
- [ ] **Reproducible Builds & Lockfiles:** Ensure `venvstacks` enforces deterministic dependency resolutions with frozen versions and date-locking (`exclude-newer`) to shield against compromised upstream updates.
- [ ] **Static Security Testing (SAST):** Optionally run `bandit -r omlx -ll` for extra coverage (not installed by default; `security_scan.py` is the mandatory gate).

---

## 4. Known Security Risks (Risk Register)

| ID | Risk | Mitigation / Status |
| :--- | :--- | :--- |
| **OMLX-01** | **Auth Bypass:** `skip_api_key_verification = true` grants full unauthenticated admin access. | Default `false`; documented warning. |
| **OMLX-02** | **Plaintext Config:** Secrets in `settings.json` without encryption. | OS-level `0600` permissions; no fix planned (local-only threat model). |
| **OMLX-03** | **CSRF:** Admin panel POST actions lack explicit CSRF tokens. | Partially mitigated by `SameSite=Lax` cookies. |
| **OMLX-04** | **CORS:** default is now `["http://localhost", "http://127.0.0.1"]`; a user may still set `["*"]`. | Default restrictive; documented. |
| **OMLX-06** | **Pickle over cluster socket** (`omlx/cluster/control_plane.py`): `pickle.loads` after HMAC-SHA256 check. | Safe only while the shared token stays secret; token leak = RCE on cluster peers. Baselined. |
| **OMLX-07** | **Import-time source patching** (`omlx/patches/mlx_vlm_mlx0322_compat.py` uses `exec`). | Local installed package only, no network input. Baselined. |
| **OMLX-05** | **Benchmark Telemetry:** Benchmark module can upload hardware fingerprint + performance data. | **Mitigated (May 2026):** `allow_upload` defaults to `false`; upload is gated behind user opt-in both in backend (`BenchmarkRequest.allow_upload: bool = False`) and UI checkbox. `compliance_check.py` verifies the default. |

---

## 5. CI / Non-macOS Testing Security

### 5.1 The MLX Mock: Scope and Safety

oMLX uses a NumPy-backed import-hook mock (`omlx/utils/mlx_mock.py`) to run tests on Linux/CI where the real `mlx` C++ library cannot be installed.

**Security properties that must be preserved:**

| Property | Detail |
|----------|--------|
| **Mock never activates on macOS** | `omlx/__init__.py` guards with `if platform.system() != "Darwin"` — the mock is unconditionally skipped on the only production platform. |
| **Mock is installed before any test** | `conftest.py → import omlx → install_mock()` inserts `MockMLXFinder` at `sys.meta_path[0]`. No test can accidentally import real `mlx` on Linux. |
| **Single source of truth** | `omlx/utils/mlx_mock.py` is the only file with mock logic. `tests/mlx_mock.py` is a 5-line re-export shim — never a duplicate. |

### 5.2 Fail-Loud Mock Policy

The mock raises `NotImplementedError` for any unknown MLX function:

```python
def _default_func(*args, _n=_captured_name, **kwargs):
    raise NotImplementedError(
        f"mlx.{_n}() is not implemented in the MLX mock. "
        f"Add it to omlx/utils/mlx_mock.py."
    )
```

**Why this matters:** A silent zeros return would let tests pass with wrong results, masking real bugs. `NotImplementedError` forces every new MLX surface area to be **consciously reviewed and explicitly implemented** before tests can pass.

**Policy:** Any change that makes `_default_func` return a value instead of raising must include a documented justification in the PR.

### 5.3 Test Quality Policies

| Policy | Rule | Rationale |
|--------|------|-----------|
| `@pytest.mark.slow` | Tests requiring a real model load or hardware-dependent numerical precision | Cannot run in CI; must not block PRs |
| `@pytest.mark.integration` | Tests requiring a live oMLX server | Excluded from default `pytest` run |
| `pytest.skip()` in fixture | Tests requiring external data (vocab files, network) | Offline CI must not fail on missing downloads |
| Inline mock override | **Never** inline `sys.meta_path` hacks in test files | Use `omlx/utils/mlx_mock.py` exclusively |
| **Harmony token ID tests** | **Skip automatically via conftest.py** — do NOT add mock implementations for `HarmonyEncoding`, `StreamableParser`, etc. | Mock cannot produce exact token IDs; conftest.py `pytest_collection_modifyitems` auto-skips all tests whose file path or method name contains "harmony" when the MLX mock is active |
| **MLX-dependent tests on Linux** | **Skip automatically via conftest.py `_MLX_SKIP_REASONS`** — do NOT maintain mock implementations for MLX internals (RotatingKVCache, PoolingCache, GenerationBatch.filter, mx.quantize shapes, etc.) | Mock is NumPy-backed and cannot reproduce real MLX C++ internals; maintaining per-function shims is fragile and unmaintainable. See `tests/conftest.py` for the skip list. |

> **⚠️ Critical: Skip, don't maintain Harmony mocks.**
> The MLX mock cannot produce the exact token IDs that the real `openai_harmony`
> library generates. Tests that depend on real Harmony token IDs would fail spuriously.
> **Never add mock implementations for `HarmonyEncoding.encode()`, `StreamableParser`, etc.**
> Instead, `tests/conftest.py` automatically skips any test file or method whose
> name/path contains "harmony" when the mock is active. This keeps test files
> unchanged — zero modifications needed.

> **⚠️ Critical: Skip, don't maintain MLX internal mocks.**
> The MLX mock cannot reproduce real MLX C++ internals such as:
> - `RotatingKVCache.is_trimmable`, `PoolingCache.accumulate_windows`
> - `GenerationBatch.filter()` proper uid/logits_processors alignment
> - `mx.quantize()` return shape for mxfp4 (3-tuple vs 2-tuple)
> - Gemma-4 real regex parser (uses MLX's regex engine)
>
> **Never add mock implementations for these MLX internals.**
> Instead, `tests/conftest.py` automatically skips test files listed in
> `_MLX_SKIP_REASONS` when the MLX mock is active. This keeps test files
> unchanged — zero modifications needed. When a new test file depends on
> real MLX internals, add it to `_MLX_SKIP_REASONS` in conftest.py rather
> than extending the mock.

`pytest.ini` enforces: `addopts = -m "not slow and not integration"` — the default CI run must yield **zero failures**.

### 5.4 Compliance Verification — Mandatory Pre-Commit Gate

Run before every commit that modifies Python source files:
```bash
PYTHONPATH=. python scripts/compliance_check.py
```

Verifies:
1. **`settings.py` security defaults** — `host=127.0.0.1`, `check_updates=False`, `check_statuskit=False`, `skip_api_key_verification=False`, `benchmark.allow_upload=False`
2. **`README.md` privacy sections** — required sections exist; prohibited telemetry/tracking sections are absent
3. **License headers** — every `.py` file begins with `# SPDX-License-Identifier: Apache-2.0`

**Do not commit if this check fails.** (`compliance_check.py` does not detect conflict markers — gate 1 and `security_scan.py` do.)

### 5.5 Merge Conflict Security Checklist

When resolving merge conflicts, verify these files specifically:

- [ ] **`omlx/_version.py`** — Retains `# SPDX-License-Identifier: Apache-2.0` header; version is the higher of the two.
- [ ] **`omlx/settings.py`** — Security defaults not accidentally changed. Run `compliance_check.py` after resolving.
- [ ] **`omlx/utils/mlx_mock.py`** — Remains the single canonical mock. Not reverted to an older state.
- [ ] **`tests/mlx_mock.py`** — Remains the 5-line shim; not replaced with a 400+ line duplicate.
- [ ] **`tests/conftest.py`** — `import omlx` appears before any `import mlx.*`; `_MLX_IMPORT_SKIP` / `_MLX_SKIP_REASONS` lists not lost.
- [ ] **`scripts/security_baseline.json`** — not hand-merged; re-run `security_scan.py` and review any additions (§5.6.2).
- [ ] **No leftover conflict markers** (incl. READMEs): `grep -rlE "^(<<<<<<<|>>>>>>>) " omlx/ tests/ scripts/ apps/ *.md`.

### 5.6 Merge-Time Malware / Telemetry / Privacy Scan (mandatory on every merge)

#### 5.6.1 What it does

`scripts/security_scan.py` compares the source tree (or a built package) with the human-reviewed `scripts/security_baseline.json` and **fails on anything new**:

| Category | Detects |
|----------|---------|
| Network | URL hosts not in the baseline; IP-literal URLs; exfil sinks (webhooks, pastebin, ngrok, …) |
| Telemetry | analytics/crash SDKs (sentry, posthog, mixpanel, firebase, …), analytics URLs, device-ID harvesting, updater frameworks |
| Dangerous calls | `eval/exec`, `pickle/marshal`, `os.system`, `shell=True`, `curl \| sh`, raw sockets, `ctypes`, dynamic imports, Swift `Process`/`URLSession` |
| Exfiltration | `~/.ssh`, browser cookie/login stores, keychain reads, cloud credential files, clipboard/screen capture, keyloggers, persistence (LaunchAgents/cron) |
| Obfuscation | long base64/hex blobs, `chr()` chains, `zlib+b64` payloads |
| Supply chain | git/URL dependencies, unexpected binaries, `.pth` files, `pull_request_target` on untrusted code, unpinned third-party Actions (warn), conflict markers |
| Built package (`--dist`) | project files vs. baseline; third-party files for exfil/obfuscation only; bundled telemetry packages; every native binary (`.so/.dylib/.metallib`) must be in `dist_binaries` (SHA-256 printed) |

Pure data (`omlx/eval/data/`, calibration JSON, i18n, `*.min.js`) is excluded from pattern rules to avoid noise, but is still checked for hosts, binaries and obfuscation.

```bash
python scripts/security_scan.py                   # source tree
python scripts/security_scan.py --dist dist/*.whl # built wheel / sdist / unpacked .app
python scripts/security_scan.py --pip-audit       # + known CVEs
python scripts/security_scan.py --update-baseline # human-reviewed change only (add --dist for native binaries)
```

CI: `.github/workflows/security-scan.yml` runs compliance + source scan + wheel scan on every PR and push.

#### 5.6.2 A finding appeared — decision procedure

1. **Read the line**: `[category:rule] file: N hit(s), baseline allows M` or `[new-network-host] host`.
2. **Open the code** (`grep -n` the pattern in that file) and answer: *what does it do, and can untrusted data or the network reach it?*
3. **Decide**:

| Verdict | Action |
|---------|--------|
| Suspicious / unexplained / sends data off-device without opt-in | **Remove it**, do not merge, report to the maintainer. |
| Real risk but needed (e.g. `pickle`, `exec`) | Harden (HMAC, no untrusted input, arg lists instead of shell), add a row to the Risk Register (§4). |
| Legitimate feature (HF download, web search, cluster, keychain HF token…) | `--update-baseline`, then explain *each added line* of `git diff scripts/security_baseline.json` in the PR. |
| False positive (docs, dataset text, test fixture, minified lib) | Prefer tightening the rule / `DATA_GLOBS` in the script; otherwise baseline with a justification. |
| New native binary | Rebuild from reviewed `csrc/` in CI, then `--update-baseline --dist`; record the SHA-256 in the PR. |

4. **Never** run `--update-baseline` merely to make CI green. Reviewers must reject PRs that change the baseline without justification.

#### 5.6.3 Limits (be honest in reviews)

Static pattern matching cannot prove the absence of malware. Compiled binaries are allowlisted by name only (hash is printed, not enforced). Third-party packages inside a bundled `.app` get only the high-signal checks. Complement with `pip-audit`, pinned/date-locked dependencies (§3.5) and code review of every baseline change.

---

*Update this document whenever new API routes, storage mechanisms, security-relevant settings defaults, or test infrastructure decisions are added to oMLX.*
