# oMLX Privacy Audit Report & Remediation Plan

**Date:** May 24, 2026  
**Auditor:** Gemini CLI (Autonomous Mode)  
**Project:** oMLX (v0.3.10)  
**Status:** **NON-COMPLIANT** (due to unconditional benchmark telemetry)

---

## 1. Executive Summary
An autonomous privacy audit was conducted to verify oMLX's "Zero Telemetry" and "Local-First" claims. While most components (Inference, Update Checks, Logging) are compliant, a critical privacy violation was discovered in the **Benchmark** module. The system unconditionally uploads performance data and hardware-linked identifiers to a remote server without user consent or an opt-out mechanism.

---

## 2. Methodology
The audit utilized the following techniques:
1.  **Static Analysis:** Grep-based search for network sinks (`requests`, `httpx`, `urllib`).
2.  **Configuration Review:** Inspection of `omlx/settings.py` and `SECURITY_AUDIT.md`.
3.  **Endpoint Trace:** Mapping UI actions in `dashboard.js` to backend routes in `routes.py`.
4.  **Hardware Probe Audit:** Analyzing `omlx/utils/hardware.py` for fingerprinting logic.

---

## 3. Compliance Analysis

### 3.1 Compliant Components (PASSED)
*   **Core Inference:** Verified local-only processing. No prompt data is sent to external APIs.
*   **Update Checks:** Default is `OFF`. Logic in `omlx/admin/routes.py` respects `settings.server.check_updates`.
*   **Logging:** Centralized in `omlx/logging_config.py`; uses local file rotation. No remote logging found.
*   **Integrations:** `omlx/integrations/claude.py` proactive disables telemetry for sub-processes.

### 3.2 Non-Compliant Components (FAILED)
*   **Module:** `omlx/admin/benchmark.py`
*   **Issue:** Unconditional outbound `POST` request to `https://omlx.ai/api/benchmarks`.
*   **Data Leakage:** 
    *   **Hardware Fingerprint:** `owner_hash` computed from `IOPlatformUUID` (Unique Device ID).
    *   **Environment Info:** OS Version, Chip Name, GPU Cores, RAM Size.
    *   **Performance Data:** TPS (Tokens Per Second), Latency, Memory Usage.
*   **Violated Policy:** `README.md` ("No Analytics: No telemetry... Your data never leaves your machine.") and `SECURITY_AUDIT.md` (Section 6: "Zero Telemetry").

---

## 4. Technical Root Cause
The function `_upload_to_omlx_ai` in `omlx/admin/benchmark.py` is called at the end of every successful benchmark run (Line 824) without checking for user preference. The `BenchmarkRequest` Pydantic model lacks a consent field, and the Admin UI does not provide a toggle.

---

## 5. Remediation Proposal

### Phase 1: Backend & API Model (Opt-In Enforcement)
**File:** `omlx/admin/benchmark.py`
1.  Add `allow_upload: bool = False` to `BenchmarkRequest`.
2.  Update `run_benchmark` to wrap the upload call:
```python
# omlx/admin/benchmark.py
if run.request.allow_upload:
    try:
        await _upload_to_omlx_ai(run, engine_pool)
    except Exception as e:
        logger.warning(f"Benchmark upload failed: {e}")
```

### Phase 2: Frontend Implementation (Consent UI)
**File:** `omlx/admin/templates/dashboard/_bench.html`
Add a "Share results" checkbox to the Configuration Card:
```html
<label class="flex items-center gap-2 cursor-pointer mt-4">
    <input type="checkbox" x-model="benchAllowUpload"
           class="w-4 h-4 rounded border-neutral-300 text-neutral-900">
    <span class="text-sm font-medium text-neutral-700">Share results with community (omlx.ai)</span>
</label>
<p class="text-xs text-neutral-400">Shares performance stats and hardware model anonymously.</p>
```

**File:** `omlx/admin/static/js/dashboard.js`
Update `startBenchmark()` to include the `allow_upload` flag in the POST body to `/api/bench/start`.

### Phase 3: Automated Compliance Gate
**File:** `scripts/compliance_check.py`
Add a check to verify that no `requests.post(OMLX_AI_API_URL, ...)` calls exist without an accompanying conditional check.

---

## 6. Verification Plan
1.  **Manual Verification:** Run a benchmark with the "Share" box unchecked; verify no outbound traffic to `omlx.ai` via network logs.
2.  **Automated Verification:** Update `tests/test_benchmark.py` to assert that `_upload_to_omlx_ai` is NOT called when `allow_upload=False`.
3.  **Linter Verification:** Ensure `python scripts/compliance_check.py` passes.

---
**Approval Status:**  
[ ] Technical Lead Review  
[ ] Security Officer Review  
[ ] Compliance Verified  
