# SPDX-License-Identifier: Apache-2.0
"""
Compliance and Security Linter for oMLX.
Verifies that settings defaults and documentation adhere to privacy and security policies.
"""

import re
import sys
from pathlib import Path

def check_file_content(path, pattern, description, expected=True):
    """Check if a file's content matches a pattern."""
    if not path.exists():
        print(f"[ERROR] File not found: {path}")
        return False
    
    content = path.read_text()
    match = re.search(pattern, content)
    
    if (match is not None) == expected:
        print(f"[OK] {description}")
        return True
    else:
        status = "found" if match else "not found"
        print(f"[FAIL] {description} (Pattern {status})")
        return False

def check_settings_defaults():
    """Verify security-critical defaults in settings.py."""
    print("\n--- Verifying settings.py Defaults ---")
    settings_path = Path("omlx/settings.py")
    results = [
        check_file_content(settings_path, r'host: str = "127\.0\.0\.1"', "Default host is 127.0.0.1"),
        check_file_content(settings_path, r'check_updates: bool = False', "check_updates is False by default"),
        check_file_content(settings_path, r'check_statuskit: bool = False', "check_statuskit is False by default"),
        check_file_content(settings_path, r'skip_api_key_verification: bool = False', "API key verification is enabled by default"),
    ]
    return all(results)

def check_readme_privacy():
    """Verify README.md adheres to privacy documentation standards."""
    print("\n--- Verifying README.md Privacy Compliance ---")
    readme_path = Path("README.md")
    # Sections that should NOT exist
    prohibited = [
        (r'## Quickstart', "Prohibited section 'Quickstart'"),
        (r'## Performance Benchmark', "Prohibited section 'Performance Benchmark'"),
        (r'## Integrations', "Prohibited section 'Integrations'"),
        (r'## Connect with Us', "Prohibited section 'Connect with Us'"),
    ]
    
    results = []
    for pattern, desc in prohibited:
        results.append(check_file_content(readme_path, pattern, desc, expected=False))
    
    # Required privacy statement
    results.append(check_file_content(readme_path, r'# 🛡️ Privacy & Secure macOS Build', "Privacy & Security section exists"))
    
    return all(results)

def check_license_headers():
    """Verify all .py files have the Apache-2.0 license header."""
    print("\n--- Verifying License Headers ---")
    py_files = list(Path("omlx").rglob("*.py")) + list(Path("tests").rglob("*.py"))
    missing = []
    for f in py_files:
        if "__pycache__" in str(f): continue
        content = f.read_text()
        if "# SPDX-License-Identifier: Apache-2.0" not in content:
            missing.append(f)
    
    if not missing:
        print(f"[OK] All {len(py_files)} Python files have license headers.")
        return True
    else:
        print(f"[FAIL] Missing license headers in {len(missing)} files:")
        for f in missing[:5]: print(f"  - {f}")
        if len(missing) > 5: print(f"  ... and {len(missing)-5} more.")
        return False

def main():
    """Run all compliance checks."""
    print("oMLX Compliance & Security Audit Tool")
    
    success = True
    success &= check_settings_defaults()
    success &= check_readme_privacy()
    success &= check_license_headers()
    
    print("\n--- Audit Summary ---")
    if success:
        print("ALL CHECKS PASSED. Project is compliant with Security Audit v1.0.")
        sys.exit(0)
    else:
        print("SOME CHECKS FAILED. Please review the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
