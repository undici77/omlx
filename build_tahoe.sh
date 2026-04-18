#!/bin/bash
# oMLX macOS Tahoe (26.x) DMG Build Script (Venv isolated)
# This script creates a temporary virtual environment to run the build,
# ensuring the host macOS Python environment remains clean.
# It prefers Python 3.11 to match the project's target runtime.

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== oMLX isolated build (Tahoe) ===${NC}"

# 1. Check Requirements & Select Python
echo -e "${GREEN}[1/5] Checking host environment...${NC}"

if [[ "$(uname)" != "Darwin" ]]; then
    echo "Error: This script must be run on macOS."
    exit 1
fi

# Search for Python 3.11 (the target version for venvstacks.toml)
if command -v python3.11 &> /dev/null; then
    PYTHON_BIN="python3.11"
    echo -e "  Found Python 3.11: $(python3.11 --version)"
elif command -v python3 &> /dev/null; then
    PYTHON_BIN="python3"
    HOST_VER=$(python3 --version)
    echo -e "  ${YELLOW}Note: Python 3.11 not found in PATH. Using host $HOST_VER.${NC}"
else
    echo "Error: python3 not found."
    exit 1
fi

# 2. Create temporary venv for the build process
echo -e "${GREEN}[2/5] Creating build virtual environment (.build_venv)...${NC}"
$PYTHON_BIN -m venv .build_venv
source .build_venv/bin/activate

# 3. Install build-time requirements into the venv
echo -e "${GREEN}[3/6] Installing build dependencies (venvstacks + audit)...${NC}"
pip install --quiet --upgrade pip
pip install --quiet venvstacks setuptools pip-audit

# 4. Security Audit
echo -e "${GREEN}[4/6] Auditing packages for known vulnerabilities...${NC}"
# Scan the root project dependencies for security flaws
if pip-audit --desc on --project ..; then
    echo -e "  ✓ No known vulnerabilities found."
else
    echo -e "  ${YELLOW}Warning: Security vulnerabilities detected. Check the report above.${NC}"
    # Optionally: exit 1 here if you want to block builds with any vulnerability
fi

# 5. Navigate to packaging directory and run build
cd packaging
echo -e "${GREEN}[5/6] Running oMLX build process...${NC}"
# Use the python from our venv
python build.py

# 6. Locate the output
echo -e "${GREEN}[6/6] Locating generated DMG...${NC}"
VERSION=$(python -c "import re; print(re.search(r'__version__\s*=\s*\"([^\"]+)\"', open('../omlx/_version.py').read()).group(1))")
DMG_FILE=$(ls dist/oMLX-${VERSION}.dmg 2>/dev/null | head -n 1)

if [[ -f "$DMG_FILE" ]]; then
    echo -e "${GREEN}Success!${NC}"
    echo -e "DMG created at: ${BLUE}$(pwd)/$DMG_FILE${NC}"
    echo ""
    echo -e "${BLUE}Note: The build environment is located in .build_venv and can be removed after installation.${NC}"
else
    echo "Error: DMG file was not found in the dist/ directory."
    exit 1
fi
