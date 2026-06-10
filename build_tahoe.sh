#!/bin/bash
# oMLX macOS Tahoe (26.x) App Build Script (Venv isolated)
# This script creates a temporary virtual environment to run the build,
# ensuring the host macOS Python environment remains clean.
# It prefers Python 3.11 to match the project's target runtime.
#
# Note: DMG creation is no longer part of this script — use xcrun productbuild
# or a dedicated DMG tool against the generated oMLX.app.

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== oMLX isolated build (Tahoe) ===${NC}"

# 1. Check Requirements & Select Python
echo -e "${GREEN}[1/8] Checking host environment...${NC}"

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
echo -e "${GREEN}[2/8] Creating build virtual environment (.build_venv)...${NC}"
$PYTHON_BIN -m venv .build_venv
source .build_venv/bin/activate

# Remember repo root before we cd into packaging/
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"

# 3. Install build-time requirements into the venv
echo -e "${GREEN}[3/8] Installing build dependencies (venvstacks + audit)...${NC}"
pip install --quiet --upgrade pip
pip install --quiet venvstacks setuptools pip-audit

# 4. Security Audit
echo -e "${GREEN}[4/8] Auditing packages for known vulnerabilities...${NC}"
# Scan the root project dependencies for security flaws
if pip-audit --desc on .; then
    echo -e "  ✓ No known vulnerabilities found."
else
    echo -e "  ${YELLOW}Warning: Security vulnerabilities detected. Check the report above.${NC}"
    # Optionally: exit 1 here if you want to block builds with any vulnerability
fi

# 5. Navigate to packaging directory and run venvstacks build
cd packaging
echo -e "${GREEN}[5/8] Building venvstacks Python layers…${NC}"
# Use the python from our venv
python build.py --venvstacks-only

# 6. Build the Swift app bundle (embeds venvstacks layers)
echo -e "${GREEN}[6/8] Building Swift app bundle…${NC}"
cd "$REPO_ROOT"
"$REPO_ROOT/apps/omlx-mac/Scripts/build.sh" swift 2>&1 || {
    echo -e "${YELLOW}Warning: Swift build failed; venvstacks export is still at packaging/_export/${NC}"
    exit 1
}

# 7. Copy staged app to project root
echo -e "${GREEN}[7/8] Copying oMLX.app to project root…${NC}"
VERSION=$(python -c "import re; print(re.search(r'__version__\s*=\s*\"([^\"]+)\"', open('omlx/_version.py').read()).group(1))")
STAGED_APP="$REPO_ROOT/apps/omlx-mac/build/Stage/oMLX.app"

if [[ -d "$STAGED_APP" ]]; then
    rm -rf "$REPO_ROOT/oMLX.app"
    cp -R "$STAGED_APP" "$REPO_ROOT/oMLX.app"
    echo -e "${GREEN}Success!${NC}"
    echo -e "App created at: ${BLUE}$REPO_ROOT/oMLX.app${NC}"
else
    echo -e "${YELLOW}Warning: Staged app not found at $STAGED_APP — venvstacks layers built but Swift app missing.${NC}"
    echo -e "Check: $REPO_ROOT/apps/omlx-mac/build/Stage/"
    exit 1
fi

echo ""
echo -e "${BLUE}Note: The build environment is located in .build_venv and can be removed after installation.${NC}"
echo -e "${BLUE}DMG creation is no longer part of this script — use xcrun productbuild or a dedicated DMG tool.${NC}"
