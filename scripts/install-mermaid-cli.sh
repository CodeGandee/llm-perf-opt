#!/bin/bash
# Install Mermaid CLI (@mermaid-js/mermaid-cli) globally with browser support
# This script installs mermaid-cli system-wide for diagram generation
#
# Usage: ./scripts/install-mermaid-cli.sh [--yes|-y]
# (Do NOT run with sudo - script will use npm global install)

set -e  # Exit on any error

# Parse command line arguments
AUTO_YES=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --yes|-y)
            AUTO_YES=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--yes|-y]"
            exit 1
            ;;
    esac
done

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}Mermaid CLI Installation${NC}"
echo -e "${GREEN}================================${NC}"
echo ""

# Helper function to ask yes/no questions
ask_yes_no() {
    local prompt="$1"
    if [ "$AUTO_YES" = true ]; then
        echo -e "${prompt} ${GREEN}[auto-yes]${NC}"
        return 0
    fi

    read -p "$(echo -e ${prompt}) (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        return 0
    else
        return 1
    fi
}

# Check if Node.js and npm are installed
echo -e "${BLUE}Step 1: Checking Node.js and npm${NC}"
if ! command -v node &> /dev/null; then
    echo -e "${RED}Error: Node.js is not installed.${NC}"
    echo "Please install Node.js first:"
    echo "  Ubuntu/Debian: sudo apt-get install -y nodejs npm"
    echo "  Or use nvm: https://github.com/nvm-sh/nvm"
    exit 1
fi

if ! command -v npm &> /dev/null; then
    echo -e "${RED}Error: npm is not installed.${NC}"
    echo "Please install npm first:"
    echo "  Ubuntu/Debian: sudo apt-get install -y npm"
    exit 1
fi

NODE_VERSION=$(node --version)
NPM_VERSION=$(npm --version)
echo -e "${GREEN}✓ Node.js: $NODE_VERSION${NC}"
echo -e "${GREEN}✓ npm: $NPM_VERSION${NC}"
echo ""

# Check current mermaid-cli installation
echo -e "${BLUE}Step 2: Checking current Mermaid CLI installation${NC}"
CURRENT_VERSION=""
if npm list -g @mermaid-js/mermaid-cli &> /dev/null; then
    CURRENT_VERSION=$(npm list -g @mermaid-js/mermaid-cli --depth=0 2>/dev/null | grep @mermaid-js/mermaid-cli | sed 's/.*@//' | sed 's/ .*//' || echo "")
    if [ -n "$CURRENT_VERSION" ]; then
        echo -e "${GREEN}✓ Current version: $CURRENT_VERSION${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Mermaid CLI is not installed${NC}"
fi
echo ""

# Check latest available version
echo -e "${BLUE}Step 3: Checking latest available version${NC}"
LATEST_VERSION=$(npm view @mermaid-js/mermaid-cli version 2>/dev/null || echo "")
if [ -z "$LATEST_VERSION" ]; then
    echo -e "${RED}Error: Could not fetch latest version from npm registry${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Latest version: $LATEST_VERSION${NC}"
echo ""

# Determine if installation/upgrade is needed
SHOULD_INSTALL=false
if [ -z "$CURRENT_VERSION" ]; then
    echo -e "${YELLOW}Mermaid CLI needs to be installed.${NC}"
    SHOULD_INSTALL=true
elif [ "$CURRENT_VERSION" != "$LATEST_VERSION" ]; then
    echo -e "${YELLOW}A newer version is available: $CURRENT_VERSION → $LATEST_VERSION${NC}"
    if ask_yes_no "${YELLOW}Do you want to upgrade?${NC}"; then
        SHOULD_INSTALL=true
    else
        echo -e "${BLUE}Keeping current version $CURRENT_VERSION${NC}"
    fi
else
    echo -e "${GREEN}✓ Mermaid CLI is up to date ($CURRENT_VERSION)${NC}"
fi
echo ""

# Install or upgrade mermaid-cli
if [ "$SHOULD_INSTALL" = true ]; then
    echo -e "${BLUE}Step 4: Installing/Upgrading Mermaid CLI${NC}"
    echo -e "${YELLOW}Running: npm install -g @mermaid-js/mermaid-cli@latest${NC}"
    echo ""

    npm install -g @mermaid-js/mermaid-cli@latest

    echo ""
    echo -e "${GREEN}✓ Mermaid CLI installed successfully${NC}"
    echo ""
else
    echo -e "${BLUE}Step 4: Installation skipped${NC}"
    echo ""
fi

# Check if mmdc command is available
echo -e "${BLUE}Step 5: Verifying mmdc command${NC}"
if ! command -v mmdc &> /dev/null; then
    echo -e "${RED}Error: mmdc command not found after installation${NC}"
    echo "Try running: source ~/.bashrc or open a new terminal"
    exit 1
fi

MMDC_VERSION=$(mmdc --version 2>/dev/null || echo "unknown")
echo -e "${GREEN}✓ mmdc command available: $MMDC_VERSION${NC}"
echo ""

# Check and install browser for Puppeteer
echo -e "${BLUE}Step 6: Checking Puppeteer browser${NC}"

# Try to find puppeteer installation
PUPPETEER_DIR=$(npm root -g 2>/dev/null)/puppeteer || echo ""
MMDC_DIR=$(npm root -g 2>/dev/null)/@mermaid-js/mermaid-cli || echo ""

# Check if browser is installed by trying a simple render
echo -e "${YELLOW}Testing browser availability...${NC}"

# Create a simple test diagram
TEST_MERMAID_FILE="/tmp/mermaid-test-$$.mmd"
TEST_OUTPUT_FILE="/tmp/mermaid-test-$$.svg"

cat > "$TEST_MERMAID_FILE" << 'EOF'
graph TD
    A[Test] --> B[Diagram]
EOF

# Try to render it
if mmdc -i "$TEST_MERMAID_FILE" -o "$TEST_OUTPUT_FILE" 2>&1 | grep -q "No usable sandbox\|Failed to launch\|could not find Chrome"; then
    echo -e "${YELLOW}⚠ Browser not available or sandbox issue detected${NC}"
    echo ""

    # Clean up test files
    rm -f "$TEST_MERMAID_FILE" "$TEST_OUTPUT_FILE"

    # Check if Chrome browser is already installed via @puppeteer/browsers
    echo -e "${BLUE}Checking for existing Chrome installations...${NC}"

    # Use Puppeteer's default cache directory (since v19.0.0)
    BROWSER_CACHE_DIR="$HOME/.cache/puppeteer"
    BROWSER_LIST=$(npx -y @puppeteer/browsers list --path "$BROWSER_CACHE_DIR" 2>/dev/null || echo "")

    if echo "$BROWSER_LIST" | grep -q "chrome@"; then
        CHROME_VERSION=$(echo "$BROWSER_LIST" | grep "chrome@" | head -n 1 | awk '{print $1}')
        echo -e "${GREEN}✓ Chrome browser already installed: ${CHROME_VERSION}${NC}"
        echo -e "${GREEN}  Location: ${BROWSER_CACHE_DIR}${NC}"
        echo ""
        echo -e "${YELLOW}Existing browsers:${NC}"
        echo "$BROWSER_LIST" | sed 's/^/  /'
        echo ""

        if ask_yes_no "${YELLOW}Chrome is already installed. Do you want to reinstall/update it?${NC}"; then
            SHOULD_INSTALL_BROWSER=true
        else
            echo -e "${BLUE}Keeping existing browser installation${NC}"
            SHOULD_INSTALL_BROWSER=false
        fi
    else
        echo -e "${YELLOW}No Chrome browser found via @puppeteer/browsers${NC}"
        if ask_yes_no "${YELLOW}Do you want to install Chrome browser via @puppeteer/browsers?${NC}"; then
            SHOULD_INSTALL_BROWSER=true
        else
            SHOULD_INSTALL_BROWSER=false
        fi
    fi

    if [ "$SHOULD_INSTALL_BROWSER" = true ]; then
        echo ""
        echo -e "${BLUE}Installing Chrome browser for Puppeteer...${NC}"

        # Use Puppeteer's default cache directory (automatically discovered by mermaid-cli)
        BROWSER_CACHE_DIR="$HOME/.cache/puppeteer"
        echo -e "${YELLOW}Cache directory: ${BROWSER_CACHE_DIR}${NC}"
        echo -e "${YELLOW}Running: npx @puppeteer/browsers install chrome@stable --path ${BROWSER_CACHE_DIR}${NC}"
        echo ""

        # Install browser via @puppeteer/browsers with Puppeteer's default cache path
        if npx -y @puppeteer/browsers install chrome@stable --path "$BROWSER_CACHE_DIR"; then
            echo ""
            echo -e "${GREEN}✓ Chrome browser installed successfully${NC}"
            echo ""

            # Test again with the newly installed browser
            echo -e "${YELLOW}Testing browser again...${NC}"
            cat > "$TEST_MERMAID_FILE" << 'EOF'
graph TD
    A[Test] --> B[Working]
EOF
            if mmdc -i "$TEST_MERMAID_FILE" -o "$TEST_OUTPUT_FILE" &>/dev/null && [ -f "$TEST_OUTPUT_FILE" ]; then
                echo -e "${GREEN}✓ Browser is now working correctly${NC}"
                rm -f "$TEST_MERMAID_FILE" "$TEST_OUTPUT_FILE"

                # Automatically create global puppeteer config for convenience
                echo ""
                if ask_yes_no "${YELLOW}Create global puppeteer-config.json to make --no-sandbox work by default?${NC}"; then
                    PUPPETEER_CONFIG_GLOBAL="$HOME/.config/mermaid/puppeteer-config.json"
                    mkdir -p "$(dirname "$PUPPETEER_CONFIG_GLOBAL")"

                    cat > "$PUPPETEER_CONFIG_GLOBAL" << 'GLOBAL_EOF'
{
  "args": ["--no-sandbox", "--disable-setuid-sandbox"]
}
GLOBAL_EOF

                    echo ""
                    echo -e "${GREEN}✓ Created global config: ${PUPPETEER_CONFIG_GLOBAL}${NC}"
                    echo ""
                    echo "Usage (from any directory):"
                    echo -e "  ${GREEN}mmdc -p ~/.config/mermaid/puppeteer-config.json -i diagram.mmd -o output.png${NC}"
                fi
            else
                echo -e "${YELLOW}⚠ Browser still not working.${NC}"
                echo "Try creating puppeteer-config.json:"
                echo -e "  ${GREEN}echo '{\"args\": [\"--no-sandbox\"]}' > puppeteer-config.json${NC}"
                echo -e "  ${GREEN}mmdc -p puppeteer-config.json -i input.mmd -o output.svg${NC}"
                rm -f "$TEST_MERMAID_FILE" "$TEST_OUTPUT_FILE"
            fi
        else
            echo -e "${RED}✗ Failed to install browser${NC}"
            echo ""
            echo -e "${YELLOW}Alternative solutions:${NC}"
            echo ""
            echo "Option 1: Install chromium browser system-wide"
            echo -e "  Ubuntu/Debian: ${GREEN}sudo apt-get install -y chromium-browser${NC}"
            echo ""
            echo "Option 2: Use puppeteer config with --no-sandbox"
            echo -e "  ${GREEN}echo '{\"args\": [\"--no-sandbox\"]}' > puppeteer-config.json${NC}"
            echo -e "  ${GREEN}mmdc -p puppeteer-config.json -i input.mmd -o output.svg${NC}"
            echo ""
            echo "Option 3: Configure AppArmor (Ubuntu 23.10+)"
            echo -e "  See: ${BLUE}https://chromium.googlesource.com/chromium/src/+/main/docs/security/apparmor-userns-restrictions.md${NC}"
        fi
    else
        echo ""
        echo -e "${YELLOW}Skipping browser installation.${NC}"
        echo ""
        echo -e "${YELLOW}You can install it later with:${NC}"
        echo -e "  ${GREEN}npx @puppeteer/browsers install chrome@stable${NC}"
        echo ""
        echo "Or use puppeteer config for --no-sandbox:"
        echo -e "  ${GREEN}echo '{\"args\": [\"--no-sandbox\"]}' > puppeteer-config.json${NC}"
        echo -e "  ${GREEN}mmdc -p puppeteer-config.json -i input.mmd -o output.svg${NC}"
    fi
elif [ -f "$TEST_OUTPUT_FILE" ]; then
    echo -e "${GREEN}✓ Browser is working correctly${NC}"
    rm -f "$TEST_MERMAID_FILE" "$TEST_OUTPUT_FILE"
else
    echo -e "${YELLOW}⚠ Browser test inconclusive${NC}"
    rm -f "$TEST_MERMAID_FILE" "$TEST_OUTPUT_FILE"
fi

echo ""

# Final verification
echo -e "${BLUE}Step 7: Final verification${NC}"
echo ""

# Generate UUID for output file
if command -v uuidgen &> /dev/null; then
    UUID=$(uuidgen)
elif command -v cat /proc/sys/kernel/random/uuid &> /dev/null; then
    UUID=$(cat /proc/sys/kernel/random/uuid)
else
    # Fallback: use timestamp + random number
    UUID="$(date +%s)-$$-$RANDOM"
fi

# Create tmp directory if it doesn't exist
mkdir -p /tmp

# Test diagram file paths
FINAL_TEST_MMD="/tmp/mermaid-install-test-$$.mmd"
FINAL_TEST_PNG="/tmp/${UUID}.png"

# Create a comprehensive test diagram
cat > "$FINAL_TEST_MMD" << 'EOF'
graph TB
    subgraph "Mermaid CLI Installation"
        A[Start] --> B{Node.js?}
        B -->|Yes| C[Install Mermaid CLI]
        B -->|No| D[Install Node.js]
        D --> C
        C --> E{Browser?}
        E -->|Yes| F[Test Render]
        E -->|No| G[Install Chrome]
        G --> F
        F --> H[Success!]
    end

    style A fill:#90EE90
    style H fill:#90EE90
    style F fill:#87CEEB
EOF

echo -e "${YELLOW}Rendering final test diagram to PNG...${NC}"
echo -e "${BLUE}Output: ${FINAL_TEST_PNG}${NC}"
echo ""

# Create puppeteer config with --no-sandbox (use it by default to avoid issues)
PUPPETEER_CONFIG=$(mktemp --suffix=.json)
cat > "$PUPPETEER_CONFIG" << 'EOF'
{
  "args": ["--no-sandbox", "--disable-setuid-sandbox"]
}
EOF

# Render with puppeteer config
RENDER_OUTPUT=$(mktemp)
RENDER_SUCCESS=false

echo -e "${YELLOW}Rendering with --no-sandbox configuration...${NC}"
if mmdc -p "$PUPPETEER_CONFIG" -i "$FINAL_TEST_MMD" -o "$FINAL_TEST_PNG" -b transparent 2>&1 | tee "$RENDER_OUTPUT"; then
    if [ -f "$FINAL_TEST_PNG" ]; then
        FILE_SIZE=$(du -h "$FINAL_TEST_PNG" | cut -f1)
        echo ""
        echo -e "${GREEN}✓ Diagram rendered successfully!${NC}"
        echo -e "${GREEN}  Location: ${FINAL_TEST_PNG}${NC}"
        echo -e "${GREEN}  Size: ${FILE_SIZE}${NC}"
        echo ""

        # Offer to create global puppeteer config for out-of-box experience
        if ask_yes_no "${YELLOW}Create global puppeteer-config.json and 'mmdc-quick' alias for easy use?${NC}"; then
            PUPPETEER_CONFIG_PERMANENT="$HOME/.config/mermaid/puppeteer-config.json"
            mkdir -p "$(dirname "$PUPPETEER_CONFIG_PERMANENT")"

            cat > "$PUPPETEER_CONFIG_PERMANENT" << 'PERM_EOF'
{
  "args": ["--no-sandbox", "--disable-setuid-sandbox"]
}
PERM_EOF

            echo ""
            echo -e "${GREEN}✓ Created config: ${PUPPETEER_CONFIG_PERMANENT}${NC}"

            # Detect shell and rc file
            SHELL_RC=""
            if [ -n "$BASH_VERSION" ]; then
                SHELL_RC="$HOME/.bashrc"
            elif [ -n "$ZSH_VERSION" ]; then
                SHELL_RC="$HOME/.zshrc"
            else
                # Fallback to bashrc
                SHELL_RC="$HOME/.bashrc"
            fi

            # Check if alias already exists
            if grep -q "alias mmdc-quick=" "$SHELL_RC" 2>/dev/null; then
                echo -e "${YELLOW}⚠ Alias 'mmdc-quick' already exists in ${SHELL_RC}${NC}"
                echo "Skipping alias creation"
            else
                # Add alias to shell rc file
                cat >> "$SHELL_RC" << 'ALIAS_EOF'

# Mermaid CLI alias with --no-sandbox (added by install-mermaid-cli.sh)
alias mmdc-quick='mmdc -p ~/.config/mermaid/puppeteer-config.json'
ALIAS_EOF

                echo -e "${GREEN}✓ Created alias 'mmdc-quick' in ${SHELL_RC}${NC}"
                echo ""
                echo "Usage (after restarting shell or sourcing rc file):"
                echo -e "  ${GREEN}mmdc-quick -i diagram.mmd -o output.png${NC}"
                echo ""
                echo "To use immediately:"
                echo -e "  ${GREEN}source ${SHELL_RC}${NC}"
            fi
        fi

        RENDER_SUCCESS=true
    fi
fi

# If failed, show error
if [ "$RENDER_SUCCESS" = false ]; then
    echo ""
    echo -e "${YELLOW}⚠ Diagram rendering encountered issues${NC}"
    echo ""
    echo -e "${YELLOW}Manual test:${NC}"
    echo "  1. Create puppeteer-config.json:"
    echo -e "     ${GREEN}echo '{\"args\": [\"--no-sandbox\"]}' > /tmp/puppeteer-config.json${NC}"
    echo ""
    echo "  2. Run mmdc with config:"
    echo -e "     ${GREEN}mmdc -p /tmp/puppeteer-config.json -i $FINAL_TEST_MMD -o /tmp/test-output.png${NC}"
fi

# Clean up
rm -f "$FINAL_TEST_MMD" "$RENDER_OUTPUT" "$PUPPETEER_CONFIG"

echo ""
echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}Installation Complete!${NC}"
echo -e "${GREEN}================================${NC}"
echo ""
echo -e "${YELLOW}Usage examples:${NC}"
echo ""
echo "1. Basic usage:"
echo -e "   ${GREEN}mmdc -i diagram.mmd -o output.svg${NC}"
echo ""
echo "2. PNG output:"
echo -e "   ${GREEN}mmdc -i diagram.mmd -o output.png${NC}"
echo ""
echo "3. With custom theme:"
echo -e "   ${GREEN}mmdc -i diagram.mmd -o output.svg -t dark${NC}"
echo ""
echo "4. Batch processing:"
echo -e "   ${GREEN}mmdc -i diagrams/*.mmd${NC}"
echo ""
echo "5. With puppeteer config (for --no-sandbox):"
echo -e "   ${GREEN}mmdc -p puppeteer-config.json -i diagram.mmd -o output.png${NC}"
echo ""
echo "6. Using the mmdc-quick alias (if configured):"
echo -e "   ${GREEN}mmdc-quick -i diagram.mmd -o output.png${NC}"
echo ""
echo -e "${YELLOW}Troubleshooting:${NC}"
echo ""
echo "- If browser issues occur, install Chrome globally:"
echo -e "   ${GREEN}npx @puppeteer/browsers install chrome@stable --path ~/.cache/puppeteer${NC}"
echo ""
echo "- For sandbox issues, create puppeteer-config.json:"
echo -e "   ${GREEN}mkdir -p ~/.config/mermaid${NC}"
echo -e "   ${GREEN}echo '{\"args\": [\"--no-sandbox\"]}' > ~/.config/mermaid/puppeteer-config.json${NC}"
echo -e "   ${GREEN}alias mmdc-quick='mmdc -p ~/.config/mermaid/puppeteer-config.json'${NC}"
echo ""
echo "- Install additional browser versions:"
echo -e "   ${GREEN}npx @puppeteer/browsers install chromium@latest --path ~/.cache/puppeteer${NC}"
echo -e "   ${GREEN}npx @puppeteer/browsers install firefox@stable --path ~/.cache/puppeteer${NC}"
echo ""
echo "- List installed browsers:"
echo -e "   ${GREEN}npx @puppeteer/browsers list --path ~/.cache/puppeteer${NC}"
echo ""
echo "- Clear browser cache:"
echo -e "   ${GREEN}rm -rf ~/.cache/puppeteer${NC}"
echo ""
echo "For more information:"
echo -e "   ${GREEN}mmdc --help${NC}"
echo -e "   ${BLUE}https://github.com/mermaid-js/mermaid-cli${NC}"
echo -e "   ${BLUE}https://pptr.dev/browsers-api${NC}"
echo ""
