#!/bin/bash

# Fix Docker credential helper issue
# Usage: ./script/fix-docker-credentials.sh

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Fix Docker Credential Helper${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

CONFIG_FILE="$HOME/.docker/config.json"

# Create backup
if [ -f "$CONFIG_FILE" ]; then
    echo -e "${YELLOW}Backing up existing config...${NC}"
    cp "$CONFIG_FILE" "$CONFIG_FILE.backup.$(date +%Y%m%d_%H%M%S)"
    echo -e "${GREEN}✓${NC} Backup created: $CONFIG_FILE.backup.*"
    echo ""
fi

# Check current config
echo -e "${BLUE}Current Docker config:${NC}"
cat "$CONFIG_FILE" 2>/dev/null || echo "{}"
echo ""

# Remove credsStore
echo -e "${YELLOW}Removing credsStore setting...${NC}"
if [ -f "$CONFIG_FILE" ]; then
    # Use jq if available, otherwise use sed
    if command -v jq &> /dev/null; then
        jq 'del(.credsStore)' "$CONFIG_FILE" > "$CONFIG_FILE.tmp" && mv "$CONFIG_FILE.tmp" "$CONFIG_FILE"
    else
        # Remove credsStore line (simple method)
        sed -i.bak '/"credsStore":/d' "$CONFIG_FILE"
        rm -f "$CONFIG_FILE.bak"
    fi
fi

echo -e "${GREEN}✓${NC} credsStore setting removed"
echo ""

# Check updated config
echo -e "${BLUE}Updated Docker config:${NC}"
cat "$CONFIG_FILE"
echo ""

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✓ Fix completed!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Now try again:"
echo "  ./script/test-docker.sh"
echo ""
echo "If the problem persists, restart Docker Desktop:"
echo "  1. Quit Docker Desktop completely"
echo "  2. Restart Docker Desktop"
echo "  3. Try building again"
