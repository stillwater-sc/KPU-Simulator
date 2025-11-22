#!/bin/bash
# Run KPU-simulator tests (excluding external domain_flow tests)

set -e

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Running KPU-simulator tests...${NC}"
echo ""

# Check if build directory exists
if [ ! -d "build" ]; then
    echo -e "${RED}Error: build directory not found${NC}"
    echo "Run: cmake -B build -GNinja"
    exit 1
fi

# Exclude pattern for domain_flow tests
EXCLUDE_PATTERN="^(dsp_|nla_|dfa_|dnn_|ctl_|cnn_)"

# Run tests
if ctest --test-dir build -E "$EXCLUDE_PATTERN" --output-on-failure; then
    echo ""
    echo -e "${GREEN}✅ All KPU-simulator tests passed!${NC}"
    exit 0
else
    echo ""
    echo -e "${RED}❌ Some tests failed${NC}"
    exit 1
fi
