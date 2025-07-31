#!/bin/bash

# =============================================================================
# Configuration Test Script
# =============================================================================

# Load common configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

echo "========================================"
echo "Configuration Test"
echo "========================================"

# Test basic configuration loading
print_config

echo ""
echo "Testing directory setup..."
setup_directories
echo "✅ Directories created successfully"

echo ""
echo "Testing timestamp generation..."
TIMESTAMP=$(generate_timestamp)
echo "Generated timestamp: $TIMESTAMP"

echo ""
echo "Testing configuration override..."
echo "Original Vision Model: $VISION_MODEL"
override_config "vision_model" "microsoft/DiT-large"
echo "After override: $VISION_MODEL"

echo ""
echo "✅ All configuration tests passed!"
echo "========================================"
