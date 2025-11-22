#!/bin/bash
# Script to copy test graphs from domain_flow repository

set -e

# Check if domain_flow path is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <path_to_domain_flow_repo>"
    echo "Example: $0 ~/dev/domain_flow"
    exit 1
fi

DOMAIN_FLOW_PATH="$1"
KPU_GRAPHS_DIR="$(dirname "$0")/../test_graphs"

# Validate domain_flow path
if [ ! -d "$DOMAIN_FLOW_PATH/data/dfg" ]; then
    echo "Error: $DOMAIN_FLOW_PATH/data/dfg directory not found"
    echo "Please provide valid path to domain_flow repository"
    exit 1
fi

echo "Copying domain_flow test graphs..."
echo "  Source: $DOMAIN_FLOW_PATH/data/dfg"
echo "  Dest:   $KPU_GRAPHS_DIR"

# Create directories
mkdir -p "$KPU_GRAPHS_DIR/simple"
mkdir -p "$KPU_GRAPHS_DIR/networks"
mkdir -p "$KPU_GRAPHS_DIR/benchmarks"

# Copy graphs to appropriate subdirectories
# Simple graphs (basic operators)
if [ -f "$DOMAIN_FLOW_PATH/data/dfg/matmul.dfg" ]; then
    cp "$DOMAIN_FLOW_PATH/data/dfg/matmul.dfg" "$KPU_GRAPHS_DIR/simple/"
    echo "  ✓ Copied matmul.dfg"
fi

# Network graphs
for graph in mlp resnet; do
    if [ -f "$DOMAIN_FLOW_PATH/data/dfg/${graph}.dfg" ]; then
        cp "$DOMAIN_FLOW_PATH/data/dfg/${graph}.dfg" "$KPU_GRAPHS_DIR/networks/"
        echo "  ✓ Copied ${graph}.dfg"
    fi
done

# Benchmark graphs (larger models)
for graph in mobilenet_v1 mobilenet_v2 efficientnet bert; do
    if [ -f "$DOMAIN_FLOW_PATH/data/dfg/${graph}.dfg" ]; then
        cp "$DOMAIN_FLOW_PATH/data/dfg/${graph}.dfg" "$KPU_GRAPHS_DIR/benchmarks/"
        echo "  ✓ Copied ${graph}.dfg"
    fi
done

# Copy any other available graphs
echo ""
echo "Other available graphs in domain_flow:"
ls -1 "$DOMAIN_FLOW_PATH/data/dfg/"*.dfg 2>/dev/null | xargs -n1 basename || echo "  (none found)"

echo ""
echo "Copy complete!"
echo "Test graphs are now available in: $KPU_GRAPHS_DIR"
