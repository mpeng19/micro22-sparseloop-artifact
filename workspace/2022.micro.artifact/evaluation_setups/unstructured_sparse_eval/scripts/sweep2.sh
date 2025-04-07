#!/bin/bash

# Enable debugging
set -x

# Define workloads
WORKLOADS=("resnet50_conv1" "alexnet_conv1_sparse" "mobilenet_conv1_sparse")

# Get the absolute path to the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BASE_DIR="$(dirname "$SCRIPT_DIR")"

# Print debugging information
echo "Script directory: $SCRIPT_DIR"
echo "Base directory: $BASE_DIR"
echo "Current directory: $(pwd)"
echo "Contents of script directory:"
ls -la "$SCRIPT_DIR"
echo "Contents of base directory:"
ls -la "$BASE_DIR"

# Verify timeloop-mapper exists
which timeloop-mapper
timeloop-mapper --version

# Run evaluations for each workload
for workload in "${WORKLOADS[@]}"; do
    echo "Processing $workload with ideal_sparse_tensor_core..."
    
    # Create output directory
    mkdir -p "${BASE_DIR}/outputs/ideal_${workload}"
    
    # Verify all required files exist
    echo "Checking required files:"
    ls -la "${BASE_DIR}/arch/ideal_sparse_tensor_core.yaml"
    ls -la "${BASE_DIR}/dataflow/weight_stationary.yaml"
    ls -la "${BASE_DIR}/sparse-opt/ideal.yaml"
    ls -la "${BASE_DIR}/workload/${workload}.yaml"
    ls -la "${BASE_DIR}/mapper/mapper.yaml"
    ls -la "${BASE_DIR}/ert_art/ERT_ideal.yaml"
    ls -la "${BASE_DIR}/ert_art/ART.yaml"
    
    # Run timeloop-mapper with absolute paths
    timeloop-mapper \
        "${BASE_DIR}/arch/ideal_sparse_tensor_core.yaml" \
        "${BASE_DIR}/dataflow/weight_stationary.yaml" \
        "${BASE_DIR}/sparse-opt/ideal.yaml" \
        "${BASE_DIR}/workload/${workload}.yaml" \
        "${BASE_DIR}/mapper/mapper.yaml" \
        "${BASE_DIR}/ert_art/ERT_ideal.yaml" \
        "${BASE_DIR}/ert_art/ART.yaml" \
        -o "${BASE_DIR}/outputs/ideal_${workload}" 
done 