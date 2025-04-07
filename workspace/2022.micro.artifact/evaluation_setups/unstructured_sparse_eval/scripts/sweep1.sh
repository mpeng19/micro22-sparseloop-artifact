#!/bin/bash

# Define workloads
WORKLOADS=("resnet50_conv1" "alexnet_conv1_sparse" "mobilenet_conv1_sparse")

# Get the absolute path to the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BASE_DIR="$(dirname "$SCRIPT_DIR")"

# Run evaluations for each workload
for workload in "${WORKLOADS[@]}"; do
    echo "Processing $workload with unstructured_sparse_tensor_core..."
    
    # Create output directory
    mkdir -p "${BASE_DIR}/outputs/unstructured_${workload}"
    
    # Run timeloop-mapper with absolute paths
    timeloop-mapper \
        "${BASE_DIR}/arch/unstructured_sparse_tensor_core.yaml" \
        "${BASE_DIR}/dataflow/weight_stationary.yaml" \
        "${BASE_DIR}/sparse-opt/unstructured.yaml" \
        "${BASE_DIR}/workload/${workload}.yaml" \
        "${BASE_DIR}/mapper/mapper.yaml" \
        "${BASE_DIR}/ert_art/ERT.yaml" \
        "${BASE_DIR}/ert_art/ART.yaml" \
        -o "${BASE_DIR}/outputs/unstructured_${workload}" 
done 