#!/bin/bash

# Sweep script for STRUCTURED 2:4 sparsity

# Enable debugging (optional)
# set -x

# Define workloads
WORKLOADS=("resnet50_conv1" "alexnet_conv1" "mobilenet_conv1")

# Get the absolute path to the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_BASE_DIR="${BASE_DIR}/outputs"
MAPPINGS_DIR="${BASE_DIR}/mappings_found"

# Create base output and mappings directories
mkdir -p "${OUTPUT_BASE_DIR}"
mkdir -p "${MAPPINGS_DIR}"

# Configuration files for STRUCTURED 2:4
ARCH_FILE="${BASE_DIR}/arch/structured_2_4_tensor_core.yaml"
DATAFLOW_FILE="${BASE_DIR}/dataflow/structured_2_4_dataflow.yaml" # Using Ampere constraints
SPARSE_OPT_FILE="${BASE_DIR}/sparse-opt/structured_2_4.yaml"
ERT_FILE="${BASE_DIR}/ert_art/ERT_structured.yaml"
ART_FILE="${BASE_DIR}/ert_art/ART_structured.yaml"
MAPPER_FILE="${BASE_DIR}/mapper/mapper.yaml"
OUTPUT_SUFFIX="structured_2_4"
STORED_MAP_DIR="${MAPPINGS_DIR}/${OUTPUT_SUFFIX}"

mkdir -p "${STORED_MAP_DIR}" # Create specific map dir

echo "*************************************************************"
echo "Processing Sparsity Type: ${OUTPUT_SUFFIX}"
echo "*************************************************************"
echo "Using Configs:"
echo "  Arch: ${ARCH_FILE}"
echo "  Dataflow: ${DATAFLOW_FILE}"
echo "  Sparse Opt: ${SPARSE_OPT_FILE}"
echo "  ERT: ${ERT_FILE}"
echo "  ART: ${ART_FILE}"
echo "  Mapper: ${MAPPER_FILE}"
echo "-------------------------------------------------------------"

# Run evaluations for each workload
for workload in "${WORKLOADS[@]}"; do
    echo "============================================================="
    echo "Processing workload: ${workload} (${OUTPUT_SUFFIX})"
    echo "============================================================="

    # Define paths
    MAPPING_SEARCH_OUTPUT_DIR="${OUTPUT_BASE_DIR}/_tmp_mapping_${OUTPUT_SUFFIX}_${workload}"
    FOUND_MAP_TMP_PATH="${MAPPING_SEARCH_OUTPUT_DIR}/timeloop-mapper.map.yaml"
    STORED_MAP_PATH="${STORED_MAP_DIR}/${workload}.map.yaml"
    FINAL_OUTPUT_DIR="${OUTPUT_BASE_DIR}/${OUTPUT_SUFFIX}_${workload}"

    # Create directories
    mkdir -p "${MAPPING_SEARCH_OUTPUT_DIR}"
    # STORED_MAP_DIR created above loop
    mkdir -p "${FINAL_OUTPUT_DIR}"

    # --- Stage 1: Mapping Search ---
    echo "--- Stage 1: Running Mapping Search for ${workload} (${OUTPUT_SUFFIX})... ---"
    timeloop-mapper \
        "${ARCH_FILE}" \
        "${DATAFLOW_FILE}" \
        "${SPARSE_OPT_FILE}" \
        "${BASE_DIR}/workload/${workload}.yaml" \
        "${MAPPER_FILE}" \
        "${ERT_FILE}" \
        "${ART_FILE}" \
        -o "${MAPPING_SEARCH_OUTPUT_DIR}"

    # Check if mapping was found and store it
    if [[ -f "${FOUND_MAP_TMP_PATH}" ]]; then
        echo "INFO: Mapping found for ${workload} (${OUTPUT_SUFFIX}). Storing to ${STORED_MAP_PATH}"
        cp "${FOUND_MAP_TMP_PATH}" "${STORED_MAP_PATH}"
    else
        echo "ERROR: Mapping search failed for ${workload} (${OUTPUT_SUFFIX}). Skipping Stage 2."
        rm -rf "${MAPPING_SEARCH_OUTPUT_DIR}" # Clean up failed search dir
        continue # Skip to the next workload
    fi

    # Clean up temporary mapping search directory after successful copy
    echo "INFO: Cleaning up temporary mapping directory ${MAPPING_SEARCH_OUTPUT_DIR}"
    rm -rf "${MAPPING_SEARCH_OUTPUT_DIR}"

    # --- Stage 2: Model Evaluation using Found Mapping ---
    echo "--- Stage 2: Running Model Evaluation for ${workload} (${OUTPUT_SUFFIX}) using found mapping... ---"
    timeloop-mapper \
        "${ARCH_FILE}" \
        "${DATAFLOW_FILE}" \
        "${SPARSE_OPT_FILE}" \
        "${BASE_DIR}/workload/${workload}.yaml" \
        "${MAPPER_FILE}" \
        "${STORED_MAP_PATH}" \
        "${ERT_FILE}" \
        "${ART_FILE}" \
        -o "${FINAL_OUTPUT_DIR}"

    echo "INFO: Finished processing ${workload} (${OUTPUT_SUFFIX}). Results in ${FINAL_OUTPUT_DIR}"

done # End workload loop

echo "============================================================="
echo "All STRUCTURED 2:4 workloads processed."
echo "=============================================================" 