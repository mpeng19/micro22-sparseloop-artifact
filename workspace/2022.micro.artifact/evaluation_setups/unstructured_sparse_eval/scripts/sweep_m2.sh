#!/bin/bash

# Define base directories (using absolute paths)
BASE_DIR="/home/workspace/2022.micro.artifact/evaluation_setups/unstructured_sparse_eval"
ARCH_DIR="$BASE_DIR/arch"
SPARSE_OPT_DIR="$BASE_DIR/sparse-opt"
MAPPER_DIR="$BASE_DIR/mapper"
DATAFLOW_DIR="$BASE_DIR/dataflow"
WORKLOAD_DIR="$BASE_DIR/workload"
OUTPUT_DIR="$BASE_DIR/outputs"
ERT_ART_DIR="$BASE_DIR/ert_art"

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# --- Configuration specific to structured_2_4 --- 
ARCH="structured_2_4"
ARCH_FILE="$ARCH_DIR/${ARCH}_sparse_tensor_core.yaml"
SPARSE_OPT_FILE="$SPARSE_OPT_DIR/${ARCH}.yaml"
DATAFLOW_FILE="$DATAFLOW_DIR/weight_stationary.yaml"
MAPPER_FILE="$MAPPER_DIR/mapper.yaml"
ERT_FILE="$ERT_ART_DIR/ERT.yaml" # Using the base ERT file
ART_FILE="$ERT_ART_DIR/ART.yaml"

# Define structured workloads to evaluate
declare -a EVAL_WORKLOADS=(
    "resnet50_conv1_structured"
    "alexnet_conv1_structured"
    "mobilenet_conv1_structured"
)

# Check if common files exist (optional, can be removed if paths are certain)
echo "Checking for required configuration files..."
# ... (can add checks for ARCH_FILE, SPARSE_OPT_FILE, etc. if needed)

# --- Run sweeps for structured_2_4 --- 
echo "Evaluating architecture: $ARCH"

for workload in "${EVAL_WORKLOADS[@]}"; do
    # Extract base workload name for output directory
    # Example: resnet50_conv1_structured -> resnet50_conv1
    BASE_WORKLOAD=$(echo $workload | sed 's/_structured$//') 
    OUT_DIR="$OUTPUT_DIR/${ARCH}_${BASE_WORKLOAD}" # Changed output dir naming
    
    echo "Running Timeloop for $workload with $ARCH architecture..."
    mkdir -p $OUT_DIR
    
    # Set workload file path
    WORKLOAD_FILE="$WORKLOAD_DIR/${workload}.yaml"
    
    # Check if workload file exists (optional)
    if [ ! -f "$WORKLOAD_FILE" ]; then
        echo "Error: Workload file not found: $WORKLOAD_FILE"
        continue
    fi
    
    # Echo all files being used (for verification)
    echo "Using files:"
    echo "  Architecture: $ARCH_FILE"
    echo "  Dataflow: $DATAFLOW_FILE"
    echo "  Sparse Optimization: $SPARSE_OPT_FILE"
    echo "  Workload: $WORKLOAD_FILE"
    echo "  Mapper: $MAPPER_FILE"
    echo "  ERT: $ERT_FILE"
    echo "  ART: $ART_FILE"
    
    # Run Timeloop Mapper
    timeloop-mapper $ARCH_FILE \
                  $DATAFLOW_FILE \
                  $SPARSE_OPT_FILE \
                  $WORKLOAD_FILE \
                  $MAPPER_FILE \
                  $ERT_FILE \
                  $ART_FILE \
                  -o $OUT_DIR
                 
    if [ $? -eq 0 ]; then
        echo "Successful run for $workload with $arch"
    else
        echo "Failed to run $workload with $arch"
    fi
done

echo "Sweep complete!" 