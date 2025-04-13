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

# Define architectures to evaluate
declare -a ARCHS=(
    "structured_2_4"
)

# Define workloads to evaluate
declare -a WORKLOADS=(
    "resnet50_conv1"
    "alexnet_conv1_sparse"
    "mobilenet_conv1_sparse"
)

# Define structured workloads
declare -a STRUCT_WORKLOADS=(
    "resnet50_conv1_structured"
    "alexnet_conv1_structured"
    "mobilenet_conv1_structured"
)

# Check if required files exist
echo "Checking for required configuration files..."
if [ ! -f "$DATAFLOW_DIR/weight_stationary.yaml" ]; then
    echo "Error: Dataflow file not found: $DATAFLOW_DIR/weight_stationary.yaml"
    exit 1
fi

if [ ! -f "$MAPPER_DIR/mapper.yaml" ]; then
    echo "Error: Mapper file not found: $MAPPER_DIR/mapper.yaml"
    exit 1
fi

# Run sweeps
for arch in "${ARCHS[@]}"; do
    echo "Evaluating architecture: $arch"
    
    # Set architecture file and sparse optimization file based on architecture
    ARCH_FILE="$ARCH_DIR/${arch}_sparse_tensor_core.yaml"
    SPARSE_OPT_FILE="$SPARSE_OPT_DIR/${arch}.yaml"
    DATAFLOW_FILE="$DATAFLOW_DIR/weight_stationary.yaml"
    MAPPER_FILE="$MAPPER_DIR/mapper.yaml"
    
    # Set appropriate ERT file based on architecture
    if [ "$arch" == "ideal" ]; then
        ERT_FILE="$ERT_ART_DIR/ERT_ideal.yaml"
    else
        ERT_FILE="$ERT_ART_DIR/ERT_sparse.yaml"
    fi
    
    # Set ART file
    ART_FILE="$ERT_ART_DIR/ART.yaml"
    
    # Check if architecture file exists
    if [ ! -f "$ARCH_FILE" ]; then
        echo "Error: Architecture file not found: $ARCH_FILE"
        continue
    fi
    
    # Check if sparse optimization file exists
    if [ ! -f "$SPARSE_OPT_FILE" ]; then
        echo "Error: Sparse optimization file not found: $SPARSE_OPT_FILE"
        continue
    fi
    
    # Check if ERT and ART files exist
    if [ ! -f "$ERT_FILE" ]; then
        echo "Error: ERT file not found: $ERT_FILE"
        continue
    fi
    
    if [ ! -f "$ART_FILE" ]; then
        echo "Error: ART file not found: $ART_FILE"
        continue
    fi
    
    # Use appropriate workload files based on architecture
    if [ "$arch" == "structured_2_4" ]; then
        EVAL_WORKLOADS=("${STRUCT_WORKLOADS[@]}")
    else
        EVAL_WORKLOADS=("${WORKLOADS[@]}")
    fi
    
    for workload in "${EVAL_WORKLOADS[@]}"; do
        # Extract base workload name for output directory
        BASE_WORKLOAD=$(echo $workload | sed 's/_structured//')
        OUT_DIR="$OUTPUT_DIR/${BASE_WORKLOAD}_${arch}"
        
        echo "Running Timeloop for $workload with $arch architecture..."
        mkdir -p $OUT_DIR
        
        # Check if workload file exists
        WORKLOAD_FILE="$WORKLOAD_DIR/${workload}.yaml"
        if [ ! -f "$WORKLOAD_FILE" ]; then
            echo "Error: Workload file not found: $WORKLOAD_FILE"
            continue
        fi
        
        # Echo all files being used
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
done

echo "Sweep complete!" 