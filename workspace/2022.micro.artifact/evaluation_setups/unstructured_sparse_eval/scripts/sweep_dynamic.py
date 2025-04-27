# File: workspace/2022.micro.artifact/evaluation_setups/unstructured_sparse_eval/scripts/sweep_dynamic.py

import argparse
import os
import subprocess
import sys
from pathlib import Path
import yaml  # Import YAML library

# --- Configuration ---
# Define the base directory relative to this script's location
SCRIPT_DIR = Path(__file__).parent.resolve()
BASE_DIR = SCRIPT_DIR.parent

# Construct absolute paths to key directories
ARCH_DIR = BASE_DIR / "arch"
SPARSE_OPT_DIR = BASE_DIR / "sparse-opt"
WORKLOAD_DIR = BASE_DIR / "workload"
MAPPER_DIR = BASE_DIR / "mapper"
ERT_ART_DIR = BASE_DIR / "ert_art"
OUTPUT_DIR_BASE = BASE_DIR / "outputs" # Base output directory

print(f"BASE_DIR: {BASE_DIR}")

# Define the common configuration files (These might be constant across layers)
# Note: Arch file is now layer-specific
SPARSE_OPT_FILE = SPARSE_OPT_DIR / "dynamic_sparsity_opt.yaml" # Assuming this is common
DATAFLOW_FILE = BASE_DIR / "dataflow/weight_stationary.yaml" # Assuming this is common
MAPPER_FILE = MAPPER_DIR / "mapper_unstructured.yaml"      # Assuming this is common
ERT_FILE = ERT_ART_DIR / "ERT_dynamic_sparsity.yaml"       # Assuming this is common
ART_FILE = ERT_ART_DIR / "ART_dynamic_sparsity.yaml"       # Assuming this is common

# Modified function to accept arch_file_path
def run_single_timeloop(arch_file_path, workload_file_path, output_path):
    """ Runs a single Timeloop simulation for the given workload, arch, and output path. """

    layer_name = output_path.name # Infer layer name from output dir
    print("-" * 60)
    print(f"Running Timeloop for Layer: {layer_name}")
    print(f"  Workload: {workload_file_path.relative_to(BASE_DIR)}")
    print(f"  Arch:     {arch_file_path.relative_to(BASE_DIR)}")
    print(f"  Output:   {output_path.relative_to(BASE_DIR)}")

    if not workload_file_path.exists():
        print(f"!!! ERROR: Workload file not found: {workload_file_path}. Skipping layer {layer_name}.")
        print("-" * 60)
        return False
    if not arch_file_path.exists():
        print(f"!!! ERROR: Architecture file not found: {arch_file_path}. Skipping layer {layer_name}.")
        print("-" * 60)
        return False

    # Create output directory for this run
    output_path.mkdir(parents=True, exist_ok=True)

    # Construct the timeloop-mapper command (using the provided arch_file_path)
    command = [
        "timeloop-mapper",
        str(arch_file_path), # Use the specific arch file for this layer
        str(DATAFLOW_FILE),
        str(SPARSE_OPT_FILE),
        str(workload_file_path),
        str(MAPPER_FILE),
        str(ERT_FILE),
        str(ART_FILE),
        "-o", str(output_path)
    ]

    print(f"Executing: {' '.join(command)}")

    success = False
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True, cwd=BASE_DIR)
        print(f"Timeloop completed successfully for layer {layer_name}.")
        success = True
    except subprocess.CalledProcessError as e:
        print(f"!!! ERROR: Timeloop execution failed for layer {layer_name}.")
        print(f"Return Code: {e.returncode}")
        print("STDERR:\n", e.stderr)
    except FileNotFoundError:
         print("!!! ERROR: timeloop-mapper command not found. Is it in your PATH?")
         return False # Critical error, stop the script
    except Exception as e:
        print(f"!!! ERROR: An unexpected error occurred running Timeloop for layer {layer_name}: {e}")

    print("-" * 60)
    return success

def main():
    parser = argparse.ArgumentParser(description='Run Timeloop simulations for a sequence of layers defined in a YAML file.')
    parser.add_argument(
        'sequence_file',
        type=str,
        help='Path to the model sequence YAML file (e.g., "model_sequence_example.yaml").'
    )

    args = parser.parse_args()
    sequence_file_path = (BASE_DIR / args.sequence_file).resolve()

    # --- Check common files first ---
    common_files = [SPARSE_OPT_FILE, DATAFLOW_FILE, MAPPER_FILE, ERT_FILE, ART_FILE]
    all_common_files_exist = True
    for f in common_files:
        if not f.exists():
            print(f"Error: Common configuration file not found: {f}", file=sys.stderr)
            all_common_files_exist = False
    if not sequence_file_path.exists():
         print(f"Error: Sequence file not found: {sequence_file_path}", file=sys.stderr)
         all_common_files_exist = False

    if not all_common_files_exist:
        sys.exit(1)
    print("Common configuration files and sequence file checked.")

    # --- Load Sequence --- 
    try:
        with open(sequence_file_path, 'r') as f:
            sequence_config = yaml.safe_load(f)
        model_name = sequence_config.get('model_name', Path(args.sequence_file).stem) # Use filename stem as fallback
        layers = sequence_config.get('layers', [])
        if not layers:
             print(f"Error: No 'layers' defined in {sequence_file_path}", file=sys.stderr)
             sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing sequence file {sequence_file_path}: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
         print(f"Error reading sequence file {sequence_file_path}: {e}", file=sys.stderr)
         sys.exit(1)

    print(f"Loaded sequence for model: {model_name}")
    print(f"Found {len(layers)} layers to process.")

    # --- Run Simulations for Each Layer --- 
    success_count = 0
    total_layers = len(layers)
    model_output_base = (OUTPUT_DIR_BASE / model_name).resolve()

    for i, layer in enumerate(layers):
        layer_name = layer.get('name')
        workload_rel_path = layer.get('workload')
        arch_rel_path = layer.get('sparsity_arch')

        if not all([layer_name, workload_rel_path, arch_rel_path]):
            print(f"Warning: Skipping layer entry {i+1} due to missing 'name', 'workload', or 'sparsity_arch'. Entry: {layer}")
            continue

        workload_file_path = (WORKLOAD_DIR / workload_rel_path).resolve()
        arch_file_path = (ARCH_DIR / arch_rel_path).resolve()
        layer_output_path = (model_output_base / layer_name).resolve()

        if run_single_timeloop(arch_file_path, workload_file_path, layer_output_path):
            success_count += 1

    # --- Summary --- 
    print("="*60)
    print(f"Sequence processing complete for model: {model_name}")
    print(f"Successfully simulated {success_count} out of {total_layers} layers.")
    print(f"Outputs saved under: {model_output_base}")
    print("="*60)

    if success_count == total_layers:
        sys.exit(0) # All successful
    else:
        sys.exit(1) # Some failures

if __name__ == "__main__":
    main() 