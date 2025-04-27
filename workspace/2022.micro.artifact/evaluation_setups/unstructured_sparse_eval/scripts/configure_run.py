# File: workspace/2022.micro.artifact/evaluation_setups/unstructured_sparse_eval/scripts/configure_run.py

import argparse
import yaml
from pathlib import Path
import sys

# --- Layer Name to Workload Shape Mapping ---
# This dictionary maps the short layer names used in arguments
# to the actual workload shape YAML files.
# !!! Important: You need to expand this mapping with all the layers you intend to use!
LAYER_WORKLOAD_MAP = {
    # ResNet50 Examples (Corrected Paths to specific structured files)
    # Map generic layer name to a SPECIFIC workload file representing its shape+structure
    "resnet50_conv1": "resnet/resnet50_conv1_2x4.yaml",
    # Point to the newly generated _2x4 files
    "resnet50_conv2_block1_1": "resnet/resnet50_conv2_block1_1_2x4.yaml",
    "resnet50_conv2_block1_2": "resnet/resnet50_conv2_block1_2_2x4.yaml",
    
    # Comment out or remove other unused/unverified entries for clarity
    # "resnet50_conv2_block1_3": "resnet/resnet50_conv2_block1_3.yaml", 
    # "resnet50_conv2_block1_0_shortcut": "resnet/resnet50_conv2_block1_0_shortcut.yaml",
    # "resnet50_conv3_block1_1": "resnet/resnet50_conv3_block1_1.yaml",
    
    # BERT Examples (Need verification based on available files in workload/bert/)
    # "bert_encoder_0_attention_self_query": "bert/bert_encoder_0_attention_self_query_????.yaml",
    # "bert_encoder_0_attention_self_key": "bert/bert_encoder_0_attention_self_key_????.yaml",
    # "bert_encoder_0_attention_self_value": "bert/bert_encoder_0_attention_self_value_????.yaml",
    # "bert_encoder_0_attention_output_dense": "bert/bert_encoder_0_attention_output_dense_????.yaml",
}

# --- Known Architecture Files (for validation, optional but recommended) ---
# List known/valid architecture filenames to catch typos.
# Get these from the `arch/` directory.
KNOWN_ARCH_FILES = [
    "dynamic_sparsity_arch.yaml",
    "structured_2_4_sparse_tensor_core.yaml",
    "unstructured_sparse_tensor_core.yaml",
    "ideal_sparse_tensor_core.yaml",
    # Add any other valid architecture files here
]

def main():
    parser = argparse.ArgumentParser(description='Generate a model sequence YAML file for Timeloop simulation.')
    
    parser.add_argument(
        '--model-name',
        required=True,
        type=str,
        help='A descriptive name for this specific model configuration/run (e.g., "resnet50_mixed_config_A"). This name will be embedded in the YAML.'
    )
    parser.add_argument(
        '--output-yaml',
        required=True,
        type=str,
        help='Path where the generated sequence YAML file should be saved (e.g., "generated_configs/resnet50_config_A.yaml").'
    )
    parser.add_argument(
        '--layer',
        required=True,
        action='append',
        metavar='LAYER_NAME:ARCH_FILE.yaml',
        help='Specify a layer and its desired sparsity architecture file. Format: "layer_name:architecture_file.yaml". Use this argument multiple times for multiple layers.'
    )

    args = parser.parse_args()

    # Determine base directory relative to this script
    script_dir = Path(__file__).parent.resolve()
    base_dir = script_dir.parent 
    output_yaml_path = (base_dir / args.output_yaml).resolve()

    # Prepare the data structure for YAML output
    sequence_data = {
        'model_name': args.model_name,
        'layers': []
    }

    print(f"Generating sequence YAML for model: {args.model_name}")
    print(f"Output file will be: {output_yaml_path}")

    valid_config = True
    for layer_spec in args.layer:
        try:
            layer_name, arch_filename = layer_spec.split(':', 1)
        except ValueError:
            print(f"Error: Invalid format for --layer argument: '{layer_spec}'. Expected 'layer_name:arch_file.yaml'", file=sys.stderr)
            valid_config = False
            continue

        # Validate layer name and get workload
        workload_shape_file = LAYER_WORKLOAD_MAP.get(layer_name)
        if not workload_shape_file:
            print(f"Error: Unknown layer name '{layer_name}'. Please add it to the LAYER_WORKLOAD_MAP in the script.", file=sys.stderr)
            valid_config = False
            continue
            
        # Validate arch filename format (simple check)
        if not arch_filename.endswith(".yaml"):
             print(f"Warning: Architecture filename '{arch_filename}' for layer '{layer_name}' does not end with .yaml.", file=sys.stderr)
             # Continue, but maybe add stricter validation later

        # Optional: Validate against known arch files
        if arch_filename not in KNOWN_ARCH_FILES:
             print(f"Warning: Architecture filename '{arch_filename}' for layer '{layer_name}' is not in the KNOWN_ARCH_FILES list. Please ensure it exists in the 'arch/' directory.", file=sys.stderr)

        # Add layer info to the list
        sequence_data['layers'].append({
            'name': layer_name,
            'workload': workload_shape_file,  # Use the mapped workload path
            'sparsity_arch': arch_filename      # Use the user-provided arch filename
        })
        print(f"  + Added layer: {layer_name} (Workload: {workload_shape_file}, Arch: {arch_filename})")

    if not valid_config:
        print("\nErrors found in configuration. Aborting YAML generation.", file=sys.stderr)
        sys.exit(1)
        
    if not sequence_data['layers']:
        print("\nError: No valid layers were specified.", file=sys.stderr)
        sys.exit(1)

    # Ensure the output directory exists
    output_yaml_path.parent.mkdir(parents=True, exist_ok=True)

    # Write the YAML file
    try:
        with open(output_yaml_path, 'w') as f:
            yaml.dump(sequence_data, f, default_flow_style=False, sort_keys=False)
        print(f"\nSuccessfully generated sequence YAML: {output_yaml_path}")
    except Exception as e:
        print(f"\nError writing YAML file {output_yaml_path}: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 