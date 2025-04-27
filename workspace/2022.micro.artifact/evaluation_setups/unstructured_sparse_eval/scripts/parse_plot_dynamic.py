# File: workspace/2022.micro.artifact/evaluation_setups/unstructured_sparse_eval/scripts/parse_plot_dynamic.py

import argparse
import os
import re
import sys
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import yaml # Need yaml if we also parse the sequence file

# Define the order of structures for consistent plotting/reporting (still useful for categorical type)
# We might not need this if we derive sparsity from arch filename
# STRUCTURE_ORDER = ["1:2", "1:4", "2:4", "2:8", "4:8", "4:12", "4:16", "unstructured", "ideal"] # Added fallbacks

def extract_metrics_from_stats(stats_file_path):
    """Extracts performance metrics from a timeloop stats file."""
    metrics = {'cycles': None, 'energy': None, 'utilization': None}
    if not stats_file_path.exists():
        print(f"Warning: Stats file not found at {stats_file_path}", file=sys.stderr)
        return metrics # Return empty metrics

    try:
        with open(stats_file_path, 'r') as f:
            content = f.read()

            # Extract cycles
            cycles_match = re.search(r'^\s*Cycles:\s*(\d+)', content, re.MULTILINE)
            if cycles_match:
                metrics['cycles'] = int(cycles_match.group(1))

            # Extract energy (convert uJ to pJ if present)
            energy_match_uj = re.search(r'^\s*Energy:\s*([\d.eE+-]+)\s*uJ', content, re.MULTILINE)
            energy_match_pj = re.search(r'^\s*Energy:\s*([\d.eE+-]+)\s*pJ', content, re.MULTILINE)
            if energy_match_uj:
                metrics['energy'] = float(energy_match_uj.group(1)) * 1_000_000 # uJ to pJ
            elif energy_match_pj:
                 metrics['energy'] = float(energy_match_pj.group(1)) # Already pJ
            else:
                 print(f"Warning: Could not find Energy (uJ or pJ) in {stats_file_path}")


            # Extract utilization (as percentage)
            util_match = re.search(r'^\s*Utilization:\s*([\d.]+)\s*%', content, re.MULTILINE)
            if util_match:
                metrics['utilization'] = float(util_match.group(1))
            else: # Fallback if % sign is missing
                 util_match_frac = re.search(r'^\s*Utilization:\s*([\d.]+)\s*$', content, re.MULTILINE)
                 if util_match_frac:
                    util_val = float(util_match_frac.group(1))
                    if 0 <= util_val <= 1.01: # Allow slightly > 1 for rounding
                         metrics['utilization'] = util_val * 100
                    else:
                        metrics['utilization'] = util_val # Assume it was already a percentage
                 else:
                    print(f"Warning: Could not find Utilization in {stats_file_path}")

    except Exception as e:
        print(f"Warning: Error parsing stats file {stats_file_path}: {e}", file=sys.stderr)
    return metrics

# Function to extract sparsity type from arch filename (example)
def get_sparsity_from_arch(arch_filename):
    if not arch_filename: return "unknown"
    name_lower = arch_filename.lower()
    if "unstructured" in name_lower: return "unstructured"
    if "ideal" in name_lower: return "ideal"
    match = re.search(r'(\d+)[x:_](\d+)', name_lower) # Look for NxM, N:M, or N_M
    if match: return f"{match.group(1)}:{match.group(2)}" # Always format output with :
    return "other" # Fallback

def main():
    parser = argparse.ArgumentParser(description='Parse results from a multi-layer Timeloop simulation output directory structure.')
    parser.add_argument(
        'model_output_dir',
        type=str,
        help='Path to the top-level output directory for the model (e.g., outputs/resnet50_mixed_sparsity_example) relative to CWD.'
    )
    # Optional: Link back to the sequence file to get arch info easily
    parser.add_argument(
        '--sequence_file',
        type=str,
        default=None,
        help='Optional: Path to the model sequence YAML file used for the run (e.g., model_sequence_example.yaml) relative to CWD. Used to fetch arch filenames.'
    )
    parser.add_argument(
        '--figures_subdir',
        default='figures',
        help='Subdirectory name within the model_output_dir to save plots (default: figures).'
    )
    parser.add_argument(
        '--output_csv',
        default=None,
        help='Filename for the output CSV file (saved in model_output_dir). Default: <model_name>_summary.csv'
    )
    parser.add_argument(
        '--plot_prefix',
        default=None,
        help='Prefix for the output plot filenames (saved in figures_subdir). Default: <model_name>_'
    )

    args = parser.parse_args()

    CWD = Path.cwd() # Usually .../unstructured_sparse_eval
    model_output_path = (CWD / args.model_output_dir).resolve()
    # Base figures_path on CWD (parent dir) instead of model_output_path
    figures_path = (CWD / args.figures_subdir).resolve()
    figures_path.mkdir(parents=True, exist_ok=True)

    if not model_output_path.is_dir():
        print(f"Error: Model output directory '{model_output_path}' not found.", file=sys.stderr)
        sys.exit(1)

    model_name = model_output_path.name
    print(f"Parsing results for model: {model_name} from {model_output_path}")

    # --- Load sequence file if provided to get arch info ---
    layer_to_arch = {}
    if args.sequence_file:
        sequence_file_path = (CWD / args.sequence_file).resolve()
        if sequence_file_path.exists():
            try:
                with open(sequence_file_path, 'r') as f:
                    sequence_config = yaml.safe_load(f)
                for layer_info in sequence_config.get('layers', []):
                    layer_name = layer_info.get('name')
                    arch_file = layer_info.get('sparsity_arch')
                    if layer_name and arch_file:
                        layer_to_arch[layer_name] = arch_file
                print(f"Loaded architecture info from {sequence_file_path}")
            except Exception as e:
                print(f"Warning: Could not load or parse sequence file {sequence_file_path}: {e}")
        else:
             print(f"Warning: Specified sequence file not found: {sequence_file_path}")

    # --- Iterate through layer subdirectories ---
    results_data = []
    layer_dirs = sorted([d for d in model_output_path.iterdir() if d.is_dir() and d.name != args.figures_subdir])

    if not layer_dirs:
        print(f"Error: No layer subdirectories found in {model_output_path}. Expected directories like 'conv1', 'layer2', etc.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(layer_dirs)} potential layer directories.")

    for layer_dir in layer_dirs:
        layer_name = layer_dir.name
        stats_file = layer_dir / "timeloop-mapper.stats.txt"

        if not stats_file.exists():
            print(f"Warning: Stats file not found in {layer_dir}. Skipping layer {layer_name}.")
            continue

        metrics = extract_metrics_from_stats(stats_file)
        if metrics.get('cycles') is None:
            print(f"Warning: Could not extract metrics from {stats_file}. Skipping layer {layer_name}.")
            continue

        # Get sparsity info
        arch_filename = layer_to_arch.get(layer_name)
        sparsity_type = get_sparsity_from_arch(arch_filename)

        results_data.append({
            'Layer': layer_name,
            'Sparsity Arch': arch_filename or 'N/A', # Store the arch filename if available
            'Sparsity Type': sparsity_type,
            'Cycles': metrics.get('cycles'),
            'Energy (pJ)': metrics.get('energy'),
            'Utilization (%)': metrics.get('utilization'),
            'Output Dir': layer_dir.relative_to(CWD) # Store relative path
        })

    if not results_data:
         print("Error: No valid results were parsed from any layer directories.", file=sys.stderr)
         sys.exit(1)

    # --- Process and Save Results ---
    results_df = pd.DataFrame(results_data)
    # Optional: Make Layer categorical if order matters, though sorting layer_dirs helps
    # results_df['Layer'] = pd.Categorical(results_df['Layer'], categories=[d.name for d in layer_dirs], ordered=True)
    results_df.sort_values(by='Layer', inplace=True)

    print(f"\nParsed {len(results_df)} layer results.")
    try:
        get_ipython()
        print("Parsed Data:")
        display(results_df)
    except NameError:
        print(results_df.to_string())

    # Calculate Totals (excluding Utilization)
    total_cycles = results_df['Cycles'].sum()
    total_energy = results_df['Energy (pJ)'].sum()
    # avg_util = results_df['Utilization (%)'].mean() # Removed utilization calculation
    print("\n--- Aggregated Metrics --- ")
    print(f"Total Cycles: {total_cycles:.4e}")
    print(f"Total Energy (pJ): {total_energy:.4e}")
    # print(f"Average Utilization (%): {avg_util:.2f}") # Removed utilization print
    print("------------------------")

    # --- Save to CSV ---
    output_basename = model_name
    csv_filename = args.output_csv or f"{output_basename}_summary.csv"
    output_csv_path = model_output_path / csv_filename # Save inside model dir

    try:
        # Save all collected data, even if not plotted
        results_df.to_csv(output_csv_path, index=False)
        print(f"\nResults saved to: {output_csv_path}")
    except IOError as e:
        print(f"Error writing CSV to {output_csv_path}: {e}", file=sys.stderr)

    # --- Plotting (Cycles and Energy Only) --- 
    print("\nGenerating combined Cycles/Energy summary plot...")
    plot_prefix = args.plot_prefix or f"{model_name}_"
    plot_file_prefix = figures_path / plot_prefix

    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except OSError:
        print("Warning: 'seaborn-v0_8-darkgrid' style not found, using default.")
        plt.style.use('default')

    # --- Combined Plot (Cycles & Energy) --- 
    if not results_df.empty:
        layers = results_df['Layer'].tolist()
        x = np.arange(len(layers))  # the label locations
        width = 0.35  # the width of the bars (can be wider now)

        fig, ax1 = plt.subplots(figsize=(max(10, len(results_df) * 0.8), 6))

        # Plot Cycles and Energy on ax1
        rects1 = ax1.bar(x - width/2, results_df['Cycles'], width, label='Cycles', color='tab:blue')
        rects2 = ax1.bar(x + width/2, results_df['Energy (pJ)'], width, label='Energy (pJ)', color='tab:orange')

        # Add data labels (horizontal)
        ax1.bar_label(rects1, padding=3, fmt='%.2e') # Removed rotation
        ax1.bar_label(rects2, padding=3, fmt='%.2e') # Removed rotation

        # Add labels, title, and axes ticks
        ax1.set_ylabel('Cycles / Energy (pJ)')
        ax1.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        # Increase ylim slightly to make space for labels
        ax1.set_ylim(bottom=0, top=ax1.get_ylim()[1] * 1.15) 
        ax1.set_xticks(x)
        ax1.set_xticklabels(layers, rotation=45, ha="right")
        ax1.set_xlabel("Layer")

        # Generate informative title
        mapping_str = ", ".join([
            f"{row['Layer']}: {row['Sparsity Type']}" 
            for _, row in results_df.iterrows()
        ])
        ax1.set_title(f"Performance Metrics for {model_name}\nLayer Sparsities: {mapping_str}", pad=20)

        # Add legend
        ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

        fig.tight_layout(rect=[0, 0.05, 1, 0.95]) # Adjust layout

        # Save the combined plot
        combined_plot_path = f"{plot_file_prefix}combined_cycles_energy_per_layer.png"
        try:
            plt.savefig(combined_plot_path)
            print(f"Combined plot saved to: {combined_plot_path}")
        except Exception as e:
            print(f"Error saving combined plot: {e}")
        plt.close(fig)

    else:
        print("No data to plot.")

    print("Plotting complete.")

# Define default display function (will be overwritten if in IPython)
def display(x):
    print(x)

if __name__ == "__main__":
    # Attempt to import and use IPython display if available
    try:
        from IPython.display import display as ipython_display
        display = ipython_display # Overwrite the default display
    except (ImportError, NameError):
        pass # Keep the default display function

    main() 