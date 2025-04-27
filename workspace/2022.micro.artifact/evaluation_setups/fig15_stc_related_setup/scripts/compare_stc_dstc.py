import os
import sys
import inspect
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
from collections import defaultdict

# Import the parser function from the other script
try:
    from parse_timeloop_stats_file import parse_stats_txt
except ImportError:
    print("ERROR: Cannot find parse_timeloop_stats_file.py.")
    print("Ensure it is in the same directory as this script.")
    sys.exit(1)

# --- Configuration ---
# MODEL_NAME = "resnet50_selected" # Replaced by argparse
TARGET_WD = "WD-0.5"
TARGET_SCHEMES = {
    "DSTC": "DSTC-RF2x-24-bandwidth",
    "STC": "STC-RF2x-24-bandwidth"
}
STATS_FILENAME = "timeloop-model.stats.txt"

# --- Setup Paths ---
scripts_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
base_output_dir = os.path.abspath(os.path.join(scripts_dir, '..', 'outputs'))
# model_wd_dir = os.path.join(base_output_dir, MODEL_NAME, TARGET_WD) # Path construction moved to main()

# --- Output File ---
# plot_output_dir = base_output_dir # Save plot in the main outputs folder
# plot_filename = os.path.join(plot_output_dir, f"{MODEL_NAME}_{TARGET_WD}_stc_dstc_normalized_comparison.png")

# --- Main Logic ---
def main():
    parser = argparse.ArgumentParser(description=f"Compare STC vs DSTC results for a given workload at {TARGET_WD}")
    # Add workload argument
    parser.add_argument('--workload', type=str, required=True, 
                        choices=['resnet50_selected', 'alexnet_selected', 'mobilenetv2_selected'],
                        help='Name of the model workload to compare (e.g., resnet50_selected).')
    args = parser.parse_args()
    workload = args.workload # Use parsed workload

    # --- Construct paths using the parsed workload ---
    model_wd_dir = os.path.join(base_output_dir, workload, TARGET_WD)
    plot_output_dir = base_output_dir # Keep saving plot in the main outputs folder for now
    plot_filename = os.path.join(plot_output_dir, f"{workload}_{TARGET_WD}_stc_dstc_comparison.png") # Changed name slightly

    print(f"--- Starting Comparison for {workload} ({TARGET_WD}) ---")
    print(f"Looking for layer directories in: {model_wd_dir}")

    if not os.path.isdir(model_wd_dir):
        print(f"ERROR: Directory not found: {model_wd_dir}")
        print("Please ensure the sweep results exist.")
        sys.exit(1)

    # Structure: results[layer_config][scheme_short_name] = {'cycles': N, 'energy_pj': M}
    results = defaultdict(dict)
    processed_layers = []

    # List layer directories (e.g., M128-K1152-N1024-IAD0.44-WD0.5)
    try:
        layer_dirs = [d for d in os.listdir(model_wd_dir) if os.path.isdir(os.path.join(model_wd_dir, d))]
    except OSError as e:
         print(f"ERROR: Could not list directory contents: {model_wd_dir} - {e}")
         sys.exit(1)

    print(f"Found {len(layer_dirs)} potential layer directories.")

    for layer_config_name in sorted(layer_dirs):
        print(f"  Processing layer: {layer_config_name}")
        layer_path = os.path.join(model_wd_dir, layer_config_name)
        processed_layers.append(layer_config_name)
        layer_results_found = False

        for scheme_short_name, scheme_full_name in TARGET_SCHEMES.items():
            stats_file = os.path.join(layer_path, scheme_full_name, "output", STATS_FILENAME)

            parsed_data = parse_stats_txt(stats_file)

            if parsed_data:
                results[layer_config_name][scheme_short_name] = parsed_data
                layer_results_found = True
                # print(f"    Parsed {scheme_short_name}: Cycles={parsed_data['cycles']}, Energy={parsed_data['energy_pj']:.2f} pJ")
            else:
                # Store None if parsing failed or file not found
                results[layer_config_name][scheme_short_name] = {'cycles': None, 'energy_pj': None}
                print(f"    Warning: Could not parse stats for {scheme_short_name} in {layer_config_name}")

        if not layer_results_found:
             print(f"    Warning: No valid STC or DSTC data found for layer {layer_config_name}")


    print("\n--- Comparison Summary ---")
    # Header
    print(f"{'Layer Configuration':<40} | {'Scheme':<6} | {'Cycles':<15} | {'Energy (pJ)':<20}")
    print("-" * 90)

    # Data Rows
    for layer_config, schemes_data in sorted(results.items()):
        for scheme_name in sorted(TARGET_SCHEMES.keys()): # Print STC then DSTC
             data = schemes_data.get(scheme_name)
             if data and data['cycles'] is not None and data['energy_pj'] is not None:
                 cycles_str = f"{data['cycles']:,}"
                 energy_str = f"{data['energy_pj']:,.2f}"
                 print(f"{layer_config:<40} | {scheme_name:<6} | {cycles_str:<15} | {energy_str:<20}")
             else:
                 # Handle missing data gracefully in printout
                 print(f"{layer_config:<40} | {scheme_name:<6} | {'Not found':<15} | {'Not found':<20}")
        print("-" * 90) # Separator between layers

    # --- Convert to DataFrame and Normalize ---
    print("\nProcessing data for plotting...")
    all_data_list = []
    for layer_config, schemes_data in results.items():
        for scheme_name, data in schemes_data.items():
            if data and data['cycles'] is not None and data['energy_pj'] is not None:
                all_data_list.append({
                    'layer_config': layer_config,
                    'scheme': scheme_name,
                    'cycles': data['cycles'],
                    'energy_pj': data['energy_pj']
                })

    if not all_data_list:
        print("ERROR: No valid data collected. Cannot generate plot.")
        sys.exit(1)

    df = pd.DataFrame(all_data_list)

    # --- Prepare data for plotting (use raw values) ---
    df_plot = df.copy() # Use the original dataframe
    df_plot = df_plot.dropna() # Drop rows where parsing might have failed for a scheme

    if df_plot.empty:
        print("ERROR: Dataframe is empty after dropping NA. Cannot plot.") # Updated error message
        sys.exit(1)

    # Sort for consistent plotting
    df_plot_sorted = df_plot.sort_values(by=['layer_config', 'scheme'])

    # --- Plotting ---
    print(f"\nGenerating plot and saving to: {plot_filename}")
    plt.style.use('seaborn-whitegrid')
    plt.rcParams.update({'font.size': 10})
    DSTC_COLOR = 'firebrick'
    STC_COLOR = 'cornflowerblue'

    layer_configs = df_plot_sorted['layer_config'].unique()
    N = len(layer_configs)
    ind = np.arange(N)
    bar_width = 0.35

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(10, N * 0.8), 8), sharex=True)

    # --- Cycles Subplot (Use Absolute Values) ---
    # Pivot the raw data for plotting side-by-side bars
    cycles_pivot = df_plot_sorted.pivot(index='layer_config', columns='scheme', values='cycles').reindex(layer_configs)
    # Check if both schemes are present for plotting
    if 'DSTC' not in cycles_pivot.columns or 'STC' not in cycles_pivot.columns:
        print("ERROR: Missing STC or DSTC cycle data for plotting.")
        sys.exit(1)
    ax1.bar(ind - bar_width/2, cycles_pivot['DSTC'].values, bar_width, label='DSTC', color=DSTC_COLOR)
    ax1.bar(ind + bar_width/2, cycles_pivot['STC'].values, bar_width, label='STC', color=STC_COLOR)
    ax1.set_ylabel('Total Cycles', fontsize=12) # Updated label
    ax1.set_ylim(bottom=0)
    ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0)) # Use scientific notation if needed

    # --- Energy Subplot (Use Absolute Values) ---
    energy_pivot = df_plot_sorted.pivot(index='layer_config', columns='scheme', values='energy_pj').reindex(layer_configs)
    # Check if both schemes are present for plotting
    if 'DSTC' not in energy_pivot.columns or 'STC' not in energy_pivot.columns:
        print("ERROR: Missing STC or DSTC energy data for plotting.")
        sys.exit(1)
    ax2.bar(ind - bar_width/2, energy_pivot['DSTC'].values, bar_width, label='DSTC', color=DSTC_COLOR)
    ax2.bar(ind + bar_width/2, energy_pivot['STC'].values, bar_width, label='STC', color=STC_COLOR)
    ax2.set_ylabel('Total Energy (pJ)', fontsize=12) # Updated label
    ax2.set_ylim(bottom=0)
    ax2.set_xticks(ind)
    ax2.set_xticklabels(layer_configs, rotation=90, ha='center', fontsize=9)
    # ax2.text(0.5, -0.4, '50% (2:4)', ...) # Removed normalization text
    ax2.set_xlabel('GEMM Configuration', labelpad=40, fontsize=12)
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0)) # Use scientific notation if needed

    fig.suptitle(f'DSTC vs. STC Performance for {workload} ({TARGET_WD})', fontsize=14, y=0.99)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    # Ensure output directory exists
    os.makedirs(plot_output_dir, exist_ok=True)
    plt.savefig(plot_filename, bbox_inches='tight')
    print("Plot saved successfully.")


if __name__ == "__main__":
    main() 