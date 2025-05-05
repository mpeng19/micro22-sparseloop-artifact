import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os
from pathlib import Path

# --- Configuration ---
WORKLOAD_CONFIGS = [
    {
        'name': 'ResNet-50 Selected',
        'layer_id': 'M128-K1152-N1024-IAD0.44',
        'csv_file': 'resnet50_selected_0p25_0p3333_0p5_1p0_stc_dstc_comparison.csv',
        'output_prefix': 'fig15_resnet50'
    },
    {
        'name': 'MobileNetV2 Selected',
        'layer_id': 'M24-K96-N3136-IAD0.59',
        'csv_file': 'mobilenetv2_selected_0p25_0p3333_0p5_1p0_stc_dstc_comparison.csv',
        'output_prefix': 'fig15_mobilenetv2'
    },
    {
        'name': 'AlexNet Selected',
        'layer_id': 'M192-K64-N784-IAD0.92',
        'csv_file': 'alexnet_selected_0p25_0p3333_0p5_1p0_stc_dstc_comparison.csv',
        'output_prefix': 'fig15_alexnet'
    },
]

COMBINED_OUTPUT_FILENAME = 'fig15_all_workloads_comparison.png'

# Colors matching the target image
COLOR_STC = 'cornflowerblue'
COLOR_DSTC = 'firebrick'

# --- Paths ---
script_dir = Path(__file__).parent.resolve()
base_dir = script_dir.parent
csv_dir = base_dir / 'outputs'
output_dir = base_dir / 'outputs' / 'figures' # Store plots in a figures subdir
output_dir.mkdir(parents=True, exist_ok=True) # Create figures dir if needed

# --- Helper Functions ---
def format_sigfigs(x, n=2):
    if x == 0:
        return "0"
    elif np.isnan(x):
        return "NaN"
    # Use standard string formatting for significant figures
    return f"{x:.{n}g}"

def plot_layer_comparison(ax_cycles, ax_energy, df_layer, workload_name, layer_id, show_legend=True, show_xlabel=True):
    bar_width = 0.35

    # Ensure density is float, sort, and format for labels
    df_layer['density'] = df_layer['density'].astype(float)
    df_layer = df_layer.sort_values(by='density')
    densities_num = df_layer['density'].unique()
    # Format density labels to percentage with 1 decimal place
    density_labels = [f'{d*100:.1f}%' for d in densities_num]
    x = np.arange(len(densities_num))

    stc_data = df_layer[df_layer['scheme'] == 'STC']
    dstc_data = df_layer[df_layer['scheme'] == 'DSTC']

    # Reindex to ensure data aligns with sorted unique densities
    stc_cycles = stc_data.set_index('density').reindex(densities_num)['cycles'].fillna(0)
    dstc_cycles = dstc_data.set_index('density').reindex(densities_num)['cycles'].fillna(0)
    stc_energy = stc_data.set_index('density').reindex(densities_num)['energy_pj'].fillna(0)
    dstc_energy = dstc_data.set_index('density').reindex(densities_num)['energy_pj'].fillna(0)

    # Plotting bars
    ax_cycles.bar(x - bar_width/2, stc_cycles, bar_width, label='STC', color=COLOR_STC)
    ax_cycles.bar(x + bar_width/2, dstc_cycles, bar_width, label='DSTC', color=COLOR_DSTC)

    ax_energy.bar(x - bar_width/2, stc_energy, bar_width, label='STC', color=COLOR_STC)
    ax_energy.bar(x + bar_width/2, dstc_energy, bar_width, label='DSTC', color=COLOR_DSTC)

    # --- Formatting Axes ---
    # Remove FormatStrFormatter
    # formatter = mticker.FormatStrFormatter('%.2e')

    # Top plot (Cycles)
    ax_cycles.set_ylabel('Total Cycles')
    # Restore ticklabel_format for exponent at top
    ax_cycles.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax_cycles.grid(False) # Remove grid lines
    # Remove title setting from here - will be set in the calling loop
    # ax_cycles.set_title(f'{workload_name}\nLayer: {layer_id}')

    # Bottom plot (Energy)
    ax_energy.set_ylabel('Total Energy (pJ)')
    # Restore ticklabel_format for exponent at top
    ax_energy.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax_energy.grid(False) # Remove grid lines

    # Set x-axis ticks and labels (using formatted densities)
    ax_energy.set_xticks(x)
    ax_energy.set_xticklabels(density_labels, rotation=45, ha='right')

    if show_xlabel:
        ax_energy.set_xlabel('Density')
    if show_legend:
        ax_cycles.legend()
        ax_energy.legend()

# --- Main Execution ---
all_data = []
for config in WORKLOAD_CONFIGS:
    csv_path = csv_dir / config['csv_file']
    try:
        df_temp = pd.read_csv(csv_path)
        df_temp['workload_name'] = config['name']
        df_temp['layer_id'] = config['layer_id']
        df_temp['output_prefix'] = config['output_prefix']
        # Filter only the layer we need for this workload config
        df_layer_specific = df_temp[df_temp['layer_config'] == config['layer_id']].copy()
        if df_layer_specific.empty:
             print(f"Warning: No data found for layer '{config['layer_id']}' in {csv_path}")
             continue
        all_data.append(df_layer_specific)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
    except Exception as e:
        print(f"Error reading CSV file {csv_path}: {e}")

if not all_data:
    print("Error: No data loaded successfully. Exiting.")
    exit(1)

df_combined = pd.concat(all_data, ignore_index=True)

plt.style.use('seaborn-v0_8-whitegrid') # Base style

# 1) Generate Individual Plots
print("Generating individual plots...")
for config in WORKLOAD_CONFIGS:
    df_layer = df_combined[(df_combined['layer_id'] == config['layer_id']) &
                           (df_combined['workload_name'] == config['name'])]

    if df_layer.empty:
        print(f"Skipping individual plot for {config['name']} - layer data missing.")
        continue

    fig_single, (ax1_single, ax2_single) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    plot_layer_comparison(ax1_single, ax2_single, df_layer.copy(), config['name'], config['layer_id'], show_legend=True, show_xlabel=True)

    # Set title on the top subplot for individual plots - Remove layer info
    ax1_single.set_title(f"{config['name']}")

    fig_single.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust rect for bottom/top space

    single_output_path = output_dir / f"{config['output_prefix']}_comparison.png"
    try:
        plt.savefig(single_output_path, bbox_inches='tight')
        print(f"  Saved: {single_output_path}")
    except Exception as e:
        print(f"  Error saving plot {single_output_path}: {e}")
    plt.close(fig_single) # Close the figure to free memory

# 2) Generate Combined Plot
print("\nGenerating combined plot...")
num_workloads = len(WORKLOAD_CONFIGS)
# Increase width from 12 to 18 for a wider aspect ratio
fig_combined, axes = plt.subplots(num_workloads, 2, figsize=(18, 5 * num_workloads), sharex=True)

if num_workloads == 1:
    axes = np.array([axes]) # Ensure axes is always 2D for consistent indexing

for i, config in enumerate(WORKLOAD_CONFIGS):
    df_layer = df_combined[(df_combined['layer_id'] == config['layer_id']) &
                           (df_combined['workload_name'] == config['name'])]

    if df_layer.empty:
        print(f"Skipping combined plot section for {config['name']} - layer data missing.")
        # Optionally blank out the axes
        axes[i, 0].axis('off')
        axes[i, 1].axis('off')
        continue

    ax_cycles = axes[i, 0]
    ax_energy = axes[i, 1]

    # Show legend only for the first row, xlabel only for the last row
    show_legend = (i == 0)
    show_xlabel = (i == num_workloads - 1)

    plot_layer_comparison(ax_cycles, ax_energy, df_layer.copy(), config['name'], config['layer_id'], show_legend=show_legend, show_xlabel=show_xlabel)

    # Add centered title manually for the row in the combined plot
    pos_left = axes[i, 0].get_position()
    pos_right = axes[i, 1].get_position()
    mid_x = (pos_left.x0 + pos_right.x1) / 2
    y_top = pos_left.y1 # Top of the axes row
    fig_combined.text(mid_x, y_top + 0.01, # Add small offset above
                      # Remove layer info from combined title
                      f"{config['name']}",
                      ha='center', va='bottom', fontsize=12) # Adjust fontsize as needed

# Adjust layout for combined plot - try subplots_adjust instead of tight_layout
# fig_combined.tight_layout(rect=[0, 0.03, 1, 0.98]) # tight_layout can interfere with fig.text
plt.subplots_adjust(hspace=0.4, wspace=0.3) # Adjust spacing between subplots

combined_output_path = output_dir / COMBINED_OUTPUT_FILENAME
try:
    plt.savefig(combined_output_path, bbox_inches='tight')
    print(f"Combined plot saved to: {combined_output_path}")
except Exception as e:
    print(f"Error saving combined plot: {e}")
plt.close(fig_combined)

print("\nDone.") 