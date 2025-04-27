#!/usr/bin/env python3

import os
import re
import matplotlib.pyplot as plt
import numpy as np

def parse_stats(stats_file):
    stats = {}
    if not os.path.exists(stats_file):
        print(f"Error: Stats file not found at {stats_file}")
        return None
        
    with open(stats_file, 'r') as f:
        content = f.read()
        
        # Extract cycles
        cycles_match = re.search(r'Cycles:\s+(\d+)', content)
        if cycles_match:
            stats['cycles'] = int(cycles_match.group(1))
        else:
            print(f"Warning: Could not find 'Cycles:' in {stats_file}")
            stats['cycles'] = 0
        
        # Extract energy from Summary Stats (convert uJ to pJ)
        # Look for "Energy: <value> uJ" in the Summary Stats section
        summary_energy_match = re.search(r'^Energy:\s+([\d.]+)\s+uJ', content, re.MULTILINE)
        if summary_energy_match:
            stats['energy'] = float(summary_energy_match.group(1)) * 1_000_000 # Convert uJ to pJ
        else:
            print(f"Warning: Could not find 'Energy: ... uJ' in summary of {stats_file}")
            stats['energy'] = 0
        
        # Extract utilization from Summary Stats (optional)
        util_match = re.search(r'^Utilization:\s+([\d.]+)', content, re.MULTILINE)
        if util_match:
            stats['utilization'] = float(util_match.group(1)) * 100 # Assuming the value is fraction, convert to %
    
    return stats

def plot_sparsity_results(workload_name, sparsity_stats, figures_dir, output_filename):
    """Plot graphs for different sparsity pairs for a given workload"""
    
    if not sparsity_stats:
        print(f"No data available to plot for workload: {workload_name}")
        return
        
    # Sort sparsity pairs for consistent plotting order
    sparsity_pairs = sorted(list(sparsity_stats.keys()))
    
    # Create x-axis labels from sparsity tuples
    x_labels = [f"A_sp:{p[0]:.1f}_B_sp:{p[1]:.1f}" for p in sparsity_pairs] 
    num_pairs = len(sparsity_pairs)
    
    # Create 1x2 figure layout
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(8, num_pairs * 2.5), 6)) 
    fig.suptitle(f'{workload_name.capitalize()} Workload Performance vs. Sparsity') # Use workload name in title
    
    # Cycles plot
    cycles = [sparsity_stats[p]['cycles'] for p in sparsity_pairs]
    x_pos = np.arange(num_pairs)
    bars = ax1.bar(x_pos, cycles, width=0.6) 
    ax1.set_title('Cycles vs. Sparsity Pair') 
    ax1.set_xlabel('A_Sparsity_B_Sparsity') 
    ax1.set_ylabel('Cycles')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels)
    ax1.tick_params(axis='x', rotation=45)
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height, f'{int(height):,}', ha='center', va='bottom', fontsize=9)
    
    # Energy plot
    energy = [sparsity_stats[p]['energy'] for p in sparsity_pairs]
    bars = ax2.bar(x_pos, energy, width=0.6)
    ax2.set_title('Energy vs. Sparsity Pair') 
    ax2.set_xlabel('A_Sparsity_B_Sparsity') 
    ax2.set_ylabel('Energy (pJ)')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(x_labels)
    ax2.tick_params(axis='x', rotation=45)
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height, f'{int(height):,}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    # Save with the provided filename
    plot_filepath = os.path.join(figures_dir, output_filename)
    plt.savefig(plot_filepath)
    plt.close()
    
    print(f"Sparsity comparison plot created successfully: {plot_filepath}")

def main():
    base_dir = "/home/workspace/2022.micro.artifact/evaluation_setups/unstructured_eval_updated"
    output_dir = os.path.join(base_dir, 'outputs')
    figures_dir = os.path.join(output_dir, 'figures')
    
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir, exist_ok=True)
    
    # Define workloads and densities
    workloads = ["resnet", "alexnet", "mobilenet"]
    densities_A = [1.0, 0.9, 0.7, 0.5, 0.3]
    densities_B = [1.0, 0.4]
    
    # Store stats for all workloads
    all_workload_stats = {}

    # === ADDED: Loop over workloads ===
    print("Parsing results across all workloads and density pairs...")
    for workload in workloads:
        print(f"\n--- Processing workload: {workload} ---")
        workload_density_stats = {}
        
        for A_density in densities_A:
            for B_density in densities_B:
                current_density_pair = (A_density, B_density)
                # Construct the job name used by sweep.py
                job_name = f"{workload}_B_{B_density}-A_{A_density}"
                # Path to the stats file for this job
                stats_file = os.path.join(output_dir, job_name, 'outputs', 'timeloop-model.stats.txt') 
                
                # Parse stats (Sparsity conversion happens only for plotting labels)
                stats = parse_stats(stats_file)
                if stats:
                    # Store stats keyed by DENSITY pair
                    workload_density_stats[current_density_pair] = stats 
                else:
                    print(f"---> Skipping density pair {current_density_pair} for {workload} (missing file or parse error).")
        
        # Store results for this workload
        all_workload_stats[workload] = workload_density_stats
        
        # --- Print summary for this workload --- 
        print(f"\nSummary Statistics ({workload.capitalize()} Workload):")
        if not workload_density_stats:
            print("  No data parsed for this workload.")
        else:
            # Sort by density key for consistent output order
            for density_pair in sorted(workload_density_stats.keys()):
                stats = workload_density_stats[density_pair]
                # Print density, but plot labels will be sparsity
                print(f"  Density (A={density_pair[0]}, B={density_pair[1]}):")
                print(f"    Cycles: {stats.get('cycles', 'N/A'):,}")
                print(f"    Energy: {stats.get('energy', 'N/A'):,} pJ")
                if 'utilization' in stats:
                    print(f"    Utilization: {stats['utilization']:.2f}%")
        
        # --- Plot results for this workload --- 
        output_plot_filename = f"{workload}_sparsity_comparison.png"
        # Pass the density stats (plot function calculates sparsity for labels)
        plot_sparsity_results(workload, workload_density_stats, figures_dir, output_plot_filename)

    print("\nFinished processing all workloads.")
    # Note: all_workload_stats dictionary contains all parsed data if needed later.

if __name__ == "__main__":
    main() 