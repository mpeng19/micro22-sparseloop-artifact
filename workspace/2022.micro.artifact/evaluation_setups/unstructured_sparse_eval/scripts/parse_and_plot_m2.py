#!/usr/bin/env python3

import os
import re
import matplotlib.pyplot as plt
import numpy as np

def parse_stats(stats_file):
    stats = {}
    with open(stats_file, 'r') as f:
        content = f.read()
        
        # Extract cycles
        cycles_match = re.search(r'Cycles:\s+(\d+)', content)
        if cycles_match:
            stats['cycles'] = int(cycles_match.group(1))
        
        # Extract energy from Summary Stats (convert uJ to pJ)
        summary_energy_match = re.search(r'^Energy:\s+([\d.]+)\s+uJ', content, re.MULTILINE)
        if summary_energy_match:
            stats['energy'] = float(summary_energy_match.group(1)) * 1_000_000 # Convert uJ to pJ
        else:
            print(f"Warning: Could not find 'Energy: ... uJ' in summary of {stats_file}")
            stats['energy'] = 0
        
        # Extract utilization from Summary Stats
        util_match = re.search(r'^Utilization:\s+([\d.]+)', content, re.MULTILINE)
        if util_match:
            stats['utilization'] = float(util_match.group(1)) * 100 # Assuming the value is fraction, convert to %
    
    return stats

def plot_structured(structured_stats, figures_dir):
    """Plot graphs for structured 2:4 sparse architecture"""
    workloads = ['resnet50_conv1', 'alexnet_conv1', 'mobilenet_conv1']
    available_workloads = [w for w in workloads if w in structured_stats]
    
    if not structured_stats:
        print("No data available for structured 2:4 sparse architecture.")
        return
    
    # Create 1x2 figure layout
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Cycles plot
    cycles = [structured_stats[w]['cycles'] for w in available_workloads]
    bars = ax1.bar(available_workloads, cycles)
    ax1.set_title('Cycles Comparison (Structured 2:4 Sparse)')
    ax1.set_xlabel('Workload')
    ax1.set_ylabel('Cycles')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom')
    
    # Energy plot
    energy = [structured_stats[w]['energy'] for w in available_workloads]
    bars = ax2.bar(available_workloads, energy)
    ax2.set_title('Energy Comparison (Structured 2:4 Sparse)')
    ax2.set_xlabel('Workload')
    ax2.set_ylabel('Energy (pJ)')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'structured_comparison.png'))
    plt.close()
    
    print("Structured 2:4 sparse architecture plots created successfully!")

def main():
    # Base directory - use Docker path
    base_dir = "/home/workspace/2022.micro.artifact/evaluation_setups/unstructured_sparse_eval"
    output_dir = os.path.join(base_dir, 'outputs')
    figures_dir = os.path.join(output_dir, 'figures')
    
    # Create figures directory if it doesn't exist
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir, exist_ok=True)
    
    # Initialize stats dictionary for structured architecture
    structured_stats = {}
    
    # Process structured sparse results
    for workload in ['resnet50_conv1', 'alexnet_conv1', 'mobilenet_conv1']:
        stats_file = os.path.join(output_dir, f'structured_2_4_{workload}', 'timeloop-mapper.stats.txt')
        if os.path.exists(stats_file):
            stats = parse_stats(stats_file)
            if stats:  # Only add if we successfully parsed stats
                structured_stats[workload] = stats
        else:
            print(f"Warning: Stats file not found for {workload} at {stats_file}")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("\nStructured 2:4 Sparse Architecture:")
    for workload, stats in structured_stats.items():
        print(f"{workload}:")
        print(f"  Cycles: {stats['cycles']:,}")
        print(f"  Energy: {stats['energy']:,} pJ")
        if 'utilization' in stats:
            print(f"  Utilization: {stats['utilization']:.2f}%")
    
    # Plot structured sparse architecture results
    plot_structured(structured_stats, figures_dir)

if __name__ == "__main__":
    main()