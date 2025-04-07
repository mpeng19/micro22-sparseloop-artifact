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
        cycles_match = re.search(r'Cycles\s+:\s+(\d+)', content)
        if cycles_match:
            stats['cycles'] = int(cycles_match.group(1))
        
        # Extract energy
        energy_match = re.search(r'Energy \(total\)\s+:\s+([\d.]+)', content)
        if energy_match:
            stats['energy'] = float(energy_match.group(1))
        
        # Extract utilization
        util_match = re.search(r'Utilized instances \(average\)\s+:\s+([\d.]+)', content)
        if util_match:
            stats['utilization'] = float(util_match.group(1))
    
    return stats

def plot_ideal(ideal_stats, figures_dir):
    """Plot graphs for ideal sparse architecture"""
    workloads = ['resnet50_conv1', 'alexnet_conv1_sparse', 'mobilenet_conv1_sparse']
    available_workloads = [w for w in workloads if w in ideal_stats]
    
    if not ideal_stats:
        print("No data available for ideal sparse architecture.")
        return
    
    # Create 1x2 figure layout
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Cycles plot
    cycles = [ideal_stats[w]['cycles'] for w in available_workloads]
    bars = ax1.bar(available_workloads, cycles)
    ax1.set_title('Cycles Comparison (Ideal Sparse)')
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
    energy = [ideal_stats[w]['energy'] for w in available_workloads]
    bars = ax2.bar(available_workloads, energy)
    ax2.set_title('Energy Comparison (Ideal Sparse)')
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
    plt.savefig(os.path.join(figures_dir, 'ideal_comparison.png'))
    plt.close()
    
    print("Ideal sparse architecture plots created successfully!")

def main():
    # Base directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, 'outputs')
    figures_dir = os.path.join(output_dir, 'figures')
    
    # Create figures directory if it doesn't exist
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir, exist_ok=True)
    
    # Initialize stats dictionary for ideal architecture
    ideal_stats = {}
    
    # Process ideal sparse results
    for workload in ['resnet50_conv1', 'alexnet_conv1_sparse', 'mobilenet_conv1_sparse']:
        stats_file = os.path.join(output_dir, f'ideal_{workload}', 'timeloop-mapper.stats.txt')
        if os.path.exists(stats_file):
            stats = parse_stats(stats_file)
            if stats:  # Only add if we successfully parsed stats
                ideal_stats[workload] = stats
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("\nIdeal Sparse Architecture:")
    for workload, stats in ideal_stats.items():
        print(f"{workload}:")
        print(f"  Cycles: {stats['cycles']:,}")
        print(f"  Energy: {stats['energy']:,} pJ")
        if 'utilization' in stats:
            print(f"  Utilization: {stats['utilization']:.2f}%")
    
    # Plot ideal sparse architecture results
    plot_ideal(ideal_stats, figures_dir)

if __name__ == "__main__":
    main() 