#!/usr/bin/env python3

import os
import re
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Configuration - focus on structured sparsity
ARCHITECTURES = ["structured_2_4"]
WORKLOADS = ["resnet50_conv1", "alexnet_conv1_sparse", "mobilenet_conv1_sparse"]
COLORS = {'structured_2_4': 'blue', 'unstructured': 'orange', 'ideal': 'green'}

def parse_timeloop_stats(stats_file):
    """Parse Timeloop stats file to extract metrics."""
    if not os.path.exists(stats_file):
        print(f"Stats file not found: {stats_file}")
        return None

    stats = {'EnergyBreakdown': {}}
    total_energy = 0
    
    try:
        with open(stats_file, 'r') as f:
            lines = f.readlines()

        # Extract cycles
        cycles_pattern = r"Cycles\s*:\s*([\d,]+)"
        for line in lines:
            cycles_match = re.search(cycles_pattern, line)
            if cycles_match:
                stats['Cycles'] = int(cycles_match.group(1).replace(',', ''))
                break
        
        # Extract utilization
        util_pattern = r"Utilized instances \(average\)\s*:\s*([\d.]+)"
        total_instances_pattern = r"Instances\s*:\s*(\d+)"
        
        total_instances = 0
        utilized_instances = 0
        
        for i, line in enumerate(lines):
            if "=== MAC ===" in line:
                # Look for total instances
                for j in range(i, min(i+10, len(lines))):
                    instances_match = re.search(total_instances_pattern, lines[j])
                    if instances_match:
                        total_instances = int(instances_match.group(1))
                        break
                
                # Look for utilized instances
                for j in range(i, min(i+20, len(lines))):
                    util_match = re.search(util_pattern, lines[j])
                    if util_match:
                        utilized_instances = float(util_match.group(1))
                        break
                break
        
        if total_instances > 0:
            stats['Utilization'] = (utilized_instances / total_instances) * 100
        
        # Extract energy from individual components
        energy_pattern = r"Energy \(total\)\s*:\s*([\d.,]+)\s*pJ"
        component_pattern = r"=== ([A-Za-z0-9_]+) ==="
        current_component = None
        
        for line in lines:
            component_match = re.search(component_pattern, line)
            if component_match:
                current_component = component_match.group(1)
            
            energy_match = re.search(energy_pattern, line)
            if energy_match and current_component:
                energy_val = float(energy_match.group(1).replace(',', ''))
                stats['EnergyBreakdown'][current_component] = energy_val
                total_energy += energy_val
        
        stats['Energy'] = total_energy
        
        return stats
    except Exception as e:
        print(f"Error parsing stats file {stats_file}: {e}")
        return None

def gather_results(base_dir):
    """Gather all results from output directory."""
    output_dir = os.path.join(base_dir, "outputs")  # Use existing outputs directory
    structured_stats = {}
    
    # Directory mapping for structured workloads
    workload_dir_mapping = {
        "resnet50_conv1": "resnet50_conv1_structured_2_4",
        "alexnet_conv1_sparse": "alexnet_conv1_structured_2_4",
        "mobilenet_conv1_sparse": "mobilenet_conv1_structured_2_4"
    }
    
    # For each workload, look for structured_2_4 results
    for workload in WORKLOADS:
        # Get the correct directory name for this workload
        output_dir_name = workload_dir_mapping.get(workload)
        if not output_dir_name:
            print(f"Warning: No directory mapping for {workload}")
            continue
            
        output_path = os.path.join(output_dir, output_dir_name)
        
        # Check for timeloop-mapper.stats.txt
        stats_file = os.path.join(output_path, "timeloop-mapper.stats.txt")
        
        if os.path.exists(stats_file):
            stats = parse_timeloop_stats(stats_file)
            if stats:  # Only add if we successfully parsed stats
                structured_stats[workload] = stats
        else:
            print(f"Warning: Stats file not found for {workload} with structured_2_4")
            print(f"  Looked in: {stats_file}")
    
    return structured_stats

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <base_directory>")
        sys.exit(1)
    
    base_dir = sys.argv[1]
    print(f"Processing results from {base_dir}")
    
    # Create figures directory if it doesn't exist
    output_dir = os.path.join(base_dir, "outputs")
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    # Gather structured_2_4 results
    structured_stats = gather_results(base_dir)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("\nStructured 2:4 Sparse Architecture:")
    
    if structured_stats:
        for workload, stats in structured_stats.items():
            print(f"{workload}:")
            print(f"  Cycles: {stats['Cycles']:,}")
            print(f"  Energy: {stats['Energy']:,} pJ")
            if 'Utilization' in stats:
                print(f"  Utilization: {stats['Utilization']:.2f}%")
        
        # Plot structured sparse architecture results without energy breakdown
        available_workloads = [w for w in WORKLOADS if w in structured_stats]
        
        if available_workloads:
            # Create 1x2 figure layout
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Cycles plot
            cycles = [structured_stats[w]['Cycles'] for w in available_workloads]
            bars = ax1.bar(available_workloads, cycles, color=COLORS['structured_2_4'])
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
            energy = [structured_stats[w]['Energy'] for w in available_workloads]
            bars = ax2.bar(available_workloads, energy, color=COLORS['structured_2_4'])
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
            
            print("Structured sparse architecture plot created successfully!")
        else:
            print("No available workloads to plot.")
    else:
        print("No data found for structured_2_4 architecture.")
    
    print(f"Plots saved to {figures_dir}")

if __name__ == "__main__":
    main() 