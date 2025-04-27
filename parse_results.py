import argparse
import yaml
import os
import sys
import csv
import re

def parse_file_path(file_path):
    \"\"\"Parses architecture, num_PEs, density_distribution, and sparsity from a file path.\"\"\"
    parts = file_path.split('/')
    if len(parts) < 2:
        return None, None, None, None

    dir_name = parts[-2] # Get the directory name containing the results file
    dir_parts = dir_name.split('_')

    # Expected format: <test_type>_<workload>_<architecture>_<num_PEs>_<density_distribution>_<sparsity>
    # Adjust indices based on the actual naming convention observed
    if len(dir_parts) >= 6:
        architecture = dir_parts[2]
        num_PEs = dir_parts[3]
        density_distribution = dir_parts[4]
        sparsity = dir_parts[5]
        return architecture, num_PEs, density_distribution, sparsity
    else:
        # Handle cases where the naming convention might differ slightly
        # For example, if sparsity is implicitly defined or part of the workload name
        print(f"Warning: Could not parse all details from directory name: {dir_name}. Using defaults or skipping.")
        # Provide default or placeholder values if necessary, or return None
        # Adjust this logic based on expected variations in directory names
        architecture = dir_parts[2] if len(dir_parts) > 2 else None
        num_PEs = dir_parts[3] if len(dir_parts) > 3 else None
        density_distribution = None # Or derive if possible
        sparsity = None # Or derive if possible
        return architecture, num_PEs, density_distribution, sparsity


def extract_metrics_from_stats(stats_file_path):
    \"\"\"Extracts performance metrics from the timeloop stats file.\"\"\"
    metrics = {'cycles': None, 'energy': None, 'utilization': None}
    try:
        with open(stats_file_path, 'r') as f:
            content = f.read()

            # Extract cycles
            cycles_match = re.search(r'Cycles:\\s+(\\d+)', content)
            if cycles_match:
                metrics['cycles'] = int(cycles_match.group(1))

            # Extract energy (assuming uJ and converting to pJ)
            energy_match = re.search(r'^Energy:\\s+([\\d.]+)\\s+uJ', content, re.MULTILINE)
            if energy_match:
                metrics['energy'] = float(energy_match.group(1)) * 1_000_000 # Convert uJ to pJ

            # Extract utilization
            util_match = re.search(r'^Utilization:\\s+([\\d.]+)', content, re.MULTILINE)
            if util_match:
                metrics['utilization'] = float(util_match.group(1)) * 100 # Convert to percentage

    except FileNotFoundError:
        print(f"Error: Stats file not found at {stats_file_path}", file=sys.stderr)
    except Exception as e:
        print(f"Error parsing stats file {stats_file_path}: {e}", file=sys.stderr)
    return metrics

def extract_config_from_arch(arch_file_path):
    \"\"\"Extracts configuration details from the architecture YAML file.\"\"\"
    config = {'arch_name': None, 'arch_PEs': None, 'arch_sparse_support': None, 'mem_size_SMEM': None}
    try:
        with open(arch_file_path, 'r') as f:
            arch_data = yaml.safe_load(f)
            arch = arch_data.get('architecture', {})

            config['arch_name'] = arch.get('name')

            # Navigate nested structure to find PEs count
            subtree_system = next((item for item in arch.get('subtree', []) if item['name'] == 'system'), None)
            if subtree_system:
                subtree_sm = next((item for item in subtree_system.get('subtree', []) if item['name'] == 'SM'), None)
                if subtree_sm:
                    subtree_subpartition = next((item for item in subtree_sm.get('subtree', []) if 'Subpartition' in item['name']), None)
                    if subtree_subpartition:
                         pe_item = next((item for item in subtree_subpartition.get('subtree', []) if 'PE' in item['name']), None)
                         if pe_item and 'instances' in pe_item.get('local', [{}])[1].get('attributes', {}):
                             # Correctly access instances from the MAC unit attributes
                             mac_attributes = next((loc['attributes'] for loc in pe_item.get('local', []) if loc['name'] == 'MAC'), {})
                             config['arch_PEs'] = mac_attributes.get('instances')

                    # Extract SMEM size
                    smem_item = next((item for item in subtree_sm.get('local', []) if item['name'] == 'SMEM'), None)
                    if smem_item and 'attributes' in smem_item:
                         config['mem_size_SMEM'] = smem_item['attributes'].get('data_storage_depth') # Assuming depth represents size in some unit


            # Determine sparse support (example logic, adjust as needed)
            # This is a simplified check; actual logic might depend on specific config keys
            sparse_keywords = ['sparse', 'gating', 'skipping']
            arch_str = json.dumps(arch_data).lower() # Use json dump for easy string search
            config['arch_sparse_support'] = any(keyword in arch_str for keyword in sparse_keywords)


    except FileNotFoundError:
        print(f"Error: Architecture file not found at {arch_file_path}", file=sys.stderr)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file {arch_file_path}: {e}", file=sys.stderr)
    except Exception as e:
        print(f"Error extracting config from {arch_file_path}: {e}", file=sys.stderr)
    return config

def main():
    parser = argparse.ArgumentParser(description='Parse Timeloop simulation results.')
    parser.add_argument('input_dir', help='Directory containing the simulation output subdirectories.')
    parser.add_argument('output_csv', help='Path to the output CSV file.')
    args = parser.parse_args()

    input_dir = args.input_dir
    output_csv_path = args.output_csv

    all_data = []

    # Check if the input directory exists
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory '{input_dir}' not found.", file=sys.stderr)
        sys.exit(1)

    for item in os.listdir(input_dir):
        subdir_path = os.path.join(input_dir, item)
        if os.path.isdir(subdir_path):
            print(f"Processing directory: {subdir_path}")

            # Construct file paths relative to the input directory
            stats_file_rel_path = os.path.join(item, 'timeloop-mapper.map+stats.xml')
            stats_file_abs_path = os.path.join(input_dir, stats_file_rel_path)

            # Parse directory name for experiment parameters
            architecture, num_PEs, density_distribution, sparsity = parse_file_path(stats_file_abs_path)

            if architecture is None: # Skip if parsing failed significantly
                 print(f"Skipping directory {item} due to parsing issues.")
                 continue

            # Construct architecture file path
            # Assuming arch file is in an 'arch' subfolder relative to input_dir
            # Adjust this logic if the arch files are located elsewhere
            arch_file_name = f"{architecture}_{num_PEs}.yaml" if num_PEs else f"{architecture}.yaml" # Handle cases where num_PEs might be missing
            # Let's assume arch files are in a standard location, e.g., input_dir/../arch/
            arch_dir_path = os.path.abspath(os.path.join(input_dir, '..', 'arch')) # Example path
            # If arch files are within the input_dir structure, adjust accordingly
            # arch_dir_path = os.path.join(input_dir, 'arch') # Alternative if arch is inside input_dir

            # Fallback: Search for the arch file within the specific run directory if not found in a central 'arch' dir
            arch_file_path_primary = os.path.join(arch_dir_path, arch_file_name)
            arch_file_path_secondary = os.path.join(subdir_path, arch_file_name) # Check inside the run folder

            arch_file_path = None
            if os.path.exists(arch_file_path_primary):
                 arch_file_path = arch_file_path_primary
            elif os.path.exists(arch_file_path_secondary):
                 arch_file_path = arch_file_path_secondary
            # Add more fallback locations if needed

            if not arch_file_path:
                print(f"Warning: Architecture file '{arch_file_name}' not found for {item}. Searched in {arch_dir_path} and {subdir_path}.", file=sys.stderr)
                # Continue processing without arch config or skip this entry
                arch_config = {'arch_name': None, 'arch_PEs': None, 'arch_sparse_support': None, 'mem_size_SMEM': None}
            else:
                print(f"Found architecture file: {arch_file_path}")
                arch_config = extract_config_from_arch(arch_file_path)


            # Extract performance metrics
            metrics = extract_metrics_from_stats(stats_file_abs_path)

            # Combine data
            combined_data = {
                'directory': item, # Add directory name for context
                'architecture': architecture,
                'num_PEs': num_PEs,
                'density_distribution': density_distribution,
                'sparsity': sparsity,
                'cycles': metrics['cycles'],
                'energy': metrics['energy'],
                'utilization': metrics['utilization'],
                'arch_name': arch_config.get('arch_name'),
                'arch_PEs': arch_config.get('arch_PEs'),
                'arch_sparse_support': arch_config.get('arch_sparse_support'),
                'mem_size_SMEM': arch_config.get('mem_size_SMEM') # Add extra config param
            }
            all_data.append(combined_data)

    # Write data to CSV
    if not all_data:
        print("No data collected. Exiting.", file=sys.stderr)
        sys.exit(0)

    # Determine header - dynamically based on keys in the first data dict
    # Ensure consistent order
    header = list(all_data[0].keys())


    file_exists = os.path.isfile(output_csv_path)
    try:
        with open(output_csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=header)
            if not file_exists or os.path.getsize(output_csv_path) == 0:
                writer.writeheader()
            writer.writerows(all_data)
        print(f"Successfully wrote results to {output_csv_path}")
    except IOError as e:
        print(f"Error writing to CSV file {output_csv_path}: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main() 