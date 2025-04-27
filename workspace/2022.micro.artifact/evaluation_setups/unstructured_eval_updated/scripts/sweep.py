import yaml, inspect, os, sys, subprocess, pprint, shutil
from copy import deepcopy
import numpy as np

OVERWRITE = True

this_file_path = os.path.abspath(inspect.getfile(inspect.currentframe()))
this_directory = os.path.dirname(this_file_path)

# === Base input paths (will be formatted with workload) ===
base_input_dir = os.path.join(this_directory, "..", "input_specs")
problem_template_path_fmt = os.path.join(base_input_dir, "{workload}_unstructured_prob.yaml")
arch_path_fmt = os.path.join(base_input_dir, "{workload}_unstructured_architecture.yaml")
component_path = os.path.join(base_input_dir, "compound_components.yaml") # Shared component
mapping_path_fmt = os.path.join(base_input_dir, "{workload}_unstructured_Os-mapping.yaml")
sparse_opt_path_fmt = os.path.join(base_input_dir, "{workload}_unstructured_sparse-opt.yaml")
ert_path = os.path.join(base_input_dir, "ERT.yaml") # Shared ERT/ART
art_path = os.path.join(base_input_dir, "ART.yaml")


def run_timeloop(job_name, input_dict, base_dir, ert_path, art_path):
    output_dir = os.path.join(base_dir +"/outputs")
   
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        if not OVERWRITE:
            print(f"Found existing results for {job_name}: {output_dir}. Skipping.")
            return # Skip if OVERWRITE is False and dir exists
        else:
            print(f"Found and overwrite existing results for {job_name}: {output_dir}")

    # reuse generated ERT and ART files
    if os.path.exists(ert_path):
        shutil.copy(ert_path, os.path.join(base_dir, "ERT.yaml"))
    else:
        print(f"Warning: ERT file not found at {ert_path}")
    if os.path.exists(art_path):
        shutil.copy(art_path, os.path.join(base_dir, "ART.yaml"))
    else:
        print(f"Warning: ART file not found at {art_path}")
    
    input_file_path = os.path.join(base_dir, "aggregated_input.yaml")
    ert_file_path = os.path.join(base_dir, "ERT.yaml")
    art_file_path = os.path.join(base_dir, "ART.yaml")
    logfile_path = os.path.join(output_dir, "timeloop.log")
    
    yaml.dump(input_dict, open(input_file_path, "w"), default_flow_style=False)
    
    # --- Execute Timeloop --- 
    orig_dir = os.getcwd() # Store original directory
    try:
        os.chdir(output_dir)
        subprocess_cmd = ["timeloop-model", os.path.relpath(input_file_path, output_dir)]
        if os.path.exists(ert_file_path):
             subprocess_cmd.append(os.path.relpath(ert_file_path, output_dir))
        if os.path.exists(art_file_path):
             subprocess_cmd.append(os.path.relpath(art_file_path, output_dir))
        print(f"\tRunning test: {job_name}")

        print(f"Running command: {' '.join(subprocess_cmd)} in {os.getcwd()}")
        with open(logfile_path, 'w') as log_f:
            p = subprocess.Popen(subprocess_cmd, stdout=log_f, stderr=subprocess.STDOUT)
            p.communicate(timeout=None) 
            # Check return code 
            if p.returncode != 0:
                 print(f"ERROR: timeloop-model failed for {job_name} with code {p.returncode}. See log: {logfile_path}")
            else:
                print(f"\tFinished test: {job_name}, log: {logfile_path}")
    finally:
        os.chdir(orig_dir) # Change back to original directory

def main():
    print(f"Using shared ERT path: {ert_path}")
    print(f"Using shared ART path: {art_path}")
    
    output_base_dir = os.path.join(this_directory, "..", "outputs")
    
    workloads = ["resnet", "alexnet", "mobilenet"]
    densities_A = [1.0, 0.9, 0.7, 0.5, 0.3]
    densities_B = [1.0, 0.4]

    # === ADDED: Loop over workloads ===
    for workload in workloads:
        print(f"\nProcessing workload: {workload} === ")
        
        # --- Load base configs for this workload --- 
        try:
            problem_template_path = problem_template_path_fmt.format(workload=workload)
            arch_path = arch_path_fmt.format(workload=workload)
            mapping_path = mapping_path_fmt.format(workload=workload)
            sparse_opt_path = sparse_opt_path_fmt.format(workload=workload)

            problem_template = yaml.load(open(problem_template_path), Loader=yaml.SafeLoader)
            arch = yaml.load(open(arch_path), Loader=yaml.SafeLoader)
            components = yaml.load(open(component_path), Loader=yaml.SafeLoader) # Shared
            mapping = yaml.load(open(mapping_path), Loader=yaml.SafeLoader)
            sparse_opt = yaml.load(open(sparse_opt_path), Loader=yaml.SafeLoader)
        except FileNotFoundError as e:
            print(f"ERROR: Config file missing for workload '{workload}': {e}. Skipping workload.")
            continue # Skip to next workload if files are missing
        except yaml.YAMLError as e:
            print(f"ERROR: YAML parsing error for workload '{workload}': {e}. Skipping workload.")
            continue
            
        # --- Loop over densities for this workload --- 
        for A_density in densities_A:
            for B_density in densities_B:
                
                new_problem = deepcopy(problem_template)
                # Override densities (ensure keys 'Inputs', 'Weights' exist in all prob files)
                try:
                    new_problem["problem"]["instance"]["densities"]["Inputs"]["density"] = A_density
                    new_problem["problem"]["instance"]["densities"]["Weights"]["density"] = B_density
                except KeyError as e:
                    print(f"ERROR: Density key {e} missing in problem file for {workload}. Skipping density pair ({A_density}, {B_density}).")
                    continue

                aggregated_input = {}
                aggregated_input.update(arch)
                aggregated_input.update(new_problem)
                aggregated_input.update(components)
                aggregated_input.update(mapping)
                aggregated_input.update(sparse_opt)
                
                # Define job name including workload and densities
                job_name  = f"{workload}_B_{B_density}-A_{A_density}"
                job_dir = os.path.join(output_base_dir, job_name)
                print(f"Preparing job: {job_name}")

                run_timeloop(job_name, aggregated_input, job_dir, ert_path, art_path)

if __name__ == "__main__":
    main()
        
