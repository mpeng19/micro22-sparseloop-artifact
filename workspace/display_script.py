from IPython.display import Image, display
import os

# --- Configuration ---
WORKLOADS = ['alexnet_selected', 'resnet50_selected', 'mobilenetv2_selected']
# This string should match how it's generated in compare_stc_dstc.py
DENSITY_SORT_ORDER = [0.25, 0.3333, 0.5, 1.0]
densities_str = "_".join(map(str, DENSITY_SORT_ORDER)).replace(".", "p") # e.g., 0p25_0p3333_0p5_1p0
base_output_dir = "/home/workspace/2022.micro.artifact/evaluation_setups/fig15_stc_related_setup/outputs"
# ---

print(f"Looking for plots in: {base_output_dir}")
print(f"Density string part: {densities_str}")

for workload in WORKLOADS:
    # Construct the expected filename based on the compare script's output
    image_filename = os.path.join(base_output_dir, f"{workload}_{densities_str}_stc_dstc_comparison.png")

    # Check if the image file exists before trying to display
    print(f"\\nChecking for: {image_filename}")
    if os.path.exists(image_filename):
        print(f"  Displaying image: {os.path.basename(image_filename)}")
        display(Image(filename=image_filename))
    else:
        print(f"  ERROR: Image file not found.")
        print("  Please ensure the comparison script ran successfully for this workload.") 