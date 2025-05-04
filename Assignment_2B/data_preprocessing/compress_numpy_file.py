"""
SCATS Traffic Prediction - NPY to NPZ Converter
This script converts existing NPY files to compressed NPZ format to reduce file sizes.
"""

import os
import numpy as np
import glob
from tqdm import tqdm  # Optional, for progress bars


def convert_npy_to_npz(input_dir="processed_data", output_dir=None):
    """
    Convert all .npy files in a directory to compressed .npz files.

    Args:
        input_dir: Directory containing .npy files
        output_dir: Directory to save .npz files (defaults to input_dir)
    """
    if output_dir is None:
        output_dir = input_dir

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Find all .npy files in input directory
    npy_files = glob.glob(os.path.join(input_dir, "*.npy"))

    print(f"Found {len(npy_files)} .npy files to convert")

    # Convert each file
    for npy_file in tqdm(npy_files, desc="Converting files"):
        # Get base filename without extension
        base_name = os.path.splitext(os.path.basename(npy_file))[0]

        # Output path for npz file
        npz_file = os.path.join(output_dir, f"{base_name}.npz")

        try:
            # Load the numpy array
            data = np.load(npy_file)

            # Save as compressed npz
            np.savez_compressed(npz_file, data=data)

            # Calculate compression ratio
            original_size = os.path.getsize(npy_file)
            compressed_size = os.path.getsize(npz_file)
            ratio = original_size / compressed_size

            print(f"Converted {npy_file} to {npz_file}")
            print(
                f"Size reduction: {original_size/1024/1024:.2f} MB → {compressed_size/1024/1024:.2f} MB ({ratio:.2f}x)"
            )

            # Optionally delete original file to save space
            # os.remove(npy_file)

        except Exception as e:
            print(f"Error converting {npy_file}: {e}")

    print("Conversion completed!")


if __name__ == "__main__":
    # You can specify your directories here
    input_directory = "processed_data"
    output_directory = "processed_data"

    convert_npy_to_npz(input_directory, output_directory)

    print("\nReminder: Update your code to load from .npz files using:")
    print("data = np.load('file.npz')['data']")
