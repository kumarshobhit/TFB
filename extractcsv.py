import base64
import os
import argparse
import pickle
import time

import numpy as np
import pandas as pd


def to_csv(data, save_dir: str, save_name: str):
    """
    Save the input data (either a DataFrame or a 3D NumPy array) into CSV file(s).

    If the input is a pandas DataFrame, it will be saved directly to the given directory.
    If the input is a 3D NumPy array (with shape [num, time, dim]), each 2D slice (data[i])
    will be saved into a separate subdirectory named 'sample_i'.

    :param data: The data to save, either a pandas DataFrame or a NumPy array of shape (num, time, dim).
    :param save_dir: The root directory where the files will be saved.
    :param save_name: The name of the CSV file(s) to be written.
    :raises TypeError: If the input data type is not supported.
    """
    os.makedirs(save_dir, exist_ok=True)

    if isinstance(data, pd.DataFrame):
        data.to_csv(os.path.join(save_dir, save_name), index=False)

    elif isinstance(data, np.ndarray):
        num = data.shape[0]
        for i in range(num):
            sample_dir = os.path.join(save_dir, f"sample_{i}")
            os.makedirs(sample_dir, exist_ok=True)
            df = pd.DataFrame(data[i])
            df.to_csv(os.path.join(sample_dir, save_name), index=False)

    elif isinstance(data, list):
        num = len(data)

        for i in range(num):
            sample_dir = os.path.join(save_dir, f"sample_{i}")
            os.makedirs(sample_dir, exist_ok=True)
            df = data[i]
            df.to_csv(os.path.join(sample_dir, save_name), index=False)

    else:
        raise TypeError("Unsupported type for data. Must be pd.DataFrame or np.ndarray.")


def calculate_per_feature_metrics(save_dir: str):
    """
    Calculate and save per-feature metrics (WAPE, MAE, MSE) from the extracted CSVs.
    Aggregates errors across all samples to ensure correct WAPE calculation.
    """
    print(f"Calculating per-feature metrics for {save_dir}...")

    total_abs_diff = None
    total_abs_actual = None
    total_sq_diff = None
    total_rows = 0
    all_metrics = []

    # Identify if data is in sample subdirectories or root
    sample_dirs = [d for d in os.listdir(save_dir) if os.path.isdir(os.path.join(save_dir, d)) and d.startswith("sample_")]
    
    # If no sample dirs, treat the save_dir itself as the source (case for single DataFrame output)
    iterator = sample_dirs if sample_dirs else ["."]

    for subdir in iterator:
        pred_path = os.path.join(save_dir, subdir, "inference_data.csv")
        actual_path = os.path.join(save_dir, subdir, "actual_data.csv")

        if not os.path.exists(pred_path) or not os.path.exists(actual_path):
            print(f"Warning: Missing prediction or actual data in {subdir}. Skipping.")
            continue

        try:
            pred_df = pd.read_csv(pred_path)
            actual_df = pd.read_csv(actual_path)
        except Exception as e:
            print(f"Error reading CSVs in {subdir}: {e}. Skipping.")
            continue

        # Calculate differences
        try:
            diff = actual_df - pred_df
            abs_diff = diff.abs()

            if total_abs_diff is None:
                total_abs_diff = abs_diff.sum()
                total_abs_actual = actual_df.abs().sum()
                total_sq_diff = (diff ** 2).sum()
            else:
                total_abs_diff += abs_diff.sum()
                total_abs_actual += actual_df.abs().sum()
                total_sq_diff += (diff ** 2).sum()

            total_rows += len(actual_df)
        except Exception as e:
            print(f"Error calculating metrics in {subdir}: {e}. Skipping.")
            continue

    if total_rows > 0:
        try:
             metrics_df = pd.DataFrame({
                'WAPE': (total_abs_diff / (total_abs_actual + 1e-9)),
                'MAE': total_abs_diff / total_rows,
                'MSE': total_sq_diff / total_rows
            })
        except Exception as e:
            print(f"Error creating metrics DataFrame: {e}")
            return

        output_path = os.path.join(save_dir, "per_feature_metrics.csv")

        try:
            metrics_df.to_csv(output_path)
            print(f"Per-feature metrics saved to: {output_path}")
        except Exception as e:
            print(f"Error saving metrics to CSV: {e}")

    else:
        print("No valid data found for metric calculation.")


def decode_data(filepath: str):
    """
    Load the result CSV file and decode the base64-encoded 'inference_data' and 'actual_data' columns.

    :param filepath: Path to the input CSV file containing encoded data.
    :return: None. The decoded data will be saved as CSV files in corresponding folders.
    """
    try:
        data = pd.read_csv(filepath)  # Read the CSV file with encoded columns
    except Exception as e:
        print(f"Error reading {filepath}: {e}")

    for index, row in data.iterrows():
        # Decode base64 strings and deserialize them back to original DataFrames
        decoded_inference_data = base64.b64decode(row["inference_data"])
        decoded_actual_data = base64.b64decode(row["actual_data"])
        inference_data = pickle.loads(decoded_inference_data)
        actual_data = pickle.loads(decoded_actual_data)

        # Construct directory name by removing special characters from model parameters
        file_name = os.path.splitext(row["file_name"])[0]
        model_name = row["model_name"]
        model_params = row["model_params"].translate(str.maketrans('', '', '":, {}'))
        save_dir = f"{file_name}_{model_name}_{model_params}"
        timestamp = f"{int(time.time() * 1000)}"
        save_dir = os.path.join(save_dir, timestamp)
        print(f"Saving data to directory: {save_dir}")

        # Save the decoded data as CSV files
        to_csv(inference_data, save_dir, "inference_data.csv")
        to_csv(actual_data, save_dir, "actual_data.csv")

        # Calculate per-feature metrics
        calculate_per_feature_metrics(save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decode and process data from a CSV file.")

    parser.add_argument("--extract-wape", action="store_true", help="Extract WAPE values per feature.")

    parser.add_argument("input_file", help="Path to the input CSV file.")
    # parser.add_argument("--output_dir", help="Base directory for output (optional, defaults to same directory as input)", default=None) # Removed as the output directory is determined inside decode_data

    args = parser.parse_args()

    input_file_path = args.input_file
    # output_directory = args.output_dir # Removed as the output directory is determined inside decode_data
    
    decode_data(input_file_path)

# Example usage:
# python extractcsv.py /path/to/your/input.csv
# The output will be saved in a directory created based on the input file's contents.