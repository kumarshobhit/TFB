import base64
import os
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

    # Identify if data is in sample subdirectories or root
    sample_dirs = [d for d in os.listdir(save_dir) if os.path.isdir(os.path.join(save_dir, d)) and d.startswith("sample_")]
    
    # If no sample dirs, treat the save_dir itself as the source (case for single DataFrame output)
    iterator = sample_dirs if sample_dirs else ["."]

    for subdir in iterator:
        pred_path = os.path.join(save_dir, subdir, "inference_data.csv")
        actual_path = os.path.join(save_dir, subdir, "actual_data.csv")

        if not os.path.exists(pred_path) or not os.path.exists(actual_path):
            continue

        pred_df = pd.read_csv(pred_path)
        actual_df = pd.read_csv(actual_path)

        # Calculate differences
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

    if total_rows > 0:
        metrics_df = pd.DataFrame({
            'WAPE': (total_abs_diff / (total_abs_actual + 1e-9)),
            'MAE': total_abs_diff / total_rows,
            'MSE': total_sq_diff / total_rows
        })
        output_path = os.path.join(save_dir, "per_feature_metrics.csv")
        metrics_df.to_csv(output_path)
        print(f"Per-feature metrics saved to: {output_path}")


def decode_data(filepath: str):
    """
    Load the result CSV file and decode the base64-encoded 'inference_data' and 'actual_data' columns.

    :param filepath: Path to the input CSV file containing encoded data.
    :return: None. The decoded data will be saved as CSV files in corresponding folders.
    """
    data = pd.read_csv(filepath)  # Read the CSV file with encoded columns

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


# Example usage
your_result_csv_path = r"TST.1768886354.nrgpu1.3478941.csv"
decode_data(your_result_csv_path)