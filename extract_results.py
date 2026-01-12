import pandas as pd
import ast
import json
import argparse
import os

def parse_log_info(log_info_str):
    """
    Parses the log_info string to extract MSE and MAE lists.
    Expected format: "Per-Feature MSE: [...]; Per-Feature MAE: [...]"
    """
    if not isinstance(log_info_str, str) or "Per-Feature MSE" not in log_info_str:
        return None
    
    try:
        parts = log_info_str.split(';')
        mse_part = parts[0].split('Per-Feature MSE:')[1].strip()
        mae_part = parts[1].split('Per-Feature MAE:')[1].strip()
        
        # ast.literal_eval safely evaluates a string containing a Python literal (like a list)
        mse_list = ast.literal_eval(mse_part)
        mae_list = ast.literal_eval(mae_part)
        
        return {"MSE": mse_list, "MAE": mae_list}
    except Exception as e:
        print(f"Error parsing log_info: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Extract per-feature metrics from results CSV.")
    parser.add_argument("file_path", type=str, help="Path to the results CSV file.")
    parser.add_argument("--output", type=str, default="per_feature_metrics.json", help="Output JSON file path.")
    args = parser.parse_args()

    if not os.path.exists(args.file_path):
        print(f"Error: File not found at {args.file_path}")
        return

    print(f"Reading {args.file_path}...")
    try:
        df = pd.read_csv(args.file_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    if "log_info" not in df.columns:
        print("Error: Column 'log_info' not found in CSV.")
        return

    results = []

    for index, row in df.iterrows():
        file_name = row.get("file_name", f"row_{index}")
        log_info = row.get("log_info")
        
        metrics = parse_log_info(log_info)
        if metrics:
            entry = {"file_name": file_name, "metrics": metrics}
            if "model_hyper_params" in df.columns:
                entry["model_hyper_params"] = row["model_hyper_params"]
            results.append(entry)

    if results:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Success! Extracted metrics for {len(results)} entries to {args.output}")
    else:
        print("No valid per-feature metrics found.")

if __name__ == "__main__":
    main()