import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def visualize_sample(directory: str):
    """
    Visualizes actual vs predicted data from CSV files in the specified directory.
    """
    actual_path = os.path.join(directory, "actual_data.csv")
    pred_path = os.path.join(directory, "inference_data.csv")

    if not os.path.exists(actual_path) or not os.path.exists(pred_path):
        print(f"Error: CSV files not found in {directory}")
        return

    # Load data
    df_actual = pd.read_csv(actual_path)
    df_pred = pd.read_csv(pred_path)

    # Identify columns (variates)
    columns = df_actual.columns
    num_vars = len(columns)

    # Create subplots
    fig, axes = plt.subplots(num_vars, 1, figsize=(12, 3 * num_vars), sharex=True)
    if num_vars == 1:
        axes = [axes]

    for i, col in enumerate(columns):
        ax = axes[i]
        ax.plot(df_actual[col], label="Actual", color="black", linewidth=1.5)
        if col in df_pred.columns:
            ax.plot(df_pred[col], label="Predicted", color="red", linestyle="--", linewidth=1.5)
        
        ax.set_title(f"Variable: {col}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylabel("Value")

    axes[-1].set_xlabel("Time Step")
    plt.tight_layout()

    # Save plot
    output_file = os.path.join(directory, "actual_vs_pred.png")
    plt.savefig(output_file)
    print(f"Plot saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Actual vs Predicted results.")
    parser.add_argument("dir", nargs="?", default=".", help="Directory containing the CSV files")
    args = parser.parse_args()

    visualize_sample(args.dir)