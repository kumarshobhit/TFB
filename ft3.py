import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
import argparse
import torch
import math

# Constants
SAMPLES_PER_DAY = 24
SAMPLES_PER_WEEK = 7 * SAMPLES_PER_DAY # 168 samples

# --- 1. Encoding Implementations ---

def sinespe_phase(time_steps_hours: np.ndarray, period_days: float) -> np.ndarray:
    """
    Returns the 'Sawtooth' phase used by SineSPE: (t % period).
    This is the clearest way to see alignment.
    """
    period_hours = period_days * SAMPLES_PER_DAY
    return time_steps_hours % period_hours

def sinespe_encoding_dim0(time_steps_hours: np.ndarray, period_days: float, d_model: int = 128) -> np.ndarray:
    """
    Returns the first dimension (highest freq sine) of the SineSPE encoding.
    """
    period_hours = period_days * SAMPLES_PER_DAY
    
    # Logic from pos_encoding.py
    position = torch.tensor(time_steps_hours).float().unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
    
    periodic_position = position % period_hours
    
    # sin(pos * div_term[0]) -> div_term[0] is 1.0
    return torch.sin(periodic_position * div_term[0]).numpy().flatten()

# --- 2. Analysis Function ---

def run_encoding_distance_analysis(
    df: pd.DataFrame, 
    time_column: str, 
    target_column: str,
    period_aligned: float, 
    period_misaligned: float
) -> pd.DataFrame:
    """
    Generates encoding values and normalizes data for overlay comparison.
    """
    df_analysis = pd.DataFrame(index=df.index)
    
    # Calculate the base numeric time index (in hours)
    df_analysis['time_numeric'] = (df[time_column] - df[time_column].min()).dt.total_seconds() / 3600
    time_steps_hours = df_analysis['time_numeric'].values

    # Normalize the actual data for visualization overlay (Min-Max scaling to 0-1 roughly, or Z-score)
    data_vals = pd.to_numeric(df[target_column], errors='coerce').values
    d_min = np.nanmin(data_vals)
    d_max = np.nanmax(data_vals)
    df_analysis['data_norm'] = (data_vals - d_min) / (d_max - d_min + 1e-8)
    
    # --- SineSPE Phase (Sawtooth) ---
    # This shows the "reset" of the period
    period_hours_aligned = period_aligned * SAMPLES_PER_DAY
    df_analysis['phase_aligned'] = sinespe_phase(time_steps_hours, period_aligned) / period_hours_aligned # Scale to 0-1
    df_analysis['phase_misaligned'] = sinespe_phase(time_steps_hours, period_misaligned) / (period_misaligned * SAMPLES_PER_DAY)
        
    return df_analysis

def main():
    parser = argparse.ArgumentParser(description="Visually analyze the impact of frequency alignment on time features.")
    parser.add_argument(
        '--dataset_name', 
        type=str, 
        default='ETTh1',
        help='Name of the dataset file (e.g., "ETTh1"). Script assumes file is located at ./dataset/forecasting/{name}.csv'
    )
    parser.add_argument(
        '--target_col',
        type=str,
        default='OT',
        help='Name of the column to visualize (e.g., "OT" or "data").'
    )
    parser.add_argument(
        '--target_period_days',
        type=float,
        default=7.0,
        help='The known dominant period of the dataset in DAYS (P_aligned, e.g., 7.0 for weekly).'
    )
    parser.add_argument(
        '--mismatched_period_days',
        type=float,
        default=4.0, 
        help='A purposely mismatched encoding period in DAYS (P_misaligned, e.g., 4.0).'
    )
    args = parser.parse_args()

    # --- Setup ---
    DATA_FILEPATH = f'./dataset/forecasting/{args.dataset_name}.csv'
    RESULTS_DIR = 'results' 
    os.makedirs(RESULTS_DIR, exist_ok=True) 

    # --- Data Loading ---
    try:
        df = pd.read_csv(DATA_FILEPATH)
        TIME_COLUMN = 'date'
        if TIME_COLUMN not in df.columns:
            raise ValueError(f"Required time column '{TIME_COLUMN}' not found in dataset.")
        
        if args.target_col not in df.columns:
             # Fallback to last column if specified one missing
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if not numeric_cols.empty:
                args.target_col = numeric_cols[-1]
                print(f"Warning: Target column not found. Using last numeric column: {args.target_col}")
            else:
                print(f"Warning: {args.target_col} not found. Using last column.")
                args.target_col = df.columns[-1]

        df[TIME_COLUMN] = pd.to_datetime(df[TIME_COLUMN])
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # --- Plotting Setup ---
    # Focus on a 3-week window for clarity
    plot_start = df[TIME_COLUMN].min()
    plot_end = plot_start + pd.Timedelta(days=args.target_period_days * 3)
    df_plot = df[(df[TIME_COLUMN] >= plot_start) & (df[TIME_COLUMN] <= plot_end)].copy()
    
    # Get all analysis data
    analysis_df = run_encoding_distance_analysis(
        df_plot, TIME_COLUMN, args.target_col, args.target_period_days, args.mismatched_period_days
    )
    
    # 2x1 grid: Aligned vs Misaligned
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(
        f'Sanity Check: SineSPE Alignment with Dataset (Dataset: {args.dataset_name}, Col: {args.target_col})', 
        fontsize=16
    )

    # --- Plot 1: Aligned ---
    axes[0].plot(df_plot[TIME_COLUMN], analysis_df['data_norm'], label='Data (Normalized)', color='black', alpha=0.6, linewidth=1.5)
    axes[0].plot(df_plot[TIME_COLUMN], analysis_df['phase_aligned'], label=f'SineSPE Phase (Period={args.target_period_days}d)', color='blue', linestyle='--', alpha=0.8)
    axes[0].set_title(f'ALIGNED: Period = {args.target_period_days} Days', fontsize=14)
    axes[0].set_ylabel('Normalized Value / Phase')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # --- Plot 2: Misaligned ---
    axes[1].plot(df_plot[TIME_COLUMN], analysis_df['data_norm'], label='Data (Normalized)', color='black', alpha=0.6, linewidth=1.5)
    axes[1].plot(df_plot[TIME_COLUMN], analysis_df['phase_misaligned'], label=f'SineSPE Phase (Period={args.mismatched_period_days}d)', color='red', linestyle='--', alpha=0.8)
    axes[1].set_title(f'MISALIGNED: Period = {args.mismatched_period_days} Days', fontsize=14)
    axes[1].set_ylabel('Normalized Value / Phase')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, linestyle='--', alpha=0.6)

    axes[1].set_xlabel(f"Time ({args.target_period_days * 3:.1f} Day Window)")

    plt.tight_layout()

    # --- Save Plot ---
    output_filename = f'{args.dataset_name}_SineSPE_Alignment_Check.png'

    # --- Save Plot ---
    output_filename = f'{args.dataset_name}_Encoding_Comparison_Full.png'
    output_filepath = Path(RESULTS_DIR) / output_filename
    plt.savefig(output_filepath)
    print(f"\nSuccessfully generated and saved visual analysis to {output_filepath}")
    # plt.show() # Comment out plt.show() if running in a non-GUI environment

if __name__ == "__main__":
    main()
