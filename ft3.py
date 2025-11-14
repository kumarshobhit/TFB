import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
import argparse

# Constants
SAMPLES_PER_DAY = 24
SAMPLES_PER_WEEK = 7 * SAMPLES_PER_DAY # 168 samples

# --- 1. Encoding Implementations ---

def sine_cosine_pe(time_steps_hours: np.ndarray, period_days: float) -> np.ndarray:
    """Standard Sinusoidal Positional Encoding (relative to period)."""
    # Calculate the scale factor (frequency) based on the input period
    period_hours = period_days * SAMPLES_PER_DAY
    # The time steps are in hours
    freq = 2 * np.pi / period_hours
    
    sin_val = np.sin(freq * time_steps_hours)
    cos_val = np.cos(freq * time_steps_hours)
    return np.stack([sin_val, cos_val], axis=-1)

def tupe_encoding(time_steps_hours: np.ndarray, period_days: float) -> np.ndarray:
    """
    CORRECTED: TUPE-like relative encoding based on time distance/delta.
    This now creates a cyclical "sawtooth" wave.
    """
    period_hours = period_days * SAMPLES_PER_DAY
    # Use modulo to create a repeating pattern from 0 to period_hours
    relative_time_steps = time_steps_hours % period_hours
    return relative_time_steps

def absolute_pe(time_steps_hours: np.ndarray) -> np.ndarray:
    """Simple Absolute Positional Encoding (Linear, frequency-agnostic)."""
    # Represents position relative to the start of the sequence.
    return time_steps_hours

# --- 2. Analysis Function ---

def run_encoding_distance_analysis(
    df: pd.DataFrame, 
    time_column: str, 
    period_aligned: float, 
    period_misaligned: float
) -> pd.DataFrame:
    """
    Analyzes the encoded distance/representation for a given function across aligned and misaligned frequencies.
    """
    df_analysis = pd.DataFrame(index=df.index)
    
    # Calculate the base numeric time index (in hours)
    df_analysis['time_numeric'] = (df[time_column] - df[time_column].min()).dt.total_seconds() / 3600
    time_steps_hours = df_analysis['time_numeric'].values
    
    # --- Sine/Cosine PE ---
    aligned_sc = sine_cosine_pe(time_steps_hours, period_aligned)
    misaligned_sc = sine_cosine_pe(time_steps_hours, period_misaligned)
    df_analysis['sc_aligned'] = aligned_sc[:, 0] # Use Sine component for visualization
    df_analysis['sc_misaligned'] = misaligned_sc[:, 0]
    
    # --- TUPE-like (Corrected) ---
    aligned_tupe = tupe_encoding(time_steps_hours, period_aligned)
    misaligned_tupe = tupe_encoding(time_steps_hours, period_misaligned)
    df_analysis['tupe_aligned'] = aligned_tupe
    df_analysis['tupe_misaligned'] = misaligned_tupe
        
    # --- Absolute PE ---
    absolute_vals = absolute_pe(time_steps_hours)
    df_analysis['abs_aligned'] = absolute_vals
    df_analysis['abs_misaligned'] = absolute_vals # No change for misaligned
        
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
        df_plot, TIME_COLUMN, args.target_period_days, args.mismatched_period_days
    )
    
    # 3x2 grid for 3 encodings, comparing aligned vs. misaligned
    fig, axes = plt.subplots(3, 2, figsize=(14, 15), sharex=True)
    fig.suptitle(
        f'Impact of Frequency Alignment on Positional Encodings (Dataset: {args.dataset_name})'
        f'\nTrue Dominant Period: {args.target_period_days} Days | Misaligned Period: {args.mismatched_period_days} Days', 
        fontsize=16
    )

    # --- Row 1: Sine/Cosine PE ---
    axes[0, 0].plot(df_plot[TIME_COLUMN], analysis_df['sc_aligned'])
    axes[0, 0].set_title(f'Sine/Cosine PE - ALIGNED (P={args.target_period_days:.1f} Days)', fontsize=12)
    axes[0, 0].set_ylabel('Encoded Value')
    axes[0, 0].grid(True, linestyle='--', alpha=0.6)

    axes[0, 1].plot(df_plot[TIME_COLUMN], analysis_df['sc_misaligned'], color='red')
    axes[0, 1].set_title(f'Sine/Cosine PE - MISALIGNED (P={args.mismatched_period_days:.1f} Days)', fontsize=12, color='red')
    axes[0, 1].set_ylabel('Encoded Value')
    axes[0, 1].grid(True, linestyle='--', alpha=0.6)

    # --- Row 2: TUPE-like (Corrected) ---
    axes[1, 0].plot(df_plot[TIME_COLUMN], analysis_df['tupe_aligned'])
    axes[1, 0].set_title(f'TUPE-like (Relative) - ALIGNED (P={args.target_period_days:.1f} Days)', fontsize=12)
    axes[1, 0].set_ylabel('Encoded Value (Hours)')
    axes[1, 0].grid(True, linestyle='--', alpha=0.6)

    axes[1, 1].plot(df_plot[TIME_COLUMN], analysis_df['tupe_misaligned'], color='red')
    axes[1, 1].set_title(f'TUPE-like (Relative) - MISALIGNED (P={args.mismatched_period_days:.1f} Days)', fontsize=12, color='red')
    axes[1, 1].set_ylabel('Encoded Value (Hours)')
    axes[1, 1].grid(True, linestyle='--', alpha=0.6)

    # --- Row 3: Absolute PE ---
    axes[2, 0].plot(df_plot[TIME_COLUMN], analysis_df['abs_aligned'])
    axes[2, 0].set_title(f'Absolute PE (Control) - ALIGNED (P={args.target_period_days:.1f} Days)', fontsize=12)
    axes[2, 0].set_ylabel('Encoded Value (Hours)')
    axes[2, 0].grid(True, linestyle='--', alpha=0.6)

    axes[2, 1].plot(df_plot[TIME_COLUMN], analysis_df['abs_misaligned'], linestyle='--', color='gray')
    axes[2, 1].set_title(f'Absolute PE (Control) - MISALIGNED (No Change)', fontsize=12)
    axes[2, 1].set_ylabel('Encoded Value (Hours)')
    axes[2, 1].grid(True, linestyle='--', alpha=0.6)

    # Set common X-axis label
    axes[-1, 0].set_xlabel(f"Time ({args.target_period_days * 3:.1f} Day Window)")
    axes[-1, 1].set_xlabel(f"Time ({args.target_period_days * 3:.1f} Day Window)")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust space for suptitle

    # --- Save Plot ---
    output_filename = f'{args.dataset_name}_Encoding_Comparison_Full.png'
    output_filepath = Path(RESULTS_DIR) / output_filename
    plt.savefig(output_filepath)
    print(f"\nSuccessfully generated and saved visual analysis to {output_filepath}")
    # plt.show() # Comment out plt.show() if running in a non-GUI environment

if __name__ == "__main__":
    main()

