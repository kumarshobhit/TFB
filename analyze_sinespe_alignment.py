import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import math
from scipy import signal

def get_dominant_period(data, sample_rate=1.0):
    """
    Calculates the dominant period of a time series using Periodogram.
    Returns period in number of time steps.
    """
    # Detrend to remove linear trend and mean
    data_detrended = signal.detrend(data)
    
    # Compute Power Spectral Density
    freqs, psd = signal.periodogram(data_detrended, fs=sample_rate)
    
    # Find the frequency with the highest power
    peak_idx = np.argmax(psd)
    dom_freq = freqs[peak_idx]
    
    # Avoid division by zero
    if dom_freq == 0:
        return float('inf')
        
    return 1.0 / dom_freq

def generate_sinespe_1d(length, period, d_model=128):
    """
    Generates the SineSPE encoding (1st dimension) exactly as implemented in the model.
    """
    # Replicating SineSPE logic from pos_encoding.py
    position = torch.arange(length).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
    
    if period is not None:
        # The logic in SineSPE: position % period
        periodic_position = position % period
    else:
        periodic_position = position
        
    # We visualize the first dimension (highest frequency component)
    # div_term[0] is 1.0
    # encoding[:, 0] = sin(periodic_position * div_term[0])
    encoding_dim0 = torch.sin(periodic_position * div_term[0]).numpy().flatten()
    
    return encoding_dim0

def main():
    parser = argparse.ArgumentParser(description="Analyze Dominant Frequency and SineSPE Alignment")
    parser.add_argument('--dataset', type=str, default='ETTh1', help='Dataset name (filename without .csv)')
    parser.add_argument('--window_size', type=int, default=200, help="Number of time steps to visualize")
    parser.add_argument('--d_model', type=int, default=128, help="Model dimension for encoding generation")
    parser.add_argument('--target_col', type=str, default='OT', help="Specific column to plot alignment for")
    args = parser.parse_args()
    
    # Construct path
    path = f'dataset/forecasting/{args.dataset}.csv'
    if not os.path.exists(path):
        print(f"Error: File not found at {path}")
        return
        
    print(f"Loading dataset from {path}...")
    df = pd.read_csv(path)
    
    # 1. Calculate Dominant Frequencies for all numeric channels
    print("\n" + "="*60)
    print(f"Dominant Period Analysis for {args.dataset}")
    print("="*60)
    print(f"{'Channel':<15} | {'Dom Period (Steps)':<20} | {'Dom Period (Hours)*'}")
    print("-" * 60)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    channel_periods = {}
    
    for col in numeric_cols:
        data = df[col].values
        # Calculate period
        period = get_dominant_period(data, sample_rate=1.0)
        channel_periods[col] = period
        
        print(f"{col:<15} | {period:<20.2f} | {period:.2f}")

    print("-" * 60)
    print("* Assuming 1 step = 1 hour (typical for ETT datasets)\n")
    
    # 2. Visualization
    target_col = args.target_col if args.target_col in df.columns else numeric_cols[-1]
    print(f"Generating alignment plot for channel: {target_col}")
    
    # Normalize data to [-1, 1] for visual comparison with Sine wave
    raw_data = df[target_col].values[:args.window_size]
    d_min, d_max = raw_data.min(), raw_data.max()
    norm_data = 2 * (raw_data - d_min) / (d_max - d_min) - 1
    
    # Define periods to plot
    detected_period = channel_periods[target_col]
    periods_to_plot = [24, 36, int(detected_period)]
    periods_to_plot = sorted(list(set(periods_to_plot))) # Unique and sorted
    
    plt.figure(figsize=(15, 8))
    
    # Plot Normalized Data
    plt.plot(norm_data, label=f'Normalized Data ({target_col})', color='black', linewidth=2.5, alpha=0.6)
    
    # Plot Encodings
    styles = ['-', '--', '-.', ':']
    for i, p in enumerate(periods_to_plot):
        enc = generate_sinespe_1d(args.window_size, period=p, d_model=args.d_model)
        
        label = f'SineSPE (Period={p})'
        if abs(p - detected_period) < 0.1:
            label += " [Detected]"
            
        plt.plot(enc, label=label, linestyle=styles[i % len(styles)], alpha=0.8, linewidth=1.5)
        
    plt.title(f"SineSPE Alignment Analysis - {args.dataset} - {target_col}", fontsize=14)
    plt.xlabel("Time Steps", fontsize=12)
    plt.ylabel("Value (Normalized / Sine Encoding)", fontsize=12)
    plt.legend(loc='upper right', framealpha=0.9)
    plt.grid(True, alpha=0.3)
    
    # Save Plot
    os.makedirs('results', exist_ok=True)
    out_path = f'results/{args.dataset}_{target_col}_sinespe_alignment.png'
    plt.savefig(out_path)
    print(f"Plot saved to: {out_path}")

if __name__ == '__main__':
    main()