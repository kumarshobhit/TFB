import argparse
import os
import math
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import spearmanr, linregress
from scipy.spatial.distance import cosine

# --- 1. Define SineSPE (Self-contained for analysis) ---
class SineSPE(nn.Module):
    def __init__(self, d_model, max_len=5000, period=None):
        super(SineSPE, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.period = period
        self.register_buffer('sine', self._generate_sine_encoding())

    def _generate_sine_encoding(self):
        position = torch.arange(self.max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * -(math.log(10000.0) / self.d_model))
        encoding = torch.zeros(self.max_len, self.d_model)

        if self.period is not None:
            periodic_position = position % self.period
            encoding[:, 0::2] = torch.sin(periodic_position * div_term)
            encoding[:, 1::2] = torch.cos(periodic_position * div_term)
        else:
            # Absolute (Standard)
            encoding[:, 0::2] = torch.sin(position * div_term)
            encoding[:, 1::2] = torch.cos(position * div_term)
        
        return encoding.unsqueeze(0) # [1, max_len, d_model]

class RoPEGenerator(nn.Module):
    def __init__(self, d_model, max_len=5000, base_freq=10000.0, n_heads=8):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.base_freq = base_freq
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
    def generate(self):
        # theta_i = base^(-2i/d_head)
        theta = 1.0 / (self.base_freq ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        seq_idx = torch.arange(self.max_len).float()
        # Outer product to get angles
        idx_theta = torch.einsum('n,d->nd', seq_idx, theta)
        # RoPE uses both cos and sin
        cos_waves = torch.cos(idx_theta)
        sin_waves = torch.sin(idx_theta)
        # Concatenate to form the "features" of the PE
        head_waves = torch.cat([cos_waves, sin_waves], dim=1) 
        # Repeat for all heads to match d_model size
        full_waves = head_waves.repeat(1, self.n_heads)
        return full_waves.unsqueeze(0)

# --- 2. Helper Functions ---
def compute_psd(data, fs=1.0):
    """Compute Power Spectral Density using Periodogram."""
    # Detrending is important to remove DC offset and linear trends
    freqs, psd = signal.periodogram(data, fs=fs, detrend='linear')
    return freqs, psd

def get_pe_spectrum_norm(d_model, seq_len, period=None, pe_type='sinespe', base_freq=10000.0, n_heads=8):
    """Helper to generate normalized PE spectrum."""
    if pe_type == 'rope':
        gen = RoPEGenerator(d_model, max_len=seq_len, base_freq=base_freq, n_heads=n_heads)
        pe_matrix = gen.generate().squeeze(0).numpy()
    else:
        pe_layer = SineSPE(d_model, max_len=seq_len, period=period)
        pe_matrix = pe_layer.sine.squeeze(0).numpy() # [seq_len, d_model]
    
    pe_psd_sum = None
    freqs = None
    
    for i in range(d_model):
        f, p = compute_psd(pe_matrix[:, i])
        if pe_psd_sum is None:
            pe_psd_sum = np.zeros_like(p)
            freqs = f
        pe_psd_sum += p
        
    # Normalize PE Spectrum
    return pe_psd_sum / (np.sum(pe_psd_sum) + 1e-9), freqs

def analyze_alignment(dataset_path, seq_len, d_model, period=None, limit_cols=None, metrics_file=None, baseline_metrics_file=None, pe_type='sinespe', base_freq=10000.0, n_heads=8):
    print(f"\n=== Quantitative Frequency Analysis ===")
    print(f"Dataset: {dataset_path}")
    print(f"Settings: seq_len={seq_len}, d_model={d_model}, pe_type={pe_type}")
    if pe_type == 'sinespe':
        print(f"Period: {period if period else 'None (Absolute)'}")
    elif pe_type == 'rope':
        print(f"Base Freq: {base_freq}, n_heads: {n_heads}")
    
    if pe_type == 'sinespe' and period is not None and seq_len < period:
        print(f"Warning: seq_len ({seq_len}) < period ({period}). The modulo operation has no effect. Result will be identical to Absolute PE.")
    
    if metrics_file: print(f"Loading metrics from: {metrics_file}")

    if not os.path.exists(dataset_path):
        print(f"Error: File not found at {dataset_path}")
        return

    print(f"Loading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path)
    print(f"Dataset loaded. Shape: {df.shape}")
    # Check for TFB long format (date, data, cols) and pivot if necessary
    if 'cols' in df.columns and 'data' in df.columns:
        print("Detected long-format data. Pivoting to wide format...")
        df = df.pivot(index='date', columns='cols', values='data').reset_index()
    # Filter numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if 'date' in df.columns:
        numeric_cols = numeric_cols.drop('date', errors='ignore')
    
    if limit_cols is not None and limit_cols > 0:
        if len(numeric_cols) > limit_cols:
            print(f"Limiting analysis to first {limit_cols} columns (out of {len(numeric_cols)}).")
            numeric_cols = numeric_cols[:limit_cols]
    
    # --- Step A: Generate PE Spectrum ---
    # 1. Current/Aligned PE
    pe_psd_norm, freqs = get_pe_spectrum_norm(d_model, seq_len, period, pe_type, base_freq, n_heads)

    # 2. Default/Absolute PE (for comparison)
    pe_psd_norm_def = None
    if pe_type == 'sinespe' and period is not None:
        pe_psd_norm_def, _ = get_pe_spectrum_norm(d_model, seq_len, None, 'sinespe')
    elif pe_type == 'rope':
        # Compare against default base_freq 10000.0
        pe_psd_norm_def, _ = get_pe_spectrum_norm(d_model, seq_len, None, 'rope', 10000.0, n_heads)

    # --- Step B: Analyze Data Spectrum ---
    results = []
    
    print(f"\nAnalyzing {len(numeric_cols)} features...")
    total_data_psd = np.zeros_like(pe_psd_norm)

    for i, col in enumerate(numeric_cols):
        if (i+1) % 10 == 0: print(f"Processing feature {i+1}/{len(numeric_cols)}...")
        data = df[col].values
        
        # We analyze the average spectrum seen in sliding windows of size seq_len
        # to match exactly what the model sees during training/inference.
        col_psd_sum = np.zeros_like(freqs)
        count = 0
        
        # Stride of seq_len // 2 (50% overlap)
        for i in range(0, len(data) - seq_len, seq_len // 2):
            window = data[i:i+seq_len]
            if np.std(window) < 1e-6: continue # Skip flat windows
            
            # Normalize window to focus on shape/frequency, not amplitude
            window = (window - np.mean(window)) / np.std(window)
            
            f, p = compute_psd(window)
            col_psd_sum += p
            count += 1
            
        if count == 0: continue
        
        col_psd_avg = col_psd_sum / count
        col_psd_norm = col_psd_avg / (np.sum(col_psd_avg) + 1e-9)
        
        total_data_psd += col_psd_norm

        # --- Step C: Calculate Similarity ---
        # 1. Cosine Similarity (Current Config)
        cos_sim = 1 - cosine(col_psd_norm, pe_psd_norm)
        
        entry = {
            'feature': col,
            'cosine_sim': cos_sim
        }

        # 2. Cosine Similarity (Default Config) - if comparing
        if pe_psd_norm_def is not None:
            cos_sim_def = 1 - cosine(col_psd_norm, pe_psd_norm_def)
            entry['cosine_sim_default'] = cos_sim_def
            entry['delta_cosine'] = cos_sim - cos_sim_def # Positive = Aligned is better match

        results.append(entry)

    results_df = pd.DataFrame(results)

    # --- Step D: Integrate Metrics (Before Printing) ---
    if metrics_file:
        try:
            # Load Current Metrics
            with open(metrics_file, 'r') as f:
                m_data = json.load(f)
                curr_mse = np.array(m_data[0]['metrics']['MSE'])
                curr_mae = np.array(m_data[0]['metrics']['MAE'])
            
            if limit_cols and len(curr_mse) > limit_cols:
                curr_mse = curr_mse[:limit_cols]
                curr_mae = curr_mae[:limit_cols]
            
            if len(curr_mse) == len(results_df):
                results_df['MSE_Aligned'] = curr_mse
                results_df['MAE_Aligned'] = curr_mae

            if baseline_metrics_file and pe_psd_norm_def is not None:
                # Load Baseline Metrics
                with open(baseline_metrics_file, 'r') as f:
                    b_data = json.load(f)
                    base_mse = np.array(b_data[0]['metrics']['MSE'])
                    base_mae = np.array(b_data[0]['metrics']['MAE'])
                
                if limit_cols and len(base_mse) > limit_cols:
                    base_mse = base_mse[:limit_cols]
                    base_mae = base_mae[:limit_cols]

                if len(base_mse) == len(results_df):
                    results_df['MSE_Default'] = base_mse
                    results_df['MAE_Default'] = base_mae
                    # Delta Error: Positive = Improvement (Error Reduced)
                    results_df['delta_mse'] = results_df['MSE_Default'] - results_df['MSE_Aligned']
                    results_df['delta_mae'] = results_df['MAE_Default'] - results_df['MAE_Aligned']
        except Exception as e:
            print(f"Error loading metrics: {e}")

    # --- Step E: Output Table ---
    print("\n--- Alignment & Error Analysis ---")
    cols_to_show = ['feature', 'cosine_sim']
    if 'cosine_sim_default' in results_df.columns:
        cols_to_show.extend(['cosine_sim_default', 'delta_cosine'])
    if 'MSE_Aligned' in results_df.columns:
        cols_to_show.append('MSE_Aligned')
    if 'MSE_Default' in results_df.columns:
        cols_to_show.extend(['MSE_Default', 'delta_mse'])
    if 'MAE_Aligned' in results_df.columns:
        cols_to_show.append('MAE_Aligned')
    if 'MAE_Default' in results_df.columns:
        cols_to_show.extend(['MAE_Default', 'delta_mae'])
    
    # Filter to ensure columns exist
    cols_to_show = [c for c in cols_to_show if c in results_df.columns]
    print(results_df[cols_to_show].sort_values(by='cosine_sim', ascending=False).to_string())

    # --- Step F: Correlation with Error (The New Request) ---
    if 'delta_cosine' in results_df.columns and 'delta_mse' in results_df.columns:
        dc = results_df['delta_cosine'].values
        
        # --- MSE Analysis ---
        d_mse = results_df['delta_mse'].values
        print(f"\n>>> Correlation Analysis (Change in Alignment vs Change in MSE) <<<")
        corr, p_val = spearmanr(dc, d_mse)
        print(f"Spearman Correlation: {corr:.4f} (p-value: {p_val:.4f})")
        
        slope, intercept, r_value, p_value, std_err = linregress(dc, d_mse)
        print(f"\nLinear Quantification (Slope): {slope:.4f}")
        print(f"  -> Interpretation: On average, increasing Cosine Similarity by 0.1 results in an MSE reduction of {slope*0.1:.5f}.")

        # --- Scatter Plot: Delta Cosine vs Delta MSE ---
        plt.figure(figsize=(8, 6))
        plt.scatter(dc, d_mse, alpha=0.7, label='Features')
        
        # Regression Line
        x_vals = np.array([np.min(dc), np.max(dc)])
        y_vals = intercept + slope * x_vals
        plt.plot(x_vals, y_vals, color='red', linestyle='--', label=f'Fit: slope={slope:.2f}')
        
        plt.title(f"Impact of Alignment on Error (MSE)\n(Corr: {corr:.2f}, Slope: {slope:.2f})")
        plt.xlabel("Change in Cosine Similarity (Aligned - Default)")
        plt.ylabel("Change in MSE (Default - Aligned)")
        plt.axhline(0, color='gray', linewidth=0.5)
        plt.axvline(0, color='gray', linewidth=0.5)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("delta_cosine_vs_delta_mse.png")
        print(f"Saved scatter plot to: {os.path.abspath('delta_cosine_vs_delta_mse.png')}")

        # --- MAE Analysis ---
        if 'delta_mae' in results_df.columns:
            d_mae = results_df['delta_mae'].values
            print(f"\n>>> Correlation Analysis (Change in Alignment vs Change in MAE) <<<")
            corr, p_val = spearmanr(dc, d_mae)
            print(f"Spearman Correlation: {corr:.4f} (p-value: {p_val:.4f})")
            
            slope, intercept, r_value, p_value, std_err = linregress(dc, d_mae)
            print(f"\nLinear Quantification (Slope): {slope:.4f}")
            print(f"  -> Interpretation: On average, increasing Cosine Similarity by 0.1 results in an MAE reduction of {slope*0.1:.5f}.")
        
    elif 'cosine_sim' in results_df.columns and 'MSE_Aligned' in results_df.columns:
        # Absolute Correlation
        corr, p_val = spearmanr(results_df['cosine_sim'].values, results_df['MSE_Aligned'].values)
        print(f"\n>>> Correlation Analysis (Alignment vs Error) <<<")
        print(f"Spearman Correlation: {corr:.4f} (p-value: {p_val:.4f})")
        print("  -> Interpretation: Negative correlation implies Higher Alignment -> Lower Error.")

    # --- Step G: Visualization ---
    avg_data_psd = total_data_psd / len(numeric_cols)
    plt.figure(figsize=(12, 6))
    plt.plot(freqs, avg_data_psd, label='Average Data Spectrum', color='blue', linewidth=2, alpha=0.7)
    plt.plot(freqs, pe_psd_norm, label='PE Spectrum', color='red', linestyle='--', linewidth=2, alpha=0.7)
    plt.title(f"Spectral Alignment")
    plt.xlabel("Frequency (Cycles/Step)")
    plt.ylabel("Normalized Power Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("spectral_alignment.png")
    print(f"Saved visualization to: {os.path.abspath('spectral_alignment.png')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ETTh1')
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--period', type=int, default=None, help='Period for SineSPE. Leave empty for Absolute.')
    parser.add_argument('--pe_type', type=str, default='sinespe', choices=['sinespe', 'rope'], help='Type of PE to analyze')
    parser.add_argument('--base_freq', type=float, default=10000.0, help='Base frequency for RoPE')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of heads for RoPE')
    parser.add_argument('--limit_cols', type=int, default=10, help='Limit number of columns to analyze (default: 10). Set to 0 for all.')
    parser.add_argument('--metrics_file', type=str, default=None, help='Path to JSON file with model errors (from extract_results.py)')
    parser.add_argument('--baseline_metrics_file', type=str, default=None, help='Path to JSON file with baseline model errors')
    args = parser.parse_args()
    
    path = f'dataset/forecasting/{args.dataset}.csv'
    analyze_alignment(path, args.seq_len, args.d_model, args.period, args.limit_cols, args.metrics_file, args.baseline_metrics_file, args.pe_type, args.base_freq, args.n_heads)