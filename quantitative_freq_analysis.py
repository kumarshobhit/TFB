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

ETTH1_FEATURES = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']

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
    
    for i in range(pe_matrix.shape[1]):
        f, p = compute_psd(pe_matrix[:, i])
        if pe_psd_sum is None:
            pe_psd_sum = np.zeros_like(p)
            freqs = f
        pe_psd_sum += p
        
    # Normalize PE Spectrum
    return pe_psd_sum / (np.sum(pe_psd_sum) + 1e-9), freqs

def analyze_alignment(dataset_path, seq_len, d_model, period=None, limit_cols=None, pe_type='sinespe', base_freq=10000.0, n_heads=8, feature_names=None):
    print(f"\n=== Quantitative Frequency Analysis ===")
    print(f"Dataset: {dataset_path}")
    print(f"Settings: seq_len={seq_len}, d_model={d_model}, pe_type={pe_type}")
    if pe_type == 'sinespe':
        print(f"Period: {period if period else 'None (Absolute)'}")
        print(f"   -> Effective Frequency Dim: {d_model}")
    elif pe_type == 'rope':
        print(f"Base Freq: {base_freq}, n_heads: {n_heads}")
        print(f"   -> Effective Frequency Dim (head_dim): {d_model // n_heads} (Repeated {n_heads} times)")
    
    if pe_type == 'sinespe' and period is not None and seq_len < period:
        print(f"Warning: seq_len ({seq_len}) < period ({period}). The modulo operation has no effect. Result will be identical to Absolute PE.")
    
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
    
    if feature_names is not None:
        # If columns are just indices (0, 1, 2...), map them to feature names
        is_indices = all(str(c).isdigit() for c in numeric_cols)
        if is_indices and len(numeric_cols) == len(feature_names):
            print(f"Mapping numeric columns to provided feature names: {feature_names}")
            mapping = {c: name for c, name in zip(numeric_cols, feature_names)}
            df.rename(columns=mapping, inplace=True)
            # Re-select numeric columns to get the new names in order
            numeric_cols = pd.Index(feature_names)

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

    # --- Step E: Output Table ---
    print("\n--- Alignment Analysis ---")
    cols_to_show = ['feature', 'cosine_sim']
    if 'cosine_sim_default' in results_df.columns:
        cols_to_show.extend(['cosine_sim_default', 'delta_cosine'])
    
    # Filter to ensure columns exist
    cols_to_show = [c for c in cols_to_show if c in results_df.columns]
    
    if feature_names is not None:
        results_df['feature'] = pd.Categorical(results_df['feature'], categories=feature_names, ordered=True)
        print(results_df[cols_to_show].sort_values('feature').to_string())
    else:
        print(results_df[cols_to_show].sort_values(by='cosine_sim', ascending=False).to_string())

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
    args = parser.parse_args()

    feature_names = None
    if 'ETTh1' in args.dataset:
        feature_names = ETTH1_FEATURES
    
    path = f'dataset/forecasting/{args.dataset}.csv'
    analyze_alignment(path, args.seq_len, args.d_model, args.period, args.limit_cols, args.pe_type, args.base_freq, args.n_heads, feature_names)