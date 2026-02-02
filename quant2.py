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

# Standard Features for ETTh1
ETTH1_FEATURES = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']

# --- 1. Define Encoding Architectures ---
class SineSPE(nn.Module):
    def __init__(self, d_model, max_len=5000, period=None):
        super(SineSPE, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.period = period
        self.register_buffer('sine', self._generate_sine_encoding())

    def _generate_sine_encoding(self):
        # Standard PE generation
        position = torch.arange(self.max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * -(math.log(10000.0) / self.d_model))
        encoding = torch.zeros(self.max_len, self.d_model)

        if self.period is not None:
            # EXPLICIT BIAS: Cyclic position
            periodic_position = position % self.period
            encoding[:, 0::2] = torch.sin(periodic_position * div_term)
            encoding[:, 1::2] = torch.cos(periodic_position * div_term)
        else:
            # ABSOLUTE BIAS: Linear position
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
        # IMPLICIT BIAS: Rotational Frequencies
        theta = 1.0 / (self.base_freq ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        seq_idx = torch.arange(self.max_len).float()
        
        idx_theta = torch.einsum('n,d->nd', seq_idx, theta)
        
        cos_waves = torch.cos(idx_theta)
        sin_waves = torch.sin(idx_theta)
        
        # Repeat across heads to match d_model
        head_waves = torch.cat([cos_waves, sin_waves], dim=1) 
        full_waves = head_waves.repeat(1, self.n_heads)
        return full_waves.unsqueeze(0)

# --- 2. Spectral Analysis Helper ---
def compute_psd(data, fs=1.0):
    """Compute Power Spectral Density using Welch's method or Periodogram."""
    freqs, psd = signal.periodogram(data, fs=fs, detrend='linear')
    return freqs, psd

def get_pe_spectrum_norm(d_model, seq_len, period=None, pe_type='sinespe', base_freq=10000.0, n_heads=8):
    """
    Generate the 'Theoretical Target Spectrum' for the given PE configuration.
    This acts as the Reference for 'Spectral Alignment'.
    """
    if pe_type == 'rope':
        # Generate RoPE waves
        gen = RoPEGenerator(d_model, max_len=seq_len, base_freq=base_freq, n_heads=n_heads)
        pe_matrix = gen.generate().squeeze(0).numpy()
        
    elif pe_type == 'convspe':
        # PROXY: For ConvSPE, the "Target Bias" is the Resonance Period (Kernel Size).
        # We use SineSPE(period=kernel_size) to represent this target.
        if period is None: period = 24 # Default fallback
        print(f"  [Info] ConvSPE: Using SineSPE(period={period}) as proxy for Target Resonance.")
        pe_layer = SineSPE(d_model, max_len=seq_len, period=period)
        pe_matrix = pe_layer.sine.squeeze(0).numpy()
        
    else: # sinespe
        # Standard SineSPE
        pe_layer = SineSPE(d_model, max_len=seq_len, period=period)
        pe_matrix = pe_layer.sine.squeeze(0).numpy()
    
    pe_psd_sum = None
    freqs = None
    
    # Sum PSD across all dimensions (Aggregate Inductive Bias)
    for i in range(d_model):
        f, p = compute_psd(pe_matrix[:, i])
        if pe_psd_sum is None:
            pe_psd_sum = np.zeros_like(p)
            freqs = f
        pe_psd_sum += p
        
    # Normalize
    return pe_psd_sum / (np.sum(pe_psd_sum) + 1e-9), freqs

def analyze_alignment(dataset_path, seq_len, d_model, period=None, limit_cols=None, pe_type='sinespe', base_freq=10000.0, n_heads=8, feature_names=None):
    print(f"\n=== Quantitative Frequency Analysis ===")
    print(f"Dataset: {dataset_path}")
    print(f"Config: Type={pe_type} | Period/Ker={period} | BaseFreq={base_freq}")
    
    # --- Step A: Generate Target PE Spectrums ---
    # 1. Aligned PE (The "New" Model Configuration)
    pe_psd_norm, freqs = get_pe_spectrum_norm(d_model, seq_len, period, pe_type, base_freq, n_heads)

    # 2. Default PE (The "Baseline" Model Configuration)
    # SineSPE -> Absolute (No Period)
    # RoPE -> Base 10000
    # ConvSPE -> Absolute (K=3 approx)
    pe_psd_norm_def = None
    
    if pe_type == 'sinespe':
        pe_psd_norm_def, _ = get_pe_spectrum_norm(d_model, seq_len, None, 'sinespe')
    elif pe_type == 'convspe':
        pe_psd_norm_def, _ = get_pe_spectrum_norm(d_model, seq_len, None, 'sinespe') 
    elif pe_type == 'rope':
        pe_psd_norm_def, _ = get_pe_spectrum_norm(d_model, seq_len, None, 'rope', 10000.0, n_heads)

    # --- Step B: Load & Analyze Data Spectrum ---
    if not os.path.exists(dataset_path):
        print("Error: Dataset not found.")
        return

    df = pd.read_csv(dataset_path)
    if 'cols' in df.columns and 'data' in df.columns:
        df = df.pivot(index='date', columns='cols', values='data').reset_index()
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if 'date' in df.columns: numeric_cols = numeric_cols.drop('date', errors='ignore')
    
    # Rename features if provided
    if feature_names is not None and len(numeric_cols) == len(feature_names):
        df.rename(columns={c: n for c, n in zip(numeric_cols, feature_names)}, inplace=True)
        numeric_cols = pd.Index(feature_names)

    if limit_cols and limit_cols > 0: numeric_cols = numeric_cols[:limit_cols]
    
    results = []
    print(f"\nAnalyzing {len(numeric_cols)} features...")

    for i, col in enumerate(numeric_cols):
        data = df[col].values
        col_psd_sum = np.zeros_like(freqs)
        count = 0
        
        # Sliding Window Analysis (Matching Model Context)
        for j in range(0, len(data) - seq_len, seq_len // 2):
            window = data[j:j+seq_len]
            if np.std(window) < 1e-6: continue
            window = (window - np.mean(window)) / np.std(window)
            f, p = compute_psd(window)
            col_psd_sum += p
            count += 1
            
        if count == 0: continue
        col_psd_norm = (col_psd_sum / count) / (np.sum(col_psd_sum / count) + 1e-9)
        
        # 1. Similarity to Aligned PE
        cos_sim = 1 - cosine(col_psd_norm, pe_psd_norm)
        
        entry = {'feature': col, 'cosine_sim': cos_sim}

        # 2. Similarity to Default PE
        if pe_psd_norm_def is not None:
            cos_sim_def = 1 - cosine(col_psd_norm, pe_psd_norm_def)
            entry['cosine_sim_default'] = cos_sim_def
            entry['delta_cosine'] = cos_sim - cos_sim_def # Positive = Improved Alignment

        results.append(entry)

    results_df = pd.DataFrame(results)

    print("\n--- Alignment Analysis Results ---")
    print(results_df[['feature', 'cosine_sim', 'delta_cosine']].to_string())

    return results_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Configuration
    parser.add_argument('--dataset', type=str, default='dataset/forecasting/ETTh1.csv')
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--period', type=int, default=None, help='Target Period (Sine) or Kernel Size (Conv)')
    parser.add_argument('--pe_type', type=str, default='sinespe', choices=['sinespe', 'rope', 'convspe'])
    parser.add_argument('--base_freq', type=float, default=10000.0, help='Base frequency for RoPE')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of heads for RoPE')
    parser.add_argument('--limit_cols', type=int, default=10, help='Limit number of columns to analyze')
    
    args = parser.parse_args()

    # Use ETTh1 feature names if applicable
    feature_names = None
    if 'ETTh1' in args.dataset:
        feature_names = ETTH1_FEATURES
    
    analyze_alignment(
        args.dataset, 
        args.seq_len, 
        args.d_model, 
        period=args.period, 
        limit_cols=args.limit_cols, 
        pe_type=args.pe_type, 
        base_freq=args.base_freq, 
        n_heads=args.n_heads,
        feature_names=feature_names
    )