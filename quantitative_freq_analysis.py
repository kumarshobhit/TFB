import argparse
import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import spearmanr
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

# --- 2. Helper Functions ---
def compute_psd(data, fs=1.0):
    """Compute Power Spectral Density using Periodogram."""
    # Detrending is important to remove DC offset and linear trends
    freqs, psd = signal.periodogram(data, fs=fs, detrend='linear')
    return freqs, psd

def analyze_alignment(dataset_path, seq_len, d_model, period=None):
    print(f"\n=== Quantitative Frequency Analysis ===")
    print(f"Dataset: {dataset_path}")
    print(f"Settings: seq_len={seq_len}, d_model={d_model}, period={period if period else 'None (Absolute)'}")
    
    if not os.path.exists(dataset_path):
        print(f"Error: File not found at {dataset_path}")
        return

    df = pd.read_csv(dataset_path)
    # Filter numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if 'date' in df.columns:
        numeric_cols = numeric_cols.drop('date', errors='ignore')
    
    # --- Step A: Generate PE Spectrum ---
    pe_layer = SineSPE(d_model, max_len=seq_len, period=period)
    pe_matrix = pe_layer.sine.squeeze(0).numpy() # [seq_len, d_model]
    
    # Compute PSD for each dimension of PE and sum them to get "Total PE Capacity"
    pe_psd_sum = None
    freqs = None
    
    for i in range(d_model):
        f, p = compute_psd(pe_matrix[:, i])
        if pe_psd_sum is None:
            pe_psd_sum = np.zeros_like(p)
            freqs = f
        pe_psd_sum += p
        
    # Normalize PE Spectrum (Probability Mass Function style)
    pe_psd_norm = pe_psd_sum / (np.sum(pe_psd_sum) + 1e-9)

    # --- Step B: Analyze Data Spectrum ---
    results = []
    
    print(f"\nAnalyzing {len(numeric_cols)} features...")
    total_data_psd = np.zeros_like(pe_psd_norm)

    for col in numeric_cols:
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
        # 1. Spearman Correlation: Do the peaks align in rank?
        corr, _ = spearmanr(col_psd_norm, pe_psd_norm)
        
        # 2. Cosine Similarity: Are the spectral vectors pointing in the same direction?
        cos_sim = 1 - cosine(col_psd_norm, pe_psd_norm)
        
        results.append({
            'feature': col,
            'spearman_corr': corr,
            'cosine_sim': cos_sim
        })

    # --- Step D: Output ---
    results_df = pd.DataFrame(results)
    print("\n--- Alignment Scores (Higher is Better) ---")
    print(results_df.sort_values(by='spearman_corr', ascending=False).to_string())
    
    avg_corr = results_df['spearman_corr'].mean()
    print(f"\nAverage Spearman Correlation across all features: {avg_corr:.4f}")
    print(f"Average Cosine Similarity across all features: {results_df['cosine_sim'].mean():.4f}")

    # --- Step E: Visualization ---
    avg_data_psd = total_data_psd / len(numeric_cols)
    plt.figure(figsize=(12, 6))
    plt.plot(freqs, avg_data_psd, label='Average Data Spectrum', color='blue', linewidth=2, alpha=0.7)
    plt.plot(freqs, pe_psd_norm, label='PE Spectrum', color='red', linestyle='--', linewidth=2, alpha=0.7)
    plt.title(f"Spectral Alignment (Avg Spearman: {avg_corr:.3f})")
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
    args = parser.parse_args()
    
    path = f'dataset/forecasting/{args.dataset}.csv'
    analyze_alignment(path, args.seq_len, args.d_model, args.period)