import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import math
import argparse
import os
from scipy.signal import find_peaks

# ==========================================
# 1. POSITIONAL ENCODING CLASSES (From your file)
# ==========================================

class SineSPE(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512, period=None):
        super(SineSPE, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.period = period
        self.register_buffer('sine', self._generate_sine_encoding())

    def _generate_sine_encoding(self):
        position = torch.arange(self.max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * -(math.log(10000.0) / self.d_model))
        encoding = torch.zeros(self.max_len, self.d_model)

        if self.period is not None and self.period > 1:
            # Periodic: Reset position every 'period' steps
            periodic_position = position % self.period
            encoding[:, 0::2] = torch.sin(periodic_position * div_term)
            encoding[:, 1::2] = torch.cos(periodic_position * div_term)
        else:
            # Absolute: Standard
            encoding[:, 0::2] = torch.sin(position * div_term)
            encoding[:, 1::2] = torch.cos(position * div_term)

        return encoding.unsqueeze(0)

# Helper to analyze RoPE wavelengths
def get_rope_best_match(d_model, base_freq, target_period):
    """
    Finds the RoPE dimension that has a rotational wavelength closest to the target_period.
    Returns the wave (cosine) for that dimension.
    """
    # 1. Calculate frequencies for all dimension pairs
    # theta_j = base^(-2j/d)
    j = torch.arange(0, d_model, 2).float()
    inv_freq = 1.0 / (base_freq ** (j / d_model))
    
    # 2. Convert to wavelengths (2*pi / freq)
    wavelengths = 2 * np.pi / inv_freq
    
    # 3. Find index closest to target_period
    diff = torch.abs(wavelengths - target_period)
    best_idx = torch.argmin(diff).item()
    best_wavelength = wavelengths[best_idx].item()
    best_freq = inv_freq[best_idx].item()
    
    return best_freq, best_wavelength, best_idx

# ==========================================
# 2. VISUALIZATION FUNCTION (What Hanne asked for)
# ==========================================

def visualize_alignment(data_series, d_model=512, save_path='results'):
    """
    Overlays PE patterns on top of the actual data to check for frequency alignment.
    """
    seq_len = min(len(data_series), 200) # Look at first 200 steps
    data = data_series[:seq_len]
    t = np.arange(seq_len)
    
    # Normalize data to [-1, 1] for easy visual comparison with Sine/Cosine PEs
    data_norm = (data - np.mean(data)) / (np.std(data) + 1e-5)
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    # --- PLOT 1: SineSPE Variants ---
    ax = axes[0]
    ax.plot(t, data_norm, 'k--', linewidth=2, alpha=0.6, label='Actual Data (Normalized)')
    
    # A. Aligned (Period=24)
    spe_24 = SineSPE(d_model, max_len=seq_len, period=24).sine[0, :, 0].numpy()
    ax.plot(t, spe_24, 'g-', linewidth=2, label='SineSPE (Period=24) [Aligned]')
    
    # B. Misaligned (Period=36)
    spe_36 = SineSPE(d_model, max_len=seq_len, period=36).sine[0, :, 0].numpy()
    ax.plot(t, spe_36, 'r-', linewidth=1.5, alpha=0.7, label='SineSPE (Period=36) [Misaligned]')
    
    # C. Absolute (No Period)
    spe_none = SineSPE(d_model, max_len=seq_len, period=None).sine[0, :, 0].numpy()
    ax.plot(t, spe_none, 'b:', linewidth=1.5, alpha=0.7, label='SineSPE (Absolute/Default)')
    
    ax.set_title("Check 1: SineSPE Periodicity vs. Data Seasonality")
    ax.set_ylabel("Amplitude (Normalized)")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # --- PLOT 2: RoPE Frequency Matching ---
    ax = axes[1]
    ax.plot(t, data_norm, 'k--', linewidth=2, alpha=0.6, label='Actual Data (Normalized)')
    
    # A. Tuned RoPE (Base 500)
    freq_500, wl_500, idx_500 = get_rope_best_match(d_model, base_freq=500, target_period=24)
    rope_wave_500 = np.cos(t * freq_500) # Simulate the rotation of this specific dimension
    ax.plot(t, rope_wave_500, 'g-', linewidth=2, label=f'RoPE Base=500 (Dim {idx_500}, WL={wl_500:.1f}) [Matched]')
    
    # B. Default RoPE (Base 10000)
    freq_10k, wl_10k, idx_10k = get_rope_best_match(d_model, base_freq=10000, target_period=24)
    rope_wave_10k = np.cos(t * freq_10k)
    ax.plot(t, rope_wave_10k, 'r-', linewidth=1.5, alpha=0.7, label=f'RoPE Base=10000 (Dim {idx_10k}, WL={wl_10k:.1f}) [Gap]')
    
    ax.set_title("Check 2: RoPE Rotational Frequency vs. Data Seasonality")
    ax.set_xlabel("Time Steps (Hours)")
    ax.set_ylabel("Amplitude (Normalized)")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, 'sanity_check_overlay.png')
    plt.savefig(save_file)
    print(f"\n[Success] Sanity check visualization saved to: {save_file}")
    plt.close()

# ==========================================
# 3. MAIN SCRIPT
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='ETTh1')
    parser.add_argument('--target_column', type=str, default='OT', help='Column to use as ground truth')
    args = parser.parse_args()

    DATA_FILEPATH = f'dataset/forecasting/{args.dataset_name}.csv'
    RESULTS_DIR = 'results/sanity_checks'
    
    print(f"--- Running Positional Encoding Sanity Check for {args.dataset_name} ---")

    try:
        if not os.path.exists(DATA_FILEPATH):
            print(f"Warning: File {DATA_FILEPATH} not found. Generating synthetic sine wave for demo.")
            t = np.arange(300)
            ts = np.sin(2 * np.pi * t / 24) + np.random.normal(0, 0.1, 300)
        else:
            # 1. Load Data
            df = pd.read_csv(DATA_FILEPATH)
            
            # --- FIX: Handle Long-Format Data (Pivot if needed) ---
            if 'cols' in df.columns and 'data' in df.columns:
                print("Detected long-format data. Pivoting...")
                df = df.pivot(index='date', columns='cols', values='data').reset_index()
            
            # Drop date column if it exists to keep only numeric data
            if 'date' in df.columns: 
                df = df.drop(columns=['date'])
            
            print(f"Available columns: {df.columns.tolist()}")

            # 2. Extract Target Series
            if args.target_column not in df.columns:
                # Fallback: Use the last column if 'OT' isn't found
                fallback_col = df.columns[-1]
                print(f"Column '{args.target_column}' not found. Using '{fallback_col}' instead.")
                ts = df[fallback_col].dropna().values.astype(float)
            else:
                ts = df[args.target_column].dropna().values.astype(float)

        # 3. Run Visualization
        # We limit to first 300 steps to make the plot readable
        ts_view = ts[:300]
        visualize_alignment(ts_view, d_model=512, save_path=RESULTS_DIR)
        
        print("\nInterpretation for Hanne:")
        print("1. SineSPE(24): Should align perfectly with the peaks/troughs of the data.")
        print("2. SineSPE(36): Peaks will slowly drift away from the data peaks (Misaligned).")
        print("3. RoPE(500): The green line should resonate with the data's frequency.")
        print("4. RoPE(10000): The red line might be too slow or fast (Gap).")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()