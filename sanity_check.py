import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import math
import argparse
import os

# ==========================================
# 1. POSITIONAL ENCODING CLASSES
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
            periodic_position = position % self.period
            encoding[:, 0::2] = torch.sin(periodic_position * div_term)
            encoding[:, 1::2] = torch.cos(periodic_position * div_term)
        else:
            encoding[:, 0::2] = torch.sin(position * div_term)
            encoding[:, 1::2] = torch.cos(position * div_term)

        return encoding.unsqueeze(0)

def get_rope_best_match(d_model, base_freq, target_period):
    j = torch.arange(0, d_model, 2).float()
    inv_freq = 1.0 / (base_freq ** (j / d_model))
    wavelengths = 2 * np.pi / inv_freq
    diff = torch.abs(wavelengths - target_period)
    best_idx = torch.argmin(diff).item()
    best_wavelength = wavelengths[best_idx].item()
    best_freq = inv_freq[best_idx].item()
    return best_freq, best_wavelength, best_idx

# ==========================================
# 2. VISUALIZATION (Normalized Data, Overlapping)
# ==========================================

def visualize_alignment(data_series, d_model=512, save_path='results', channel_name='OT'):
    # View 5 Days (120 hours)
    seq_len = min(len(data_series), 120) 
    data = data_series[:seq_len]
    
    # --- NORMALIZE DATA (Z-Score) ---
    # This brings the data to roughly [-2, 2], comparable to PE's [-1, 1]
    data_mean = np.mean(data)
    data_std = np.std(data)
    data_norm = (data - data_mean) / (data_std + 1e-5)
    
    t = np.arange(seq_len)
    
    fig, axes = plt.subplots(5, 1, figsize=(15, 22), sharex=False) 
    
    def plot_dual_norm(ax, pe_signal, pe_label, pe_color, title):
        # Left Axis: Normalized Data
        color_data = 'black'
        # Plotting Normalized Data with fixed limits
        ax.plot(t, data_norm, color=color_data, linestyle='--', linewidth=2, alpha=0.4, label=f'Norm. Data ({channel_name})')
        ax.set_ylabel(f'Norm. Value (Z-score)', color=color_data)
        ax.tick_params(axis='y', labelcolor=color_data)
        ax.set_ylim(-3, 3) # Fix limits to keep data centered
        
        # Right Axis: PE Signal
        ax2 = ax.twinx()
        ax2.plot(t, pe_signal, color=pe_color, linewidth=2, alpha=0.9, label=pe_label)
        ax2.set_ylabel('PE Wave (-1 to 1)', color=pe_color)
        ax2.tick_params(axis='y', labelcolor=pe_color)
        ax2.set_ylim(-1.5, 1.5) # Fix limits so PE is comparable size to Data
        
        # X-Axis Ticks (Multiples of 24)
        xticks = np.arange(0, seq_len + 1, 24)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{x}h\n(Day {x//24})" for x in xticks])
        ax.set_xlim(0, seq_len)

        # Day Markers
        for day in xticks:
            ax.axvline(x=day, color='gray', linestyle=':', alpha=0.8, linewidth=1.5)

        ax.set_title(title, fontsize=11, fontweight='bold')
        
        lines_1, labels_1 = ax.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
        ax.grid(True, alpha=0.3)

    # 1. SineSPE Aligned (Period=24)
    spe_24 = SineSPE(d_model, max_len=seq_len, period=24).sine[0, :, 0].numpy()
    plot_dual_norm(axes[0], spe_24, 'SineSPE (Period=24)', 'green', 
              "1. SineSPE Aligned (Period=24): Green Peak overlaps with Data Peaks at Day Start")

    # 2. SineSPE Misaligned (Period=36)
    spe_36 = SineSPE(d_model, max_len=seq_len, period=36).sine[0, :, 0].numpy()
    plot_dual_norm(axes[1], spe_36, 'SineSPE (Period=36)', 'red', 
              "2. SineSPE Misaligned (Period=36): Red Peak drifts away from Data Peaks")

    # 3. SineSPE Default
    spe_none = SineSPE(d_model, max_len=seq_len, period=None).sine[0, :, 40].numpy() 
    plot_dual_norm(axes[2], spe_none, 'SineSPE Default (No Period)', 'blue', 
              "3. SineSPE Default: No correlation with Data Peaks")

    # 4. RoPE Tuned
    freq_500, wl_500, idx_500 = get_rope_best_match(d_model, base_freq=500, target_period=24)
    rope_wave_500 = np.cos(t * freq_500)
    plot_dual_norm(axes[3], rope_wave_500, f'RoPE Base=500 (WL={wl_500:.1f})', 'green', 
              "4. RoPE Tuned (Base=500): Overlaps with Data Seasonality")

    # 5. RoPE Default
    freq_10k, wl_10k, idx_10k = get_rope_best_match(d_model, base_freq=10000, target_period=24)
    rope_wave_10k = np.cos(t * freq_10k)
    plot_dual_norm(axes[4], rope_wave_10k, f'RoPE Base=10000 (WL={wl_10k:.1f})', 'red', 
              "5. RoPE Default (Base=10000): Slight Phase Shift visible over time")

    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, 'sanity_check_norm_overlap.png')
    plt.savefig(save_file)
    print(f"\n[Success] Normalized visualization saved to: {save_file}")
    plt.close()

# ==========================================
# MAIN
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='ETTh1')
    parser.add_argument('--target_column', type=str, default='OT')
    args = parser.parse_args()

    DATA_FILEPATH = f'dataset/forecasting/{args.dataset_name}.csv'
    RESULTS_DIR = 'results/sanity_checks'
    
    print(f"--- Running Normalized Overlap Check for {args.dataset_name} ---")

    try:
        if not os.path.exists(DATA_FILEPATH):
            # Fallback for demo if file missing
            print("Dataset not found, using synthetic...")
            t = np.arange(200)
            ts = 10 + 0.05*t + 5*np.sin(2 * np.pi * t / 24) + np.random.normal(0, 1, 200)
        else:
            df = pd.read_csv(DATA_FILEPATH)
            # Pivot if needed
            if 'cols' in df.columns and 'data' in df.columns:
                print("Pivoting long-format data...")
                df = df.pivot(index='date', columns='cols', values='data').reset_index()
            if 'date' in df.columns: df = df.drop(columns=['date'])

            if args.target_column not in df.columns:
                print(f"Column {args.target_column} not found. Using last column.")
                ts = df[df.columns[-1]].dropna().values.astype(float)
                args.target_column = df.columns[-1]
            else:
                ts = df[args.target_column].dropna().values.astype(float)

        visualize_alignment(ts, d_model=512, save_path=RESULTS_DIR, channel_name=args.target_column)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()