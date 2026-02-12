import pandas as pd
import numpy as np
import torch
from scipy.stats import entropy, wasserstein_distance
import seaborn as sns
import matplotlib.pyplot as plt


d_model = 128
n_heads = 8
head_dim = d_model // n_heads

def calculate_rope_frequencies(d_model, n_heads, base_freq):
    head_dim = d_model // n_heads
    theta = 1.0 / (base_freq ** (torch.arange(0, head_dim, 2).float() / head_dim))
    rope_freqs = theta.numpy() / (2 * np.pi)
    return rope_freqs

def calculate_kl_divergence(data_frequencies, rope_frequencies, num_bins=10):
    data_hist, bin_edges = np.histogram(data_frequencies, bins=num_bins, density=True)
    rope_hist, _ = np.histogram(rope_frequencies, bins=bin_edges, density=True)
    
    epsilon = 1e-9
    data_hist = data_hist + epsilon
    rope_hist = rope_hist + epsilon
    
    kl_div = entropy(data_hist, rope_hist)
    return kl_div


dataETTh1 = {
    'Channel': ['HUFL', 'HULL', 'LUFL', 'LULL', 'MUFL', 'MULL', 'OT'],
    'Peak 1 (Days)': [1, 1, 0.5, 1, 1, 1, 1],
    'Peak 2 (Days)': [0.5, 0.5, 1.01, 16.5, 0.5, 0.5, 20.74],
    'Peak 3 (Days)': [0.33, 16.5, 0.99, 21.99, 0.33, 26.88, 11.34]
}

data= {
    'Channel': ['% WEIGHTED ILI', '%UNWEIGHTED ILI', 'AGE 0-4', 'AGE 5-24', 'ILITOTAL', 'NUM. OF PROVIDERS', 'OT'],
    'Peak 1 (Days)': [375.67, 375.67, 375.67, 375.67, 375.67, 355.89, 355.89],
    'Peak 2 (Days)': [182.76, 182.76, 233.17, 233.17, 182.76, 182.76, 182.76],
    'Peak 3 (Days)': [270.48, 270.48, 182.76, 182.76, 233.17, 120.75, 120.75]
}


df = pd.DataFrame(data)
# df['Freq 1 (Hz)'] = 1 / (df['Peak 1 (Days)'] * 24)
# df['Freq 2 (Hz)'] = 1 / (df['Peak 2 (Days)'] * 24)
# df['Freq 3 (Hz)'] = 1 / (df['Peak 3 (Days)'] * 24)

df['Freq 1 (Hz)'] = 1 / (df['Peak 1 (Days)'] / 7)
df['Freq 2 (Hz)'] = 1 / (df['Peak 2 (Days)'] / 7)
df['Freq 3 (Hz)'] = 1 / (df['Peak 3 (Days)'] / 7)

frequency_series = pd.concat([df[col] for col in df.columns if 'Freq' in col])
frequency_series = frequency_series[frequency_series != np.inf]
frequency_series = frequency_series[frequency_series != 0]

min_data_freq = frequency_series.min()
max_data_freq = frequency_series.max()


print(f"Min Data Frequency: {min_data_freq:.4f}")
print(f"Max Data Frequency: {max_data_freq:.4f}")

base_freqs = np.arange(100, 10100, 100)
best_base_freq = None
min_range_diff = float('inf')

for base_freq in base_freqs:
    rope_freqs = calculate_rope_frequencies(d_model, n_heads, base_freq)
    min_rope_freq = rope_freqs.min()
    max_rope_freq = rope_freqs.max()
    
    range_diff = abs(max_data_freq - max_rope_freq) + abs(min_data_freq - min_rope_freq)
    if range_diff < min_range_diff:
        min_range_diff = range_diff
        best_base_freq = base_freq

print(f"Best Base Frequency using Range Matching: {best_base_freq}")

best_base_freq_kl = None
min_kl_divergence = float('inf')
best_rope_frequencies = None

for base_freq in base_freqs:
    rope_frequencies = calculate_rope_frequencies(d_model, n_heads, base_freq)
    kl_divergence = calculate_kl_divergence(frequency_series, rope_frequencies, num_bins=10)
    
    if kl_divergence < min_kl_divergence:
        min_kl_divergence = kl_divergence
        best_base_freq_kl = base_freq
        best_rope_frequencies = rope_frequencies
    
    print(f"Base Freq: {base_freq}, KL Divergence: {kl_divergence:.4f}")

print(f"\nBest Base Frequency using KL Divergence: {best_base_freq_kl}")
print(f"Minimum KL Divergence: {min_kl_divergence:.4f}")
# print(best_rope_frequencies)



# Plotting the histogram of the best rope frequencies
if best_rope_frequencies is not None:
    plt.figure(figsize=(8, 6))
    sns.histplot(best_rope_frequencies, kde=False)
    plt.title(f"Histogram of RoPE Frequencies (Base Freq: {best_base_freq_kl})")
    plt.xlabel("Frequency")
    plt.ylabel("Count")
    plt.show()
else:
    print("No valid RoPE frequencies to plot.")
