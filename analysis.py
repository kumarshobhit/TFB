import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.spatial.distance import cosine
from scipy import signal

# --- Data Input (Copy from your previous logs to save time) ---
features = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']

# 1. Alignment Scores (Absolute Values)
# Copied from your "Alignment Scores" table
cosine_aligned = np.array([0.9288, 0.9383, 0.8222, 0.7822, 0.9207, 0.9411, 0.6297]) # Order: HUFL...OT
cosine_default = np.array([0.4395, 0.5715, 0.7286, 0.7735, 0.4245, 0.5681, 0.8573])

# 2. MSE Errors (Absolute Values)
# Copied from your "Alignment & Error Analysis" table
mse_aligned = np.array([25.9746, 0.9383, 24.2273, 0.6583, 0.5677, 0.0557, 5.1740])
mse_default = np.array([26.1896, 0.9657, 24.1290, 0.6829, 0.5853, 0.0560, 4.6955])

# --- Normalization Note ---
# Absolute MSE is hard to compare because HUFL is ~25 and MULL is ~0.05. 
# We should log-transform MSE or normalize to visualize it on one plot, 
# BUT Spearman rank doesn't care about magnitude, only order. 
# We will use raw MSE for the calc, but maybe log-scale for the plot.

# --- Pooling the Data (N = 14) ---
# X = All Alignment Scores
X_all = np.concatenate([cosine_aligned, cosine_default])
# Y = All MSE Scores
Y_all = np.concatenate([mse_aligned, mse_default])
# Labels for plotting
labels = features * 2
colors = ['blue']*7 + ['red']*7 # Blue = Aligned, Red = Default

# --- Analysis ---
corr, p_val = spearmanr(X_all, Y_all)

print(f"=== Pooled Analysis (N={len(X_all)}) ===")
print(f"Spearman Correlation: {corr:.4f}")
print(f"P-Value: {p_val:.4f}")

# --- Visualization ---
plt.figure(figsize=(10, 6))

# Plot Default points (Red)
plt.scatter(cosine_default, mse_default, color='red', label='Default PE', alpha=0.6, s=100)
# Plot Aligned points (Blue)
plt.scatter(cosine_aligned, mse_aligned, color='blue', label='Aligned PE', alpha=0.6, s=100)

# Connect the pairs to show the shift
for i in range(len(features)):
    plt.plot([cosine_default[i], cosine_aligned[i]], 
             [mse_default[i], mse_aligned[i]], 'k--', alpha=0.2)
    plt.text(cosine_aligned[i], mse_aligned[i], features[i], fontsize=9)

plt.title(f'Absolute Spectral Alignment vs. Absolute Error (N=14)\nSpearman Corr: {corr:.2f} (p={p_val:.3f})')
plt.xlabel('Spectral Alignment (Cosine Similarity)')
plt.ylabel('Test MSE (Log Scale)')
plt.yscale('log') # Log scale handles the huge difference between HUFL (25) and MULL (0.05)
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.4)

plt.tight_layout()
plt.savefig('pooled_correlation_plot.png')
print("Saved plot to pooled_correlation_plot.png")