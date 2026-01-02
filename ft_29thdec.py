import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path
import argparse 
from scipy.signal import find_peaks

def find_top_frequencies(time_series, sample_rate, max_period_days=None, top_n=3):
    """
    Finds the top N dominant frequencies/periods, ignoring long-term trends.
    """
    N = len(time_series)
    detrended_series = time_series - np.mean(time_series)
    fft_result = np.fft.fft(detrended_series)
    frequencies = np.fft.fftfreq(N, d=1/sample_rate)
    
    # Positive half only
    positive_freqs = frequencies[1:N//2]
    power_spectrum = np.abs(fft_result[1:N//2])**2

    # --- FILTER: Ignore Long Trends ---
    if max_period_days:
        min_freq = 1.0 / max_period_days
        # Zero out power for low frequencies instead of removing arrays to keep indices aligned
        mask = positive_freqs < min_freq
        power_spectrum[mask] = 0

    # --- FIND PEAKS ---
    # We use find_peaks to get local maxima, preventing adjacent points of the same peak 
    # from counting as multiple "top" frequencies.
    # distance=5 prevents peaks closer than 5 bins from being selected.
    peaks, _ = find_peaks(power_spectrum, distance=5) 
    
    # Sort peaks by power (descending)
    sorted_peak_indices = peaks[np.argsort(power_spectrum[peaks])[::-1]]
    
    # Take top N
    top_indices = sorted_peak_indices[:top_n]
    
    results = []
    for idx in top_indices:
        freq = positive_freqs[idx]
        period = 1 / freq
        power = power_spectrum[idx]
        results.append((period, freq, power))
        
    return results, positive_freqs, power_spectrum

def plot_channel_analysis(time_series, freqs, power, top_results, 
                          dataset_name, col_name, results_dir, max_period):
    
    plt.figure(figsize=(12, 6))
    
    # Plot Frequency Spectrum only (Clean view)
    plt.plot(freqs, power, color='#1f77b4', linewidth=1, alpha=0.8)
    
    # Shade the ignored region
    if max_period:
        plt.axvspan(0, 1.0/max_period, color='gray', alpha=0.15, label=f'Ignored (> {max_period}d)')

    # Mark the Top Peaks
    colors = ['r', 'g', 'orange']
    for i, (period, freq, pwr) in enumerate(top_results):
        c = colors[i] if i < len(colors) else 'k'
        plt.plot(freq, pwr, "x", color=c, markersize=10)
        plt.annotate(f"{period:.2f}d", xy=(freq, pwr), xytext=(freq, pwr*1.1),
                     arrowprops=dict(facecolor=c, shrink=0.05), fontsize=9)

    plt.title(f'Spectrum: {col_name} (Top Periods Marked)')
    plt.xlabel('Frequency (Cycles per Day)')
    plt.ylabel('Power')
    plt.xlim(0, 4.0) # Zoom in to see Daily (1.0) and fractions. 
    plt.grid(True, alpha=0.3)
    
    safe_col_name = col_name.replace('/', '_')
    plt.savefig(Path(results_dir) / f'{dataset_name}_{safe_col_name}_spectrum.png')
    plt.close()

def get_sampling_rate(dataset_name):
    name = dataset_name.lower()
    if 'ettm' in name:
        return 96   # 15-minute intervals
    elif 'etth' in name or 'electricity' in name or 'traffic' in name:
        return 24   # Hourly
    elif 'weather' in name:
        return 144  # 10-minute intervals
    elif 'ili' in name:
        return 1/7  # Weekly
    elif 'exchange' in name:
        return 1    # Daily
    return 24       # Default

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='ETTh1')
    parser.add_argument('--max_period', type=float, default=30.0) 
    parser.add_argument('--limit_cols', type=int, default=5, help='Limit number of columns to analyze to avoid long runtimes on wide datasets')
    args = parser.parse_args()

    DATA_FILEPATH = f'dataset/forecasting/{args.dataset_name}.csv'
    RESULTS_DIR = 'results'
    
    print(f"--- Top-3 Periodicity Analysis (Max Period: {args.max_period}d) ---")

    sample_rate = get_sampling_rate(args.dataset_name)
    print(f"Detected Sampling Rate: {sample_rate} samples/day")

    try:
        print(f"Loading data from {DATA_FILEPATH}...")
        df = pd.read_csv(DATA_FILEPATH)
        print(f"Data loaded. Shape: {df.shape}")

        if 'cols' in df.columns and 'data' in df.columns:
            print("Pivoting long-format data...")
            df = df.pivot(index='date', columns='cols', values='data').reset_index()
            if 'date' in df.columns: df = df.drop(columns=['date'])

        numeric_cols = df.select_dtypes(include=np.number).columns
        
        if args.limit_cols > 0 and len(numeric_cols) > args.limit_cols:
            print(f"Dataset has {len(numeric_cols)} numeric columns. Limiting analysis to first {args.limit_cols}.")
            numeric_cols = numeric_cols[:args.limit_cols]
            
        os.makedirs(RESULTS_DIR, exist_ok=True) 
        
        summary_data = []

        for i, col in enumerate(numeric_cols):
            print(f"Processing column {i+1}/{len(numeric_cols)}: {col}")
            ts = df[col].dropna().values.astype(float)
            
            # Find Top 3
            top_3, freqs, power = find_top_frequencies(ts, sample_rate, max_period_days=args.max_period, top_n=3)
            
            # Prepare row for table
            row = {'Channel': col}
            for i, (period, freq, pwr) in enumerate(top_3):
                row[f'Peak {i+1} (Days)'] = f"{period:.2f}"
            
            summary_data.append(row)
            
            # Plot
            plot_channel_analysis(ts, freqs, power, top_3, args.dataset_name, col, RESULTS_DIR, args.max_period)

        # Print Table
        summary_df = pd.DataFrame(summary_data)
        print(f"\n## 📊 Top 3 Periodicities for {args.dataset_name}")
        print(summary_df.to_markdown(index=False))
        
        # 
        print("\nNote: Peak 1 is the strongest. If 'Peak 2' is around 7.00, that is your weekly seasonality.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()