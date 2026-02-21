import numpy as np
import pandas as pd
import argparse 
from scipy.signal import find_peaks

def find_top_raw_frequencies(time_series, top_n=5):
    """
    Performs FFT on raw data without any time-unit assumptions.
    Returns: Frequency (Cycles per Step), Power
    """
    N = len(time_series)
    
    # 1. Detrend (Center data around 0 to remove the DC component/mean)
    detrended_series = time_series - np.mean(time_series)
    
    # 2. FFT
    fft_result = np.fft.fft(detrended_series)
    
    # 3. Get Frequencies 
    # d=1.0 means "1 step". The result is in cycles/step.
    frequencies = np.fft.fftfreq(N, d=1.0)
    
    # 4. Keep Positive Half (0 to 0.5)
    positive_indices = np.where(frequencies > 0)[0]
    positive_freqs = frequencies[positive_indices]
    power_spectrum = np.abs(fft_result[positive_indices])**2

    # 5. Find Peaks (Distance=5 prevents picking neighbor points of same peak)
    peaks, _ = find_peaks(power_spectrum, distance=5)
    
    # 6. Sort by Power (Descending)
    if len(peaks) > 0:
        sorted_peak_indices = peaks[np.argsort(power_spectrum[peaks])[::-1]]
    else:
        # Fallback if no peaks are found, just take the largest magnitude bins
        sorted_peak_indices = np.argsort(power_spectrum)[::-1]
    
    # 7. Extract Top N
    top_indices = sorted_peak_indices[:top_n]
    
    results = []
    for idx in top_indices:
        freq = positive_freqs[idx]
        pwr = power_spectrum[idx]
        results.append((freq, pwr))
        
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='ETTh1')
    args = parser.parse_args()

    # Paths
    DATA_FILEPATH = f'dataset/forecasting/{args.dataset_name}.csv'
    
    print(f"--- Raw Frequency Analysis (No Time Units) ---")
    print(f"Dataset: {args.dataset_name}")

    try:
        # Load Data
        df = pd.read_csv(DATA_FILEPATH)
        
        # Handle Long Format (if necessary)
        if 'cols' in df.columns and 'data' in df.columns: 
            print("Pivoting long-format data...")
            df = df.pivot(index='date', columns='cols', values='data').reset_index()
            if 'date' in df.columns: df = df.drop(columns=['date'])

        # Select Numeric Columns
        numeric_cols = df.select_dtypes(include=np.number).columns
        print(f"Analyzing {len(numeric_cols)} columns...")
        
        summary_data = []

        for col in numeric_cols:
            # Clean NaNs
            ts = df[col].dropna().values.astype(float)
            
            # Get Top 5 Frequencies
            top_5 = find_top_raw_frequencies(ts, top_n=5)
            
            # Build Table Row
            row = {'Channel': col}
            for i in range(5):
                # Handle cases where fewer than 5 peaks might be found
                if i < len(top_5):
                    freq, pwr = top_5[i]
                    row[f'Freq {i+1} (Cycles/Step)'] = f"{freq:.5f}"
                else:
                    row[f'Freq {i+1} (Cycles/Step)'] = "N/A"
            
            summary_data.append(row)

        # Output Table
        summary_df = pd.DataFrame(summary_data)
        print(f"\n## 📊 Top 5 Raw Frequencies for {args.dataset_name}")
        # to_markdown requires the 'tabulate' library installed
        try:
            print(summary_df.to_markdown(index=False))
        except ImportError:
            print(summary_df.to_string(index=False))

    except FileNotFoundError:
        print(f"Error: File not found at {DATA_FILEPATH}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()