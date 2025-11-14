import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path
import argparse 

def find_dominant_frequency(time_series, sample_rate):
    """
    Analyzes a time series to find its dominant frequency using the Fast Fourier Transform (FFT).
    """
    print("\n--- Inside find_dominant_frequency function ---")
    N = len(time_series)
    print(f"[STEP 1] Starting with {N} data points.")
    
    # --- Operation: Detrending ---
    original_mean = np.mean(time_series)
    detrended_series = time_series - original_mean
    print(f"\n[STEP 2] Detrending: Subtracted the mean ({original_mean:.4f}) from the series.")
    print(f"         > Original data snippet: {time_series[:5]}")
    print(f"         > Detrended data snippet: {detrended_series[:5]}")

    # --- Operation: Apply FFT ---
    fft_result = np.fft.fft(detrended_series)
    print(f"\n[STEP 3] Applied FFT. The result is an array of 'complex numbers'.")
    print(f"         > Shape of FFT result: {fft_result.shape}")
    print(f"         > FFT result snippet: {fft_result[:5]}")

    # --- Operation: Get Frequencies (X-axis) ---
    frequencies = np.fft.fftfreq(N, d=1/sample_rate)
    print(f"\n[STEP 4] Calculated frequencies (the 'X-axis' for our plot).")
    print(f"         > Shape of frequencies array: {frequencies.shape}")
    print(f"         > Frequencies snippet: {frequencies[:5]}")

    # --- Operation: Get Power Spectrum (Y-axis) ---
    # We only care about the positive frequencies, skipping the DC component (index 0).
    positive_frequencies = frequencies[1:N//2]
    power_spectrum = np.abs(fft_result[1:N//2])**2
    print(f"\n[STEP 5] Calculated power spectrum (the 'Y-axis' for our plot).")
    print(f"         > We only look at positive frequencies. Shape: {power_spectrum.shape}")
    print(f"         > Power spectrum snippet: {power_spectrum[:5]}")

    # --- Operation: Find the Peak ---
    peak_index = np.argmax(power_spectrum)
    dominant_frequency = positive_frequencies[peak_index]
    dominant_period = 1 / dominant_frequency
    print(f"\n[STEP 6] Found the highest peak in the power spectrum.")
    print(f"         > The index of the highest peak is: {peak_index}")
    print(f"         > The frequency at that peak is: {dominant_frequency:.6f}")
    print(f"         > The period at that peak is: {dominant_period:.2f}")

    print("--- Exiting find_dominant_frequency function ---")
    return dominant_frequency, dominant_period, positive_frequencies, power_spectrum

def main():
    parser = argparse.ArgumentParser(description="Analyze dominant frequency in a TFB-formatted time series CSV.")
    parser.add_argument(
        '--dataset_name', 
        type=str, 
        default='ETTh1',
        help='Name of the dataset file (e.g., "ETTh1"). Script assumes file is located at dataset/forecasting/{name}.csv'
    )
    parser.add_argument(
        '--target_col',
        type=str,
        default='data',
        help='Name of the column containing the time series data (e.g., "data" or "OT")'
    )
    args = parser.parse_args()

    # --- Real Dataset Analysis Configuration ---
    DATA_FILEPATH = f'dataset/forecasting/{args.dataset_name}.csv'
    DATASET_NAME = args.dataset_name
    TARGET_COLUMN = args.target_col
    HOURS_PER_DAY = 24
    SAMPLE_RATE = HOURS_PER_DAY 
    RESULTS_DIR = 'results' 

    print(f"--- Time Series Frequency Analysis ---")
    print(f"Targeting dataset: {DATASET_NAME} (Path: {DATA_FILEPATH}), Column: '{TARGET_COLUMN}'")
    print(f"Assuming Sample Rate: {SAMPLE_RATE} samples per day (for hourly data)")

    try:
        # --- Data Loading and Preprocessing ---
        df = pd.read_csv(DATA_FILEPATH)
        print(f"\n[MAIN] Successfully loaded {DATA_FILEPATH}.")
        print("       > First 5 rows of the data:")
        print(df.head().to_string())
        
        if TARGET_COLUMN not in df.columns:
            # Fallback logic...
            numeric_cols = df.select_dtypes(include=np.number).columns
            if not numeric_cols.empty:
                TARGET_COLUMN = numeric_cols[0] 
                print(f"Warning: Column '{args.target_col}' not found. Analyzing first numeric column: '{TARGET_COLUMN}'")
            else:
                raise ValueError("No numeric columns found for analysis.")

        time_series_data = df[TARGET_COLUMN].values.astype(float)
        print(f"\n[MAIN] Extracted time series data from column '{TARGET_COLUMN}'.")
        print(f"       > Total data points (N): {len(time_series_data)}")
        
        # Run the analysis
        dom_freq, dom_period, freqs, power = find_dominant_frequency(time_series_data, SAMPLE_RATE)

        # --- Display Results ---
        print(f"\n--- Analysis Results ---")
        print(f"Calculated Dominant Frequency: {dom_freq:.6f} cycles/day")
        print(f"Calculated Dominant Period: {dom_period:.2f} days (The strongest seasonality)")

        # --- Plotting and Saving ---
        os.makedirs(RESULTS_DIR, exist_ok=True) 

        # This is the "index" for the time series plot. It's just a simple count.
        time_index = np.arange(len(time_series_data))
        print(f"\n[MAIN] Generating plot... The plot's time (X-axis) is a simple index from 0 to {len(time_index)-1}.")

        plt.figure(figsize=(16, 6))

        # Plot 1: Time Series Data (This is the visualization you asked for)
        plt.subplot(1, 2, 1)
        plt.plot(time_index, time_series_data)
        plt.title(f'Time Series: {TARGET_COLUMN} from {DATASET_NAME}')
        plt.xlabel('Time (Hours/Samples)') # The index IS the number of hours/samples
        plt.ylabel(TARGET_COLUMN)

        # Plot 2: Frequency Spectrum (The FFT result)
        plt.subplot(1, 2, 2)
        plt.plot(freqs, power, color='#1f77b4', linewidth=1)
        plt.title('Frequency Spectrum (Power vs. Frequency)')
        plt.xlabel('Frequency (Cycles per Day)')
        plt.ylabel('Power (Magnitude Squared)')

        plt.axvline(dom_freq, color='r', linestyle='--', 
                    label=f'Dominant Freq: {dom_freq:.4f} c/day\nPeriod: {dom_period:.2f} days')
        plt.legend()
        plt.tight_layout()
        
        output_filename = f'{DATASET_NAME}_{TARGET_COLUMN}_frequency_analysis.png'
        output_filepath = Path(RESULTS_DIR) / output_filename
        
        plt.savefig(output_filepath)
        print(f"\nSuccessfully saved plot to {output_filepath}")
        
        # plt.show() # You can uncomment this if you are running on your local machine

    except FileNotFoundError:
        print(f"Error: The file '{DATA_FILEPATH}' was not found.")
    except ValueError as e:
        print(f"Error processing data: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()