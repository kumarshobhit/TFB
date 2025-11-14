import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path
import argparse # Added for CLI argument parsing

def find_dominant_frequency(time_series, sample_rate):
    """
    Analyzes a time series to find its dominant frequency using the Fast Fourier Transform (FFT).

    Args:
        time_series (np.array): The input time series data (e.g., a column from a TFB dataset).
        sample_rate (float): The number of samples (data points) per unit of time (e.g., 24 for hourly data).
    
    Returns:
        float: The dominant frequency of the signal (cycles per unit of time, e.g., cycles/day).
        float: The dominant period (1 / dominant frequency, e.g., days).
        np.array: The positive frequencies analyzed.
        np.array: The power spectrum (magnitude squared).
    """
    N = len(time_series)
    
    # Detrending: Subtract the mean to remove the DC component (Frequency = 0),
    # which is often the largest power spike and can obscure genuine seasonality.
    detrended_series = time_series - np.mean(time_series)

    # 1. Apply the Fast Fourier Transform (FFT)
    fft_result = np.fft.fft(detrended_series)

    # 2. Calculate the corresponding frequencies
    frequencies = np.fft.fftfreq(N, d=1/sample_rate)

    # 3. Calculate the Power Spectral Density (PSD) / Power Spectrum
    # We only care about the positive frequencies, skipping the DC component (index 0).
    positive_frequencies = frequencies[1:N//2]
    power_spectrum = np.abs(fft_result[1:N//2])**2

    # 4. Find the dominant frequency
    peak_index = np.argmax(power_spectrum)
    dominant_frequency = positive_frequencies[peak_index]
    
    # The dominant period is 1 / dominant frequency
    dominant_period = 1 / dominant_frequency

    return dominant_frequency, dominant_period, positive_frequencies, power_spectrum

def main():
    parser = argparse.ArgumentParser(description="Analyze dominant frequency in a TFB-formatted time series CSV.")
    parser.add_argument(
        '--dataset_name', 
        type=str, 
        default='ETTh1',
        help='Name of the dataset file (e.g., "ETTh1"). Script assumes file is located at dataset/forecasting/forecasting/{name}.csv'
    )
    parser.add_argument(
        '--target_col',
        type=str,
        default='data',
        help='Name of the column containing the time series data (e.g., "data" or "OT")'
    )
    args = parser.parse_args()

    # --- Real Dataset Analysis Configuration ---
    # Construct the full file path automatically
    DATA_FILEPATH = f'dataset/forecasting/{args.dataset_name}.csv'
    DATASET_NAME = args.dataset_name
    TARGET_COLUMN = args.target_col
    HOURS_PER_DAY = 24
    SAMPLE_RATE = HOURS_PER_DAY # 24 observations per day (hourly data)
    RESULTS_DIR = 'results' 

    print(f"--- Time Series Frequency Analysis ---")
    print(f"Targeting dataset: {DATASET_NAME} (Path: {DATA_FILEPATH}), Column: '{TARGET_COLUMN}'")

    try:
        # --- Data Loading and Preprocessing ---
        df = pd.read_csv(DATA_FILEPATH)
        
        # Check if the primary target column exists, otherwise fallback to the first numeric column
        if TARGET_COLUMN not in df.columns:
            numeric_cols = df.select_dtypes(include=np.number).columns
            if not numeric_cols.empty:
                # Use the new target column for analysis and plotting
                TARGET_COLUMN = numeric_cols[0] 
                print(f"Warning: Column '{args.target_col}' not found. Analyzing first numeric column: '{TARGET_COLUMN}'")
            else:
                raise ValueError("No numeric columns found for analysis.")

        time_series_data = df[TARGET_COLUMN].values.astype(float)
        
        # Run the analysis
        dom_freq, dom_period, freqs, power = find_dominant_frequency(time_series_data, SAMPLE_RATE)

        # --- Display Results ---
        print(f"\n--- Analysis Results ---")
        print(f"Total data points: {len(time_series_data)}")
        print(f"Sampling Rate: {SAMPLE_RATE} (observations per day)")
        print(f"Calculated Dominant Frequency: {dom_freq:.6f} cycles/day")
        print(f"Calculated Dominant Period: {dom_period:.2f} days (The strongest seasonality)")

        # --- Plotting and Saving ---

        # Ensure the results directory exists
        os.makedirs(RESULTS_DIR, exist_ok=True) 

        # Prepare time index for the time series plot
        time_index = np.arange(len(time_series_data))

        plt.figure(figsize=(16, 6))

        # Plot 1: Time Series Data
        plt.subplot(1, 2, 1)
        plt.plot(time_index, time_series_data)
        plt.title(f'Time Series: {TARGET_COLUMN} from {DATASET_NAME}')
        plt.xlabel('Time (Hours/Samples)')
        plt.ylabel(TARGET_COLUMN)

        # Plot 2: Frequency Spectrum (Periodogram)
        plt.subplot(1, 2, 2)
        plt.plot(freqs, power, color='#1f77b4', linewidth=1)
        plt.title('Frequency Spectrum (Power vs. Frequency)')
        plt.xlabel('Frequency (Cycles per Day)')
        plt.ylabel('Power (Magnitude Squared)')

        # Highlight the dominant frequency peak
        plt.axvline(dom_freq, color='r', linestyle='--', 
                    label=f'Dominant Freq: {dom_freq:.4f} c/day\nPeriod: {dom_period:.2f} days')
        plt.legend()
        plt.tight_layout()
        
        # Construct the new output filename
        # Format: {dataset_name}_{target_column}_frequency_analysis.png
        output_filename = f'{DATASET_NAME}_{TARGET_COLUMN}_frequency_analysis.png'
        output_filepath = Path(RESULTS_DIR) / output_filename
        
        plt.savefig(output_filepath)
        print(f"\nSuccessfully saved plot to {output_filepath}")
        
        plt.show()

    except FileNotFoundError:
        print(f"Error: The file '{DATA_FILEPATH}' was not found.")
        print(f"Please ensure the dataset file '{DATASET_NAME}.csv' is located in the 'dataset' folder.")
    except ValueError as e:
        print(f"Error processing data: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
