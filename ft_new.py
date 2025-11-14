import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt # Import plotting library

def find_dominant_seasonal_period(time_series, sample_rate, max_period_to_ignore):
    """
    Calculates the dominant *seasonal* frequency and period by ignoring
    very low-frequency trend components.

    Args:
        time_series (np.array): The input time series data.
        sample_rate (float): Samples per unit of time (e.g., 24 for hourly data/day).
        max_period_to_ignore (float): The maximum period (in days) to filter out (e.g., 30 days).
    
    Returns:
        float: Dominant seasonal frequency (cycles/day).
        float: Dominant seasonal period (days).
        np.array: Frequencies for plotting.
        np.array: Power spectrum for plotting.
    """
    N = len(time_series)
    
    # 1. Detrend
    detrended_series = time_series - np.mean(time_series)

    # 2. Apply FFT
    fft_result = np.fft.fft(detrended_series)

    # 3. Get Frequencies and Power
    frequencies = np.fft.fftfreq(N, d=1/sample_rate)
    power_spectrum = np.abs(fft_result)**2

    # --- Filtering Logic ---
    # We only care about positive frequencies
    positive_mask = (frequencies > 0) & (frequencies <= sample_rate / 2)
    
    # Calculate the minimum frequency to consider (inverse of max_period_to_ignore)
    # This filters out long-term trends.
    min_freq_to_consider = 1 / max_period_to_ignore
    
    # Apply the frequency filter mask
    seasonal_mask = positive_mask & (frequencies >= min_freq_to_consider)
    
    # Get the filtered frequencies and power
    filtered_freqs = frequencies[seasonal_mask]
    filtered_power = power_spectrum[seasonal_mask]

    # --- Find Dominant Seasonal Peak ---
    if len(filtered_power) == 0:
        return 0, float('inf'), frequencies[positive_mask], power_spectrum[positive_mask]

    peak_index = np.argmax(filtered_power)
    dominant_frequency = filtered_freqs[peak_index]
    dominant_period = 1 / dominant_frequency

    return dominant_frequency, dominant_period, frequencies[positive_mask], power_spectrum[positive_mask]

# --- Configuration and Execution ---

# 1. File Path Settings
DATASET_NAME = 'ETTh1.csv'
TARGET_COLUMN = 'data'
DATA_FILEPATH = Path('dataset') / 'forecasting' / DATASET_NAME
RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(exist_ok=True) # Ensure results directory exists

# 2. Sampling Rate Configuration
SAMPLE_RATE = 24 # 24 observations per day

# 3. Analysis Filter
# We will ignore any "period" longer than 30 days to filter out the trend.
MAX_PERIOD_TO_IGNORE_DAYS = 30.0 

try:
    print(f"--- Loading data from: {DATA_FILEPATH} ---")
    df = pd.read_csv(DATA_FILEPATH)
    time_series_data = df[TARGET_COLUMN].values.astype(float)
    
    # Run the analysis
    dom_freq, dom_period, plot_freqs, plot_power = find_dominant_seasonal_period(
        time_series_data, 
        SAMPLE_RATE, 
        MAX_PERIOD_TO_IGNORE_DAYS
    )

    # --- Display Results ---
    print(f"\n--- Analysis Results for {DATASET_NAME} ('{TARGET_COLUMN}') ---")
    print(f"Total data points analyzed: {len(time_series_data)}")
    print(f"Filtering out periods > {MAX_PERIOD_TO_IGNORE_DAYS} days.")
    print(f"--------------------------------------------------")
    print(f"Calculated Dominant *Seasonal* Frequency: {dom_freq:.6f} cycles/day")
    print(f"Calculated Dominant *Seasonal* Period: {dom_period:.2f} days")

    # --- Plotting ---
    plt.figure(figsize=(12, 6))
    plt.plot(plot_freqs, plot_power, color='#1f77b4', linewidth=0.5)
    plt.title('Frequency Spectrum (Power vs. Frequency)')
    plt.xlabel('Frequency (Cycles per Day)')
    plt.ylabel('Power')
    
    # Highlight the dominant seasonal frequency peak
    plt.axvline(dom_freq, color='r', linestyle='--', 
                label=f'Dominant Seasonal Freq: {dom_freq:.4f} c/day\nPeriod: {dom_period:.2f} days')
    
    # Also show the 1-day peak for reference
    plt.axvline(1.0, color='gray', linestyle=':', label='1.0 cycles/day (Daily)')
    
    # Zoom in on the seasonal part of the spectrum (e.g., 0 to 2 cycles/day)
    plt.xlim(0, 2) 
    plt.legend()
    plt.tight_layout()
    
    output_filepath = RESULTS_DIR / f'{DATASET_NAME}_seasonal_spectrum.png'
    plt.savefig(output_filepath)
    print(f"\nSuccessfully saved plot to {output_filepath}")
    # plt.show() # Uncomment this if you want the plot to pop up

except FileNotFoundError:
    print(f"Error: The file '{DATA_FILEPATH}' was not found.")
except KeyError:
    print(f"Error: Column '{TARGET_COLUMN}' not found in the CSV file.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")