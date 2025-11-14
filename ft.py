import numpy as np
import matplotlib.pyplot as plt

def find_dominant_frequency(time_series, sample_rate):
    """
    Analyzes a time series to find its dominant frequency using the Fast Fourier Transform (FFT).

    Args:
        time_series (np.array): The input time series data (e.g., a column from a TFB dataset).
        sample_rate (float): The number of samples (data points) per unit of time (e.g., 1 for daily data, 24 for hourly).
    
    Returns:
        float: The dominant frequency of the signal.
        float: The dominant period (1 / dominant frequency).
    """
    N = len(time_series)

    # 1. Apply the Fast Fourier Transform (FFT)
    # The FFT converts the time-domain signal into the frequency domain.
    fft_result = np.fft.fft(time_series)

    # 2. Calculate the corresponding frequencies
    # fftfreq computes the frequencies for each point in the FFT result.
    frequencies = np.fft.fftfreq(N, d=1/sample_rate)

    # 3. Calculate the Power Spectral Density (PSD) / Power Spectrum
    # We use the magnitude (absolute value) squared of the complex FFT output.
    # We only care about the positive frequencies, as the negative ones are symmetrical.
    # The first index (0) is the DC component (mean of the signal), which we skip.
    positive_frequencies = frequencies[1:N//2]
    power_spectrum = np.abs(fft_result[1:N//2])**2

    # 4. Find the dominant frequency
    # The dominant frequency is the one corresponding to the maximum power.
    peak_index = np.argmax(power_spectrum)
    dominant_frequency = positive_frequencies[peak_index]
    
    # The dominant period is 1 / dominant frequency
    # If the sample_rate is 1 (daily data), a dominant period of 7 means weekly seasonality.
    dominant_period = 1 / dominant_frequency

    return dominant_frequency, dominant_period, positive_frequencies, power_spectrum

# --- Example Usage ---
# Assume we have DAILY data, so sample_rate = 1
SAMPLE_RATE = 1.0 # 1 observation per day
DAYS = 365 * 2 # 2 years of daily data
t = np.linspace(0, DAYS, DAYS, endpoint=False)

# Create a synthetic time series (your actual data would replace this)
# This signal has two main frequencies:
# 1. A weekly cycle (Frequency = 1/7 cycles per day)
# 2. A yearly cycle (Frequency = 1/365 cycles per day)
weekly_freq = 1 / 7
yearly_freq = 1 / 365
noise_level = 0.5

# Generate the time series data
synthetic_data = (
    5 * np.sin(2 * np.pi * weekly_freq * t) +       # Strong weekly seasonality
    2 * np.sin(2 * np.pi * yearly_freq * t) +       # Weaker yearly seasonality
    np.random.randn(DAYS) * noise_level             # Added random noise
)

# Run the analysis
dom_freq, dom_period, freqs, power = find_dominant_frequency(synthetic_data, SAMPLE_RATE)

# --- Display Results and Plotting ---

print(f"--- Analysis Results ---")
print(f"Total data points: {DAYS} days")
print(f"Sampling Rate: {SAMPLE_RATE} (1 per day)")
print(f"Calculated Dominant Frequency: {dom_freq:.4f} cycles/day")
print(f"Calculated Dominant Period: {dom_period:.2f} days (The strongest seasonality)")
print(f"Expected Dominant Period: 7 days")

# Plot the original time series
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(t, synthetic_data)
plt.title('Time Series Data (2 Years)')
plt.xlabel('Time (Days)')
plt.ylabel('Value')

# Plot the Frequency Spectrum
plt.subplot(1, 2, 2)
plt.plot(freqs, power)
plt.title('Frequency Spectrum (Power vs. Frequency)')
plt.xlabel('Frequency (Cycles per Day)')
plt.ylabel('Power (Magnitude Squared)')

# Highlight the dominant frequency peak
plt.axvline(dom_freq, color='r', linestyle='--', label=f'Dominant Freq: {dom_freq:.4f}')
plt.legend()
plt.tight_layout()
plt.show()

# You can easily adapt this by loading your TFB data instead of generating synthetic_data.
# Example for loading data (replace with your actual loading logic):
# try:
#     data = np.loadtxt('./dataset/ETTh1.csv', delimiter=',', skiprows=1, usecols=1)
#     dom_freq, dom_period, _, _ = find_dominant_frequency(data, SAMPLE_RATE)
#     print(f"Dominant Period for ETTh1: {dom_period:.2f} days")
# except FileNotFoundError:
#     print("Could not find external TFB data file.")
