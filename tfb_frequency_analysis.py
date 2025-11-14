#!/usr/bin/env python3
"""
TFB Frequency Analysis
----------------------
Analyze frequency characteristics of datasets stored in **TFB format** (three-column long table):
    - 'date' : timestamp or integer index (1..N)
    - 'data' : measurement values (float)
    - 'cols' : variable/series name (string)

What it does per series ('cols'):
    1) Validates and sorts the time index; infers sampling interval & regularity.
    2) Optionally resamples to a regular grid (if timestamps are used and irregular).
    3) Detrends (optional) and de-means; fills small gaps via forward-fill/backfill.
    4) Computes:
        - Nyquist frequency, sampling rate (Hz-like, i.e., cycles per unit time) and unit
        - Dominant frequency peaks via periodogram + Welch
        - Suggested seasonal period(s) (from top peaks + ACF)
        - Suggested **Patch Lengths** for Patch-based Transformers
    5) Writes a CSV summary and (optionally) plots spectra for each series.

CLI
---
python tfb_frequency_analysis.py \
    --input path/to/data.csv \
    --outdir ./freq_report \
    --time_unit auto \
    --detrend median \
    --max-peaks 5 \
    --plots

You can also pass a directory to --input; we'll analyze all *.csv files inside it.

Dependencies: pandas, numpy, scipy, matplotlib
Tested with Python 3.8+
"""
import argparse
import os
import sys
from pathlib import Path
from typing import Tuple, Dict, Optional, List

import numpy as np
import pandas as pd

from scipy import signal
from scipy.signal import detrend as scipy_detrend
from statsmodels.tsa.stattools import acf as sm_acf

# Matplotlib is imported only if plots are requested to avoid headless issues.
plt = None

# ---------- Utilities ----------

def _maybe_import_matplotlib():
    global plt
    if plt is None:
        import matplotlib.pyplot as plt  # noqa: F401
        return plt
    return plt

def infer_time_unit_and_rate(idx: pd.Series, user_unit: str = "auto") -> Tuple[str, float, np.ndarray]:
    """
    Infer the most plausible time unit and sampling rate (samples per unit), given a datetime or integer index.
    Returns (unit_str, samples_per_unit, diffs_in_seconds_or_units_array).
    """
    if np.issubdtype(idx.dtype, np.integer):
        # Integer index: assume unit=step and spacing=1 unless otherwise detected
        diffs = np.diff(idx.values.astype(np.int64))
        diffs = diffs[diffs > 0]
        if len(diffs) == 0:
            raise ValueError("Index has insufficient or non-increasing values.")
        step = int(pd.Series(diffs).mode().iloc[0])
        samples_per_unit = 1.0 / step  # 'unit' is one integer step
        return ("step", samples_per_unit, diffs.astype(float))
    else:
        # Datetime-like: compute deltas in seconds
        dt = pd.to_datetime(idx, errors="coerce")
        if dt.isna().any():
            raise ValueError("Failed to parse some 'date' values to datetime.")
        diffs = dt.diff().dropna().dt.total_seconds().values
        if len(diffs) == 0:
            raise ValueError("Not enough timestamps to infer sampling interval.")
        # Use mode-like robust estimator for step
        spacing = pd.Series(diffs).round().mode().iloc[0]
        # Decide a unit
        if user_unit != "auto":
            unit = user_unit
        else:
            # Choose the coarsest unit that yields an integer-ish spacing
            if np.isclose(spacing, 1, atol=0.1):
                unit = "second"
            elif np.isclose(spacing, 60, atol=1):
                unit = "minute"
            elif np.isclose(spacing, 3600, atol=10):
                unit = "hour"
            elif 3600*20 < spacing < 3600*28:  # rough daily-ish mode
                unit = "day"
            else:
                # Default: seconds
                unit = "second"
        # samples per unit (e.g., samples/day)
        seconds_per_unit = {"second": 1, "minute": 60, "hour": 3600, "day": 86400}[unit]
        samples_per_unit = seconds_per_unit / spacing
        return (unit, float(samples_per_unit), diffs)

def regularize_series(df_one: pd.DataFrame, time_unit: str, allow_resample: bool = True) -> Tuple[pd.Series, float, str]:
    """
    Ensure a regular time grid. Returns (values_series, samples_per_unit, unit)
    The input df_one must have columns ['date','data'] with a single 'cols' value.
    """
    # Sort by date
    df_one = df_one.sort_values("date")
    unit, samples_per_unit, diffs = None, None, None

    # Parse index
    if np.issubdtype(df_one["date"].dtype, np.number) and not np.issubdtype(df_one["date"].dtype, np.datetime64):
        # Integer index (TFB allows this)
        df_one = df_one.set_index("date")
        unit, samples_per_unit, diffs = infer_time_unit_and_rate(df_one.index.to_series(), "step" if time_unit == "auto" else time_unit)
        # Ensure consecutive steps; if not, reindex and fill
        full_index = pd.RangeIndex(start=int(df_one.index.min()), stop=int(df_one.index.max()) + 1, step=1)
        df_one = df_one.reindex(full_index)
        df_one["data"] = df_one["data"].interpolate().ffill().bfill()
        return df_one["data"], samples_per_unit, unit
    else:
        # Datetime-like
        df_one["date"] = pd.to_datetime(df_one["date"], errors="coerce")
        if df_one["date"].isna().any():
            raise ValueError("Some 'date' values are invalid datetimes.")
        df_one = df_one.set_index("date")
        unit, samples_per_unit, diffs = infer_time_unit_and_rate(df_one.index.to_series(), time_unit)
        seconds_per_unit = {"second": 1, "minute": 60, "hour": 3600, "day": 86400}[unit]
        nominal_step_seconds = seconds_per_unit / samples_per_unit
        # If irregular and allowed, resample
        if allow_resample:
            rule = {"second": "S", "minute": "T", "hour": "H", "day": "D"}[unit]
            # Round nominal step to nearest whole unit
            if unit in ("second", "minute"):
                # keep as is
                pass
            # Resample to regular grid
            s = df_one["data"].asfreq(rule)
            if s.isna().mean() > 0.5:
                s = df_one["data"].resample(rule).mean()
            s = s.interpolate(limit_direction="both").ffill().bfill()
            return s, samples_per_unit, unit
        else:
            return df_one["data"], samples_per_unit, unit

def remove_trend(values: pd.Series, method: str = "median") -> pd.Series:
    x = values.astype(float).copy()
    if method is None or method == "none":
        return x - x.mean()
    method = method.lower()
    if method == "linear":
        return pd.Series(scipy_detrend(x.values), index=x.index)
    elif method == "median":
        med = x.rolling(window=max(5, int(len(x)*0.05)), min_periods=1, center=True).median()
        return x - med
    elif method == "ema":
        trend = x.ewm(span=max(5, int(len(x)*0.05)), adjust=False).mean()
        return x - trend
    else:
        return x - x.mean()

def spectral_peaks(x: np.ndarray, fs: float, max_peaks: int = 5) -> List[Tuple[float, float]]:
    """
    Return list of (freq, power) sorted by descending power.
    freq is in cycles per unit time (based on fs units).
    """
    # Periodogram
    freqs, pxx = signal.periodogram(x, fs=fs, scaling="spectrum", detrend="constant", window="hann")
    # Welch (smoother)
    wf, wpxx = signal.welch(x, fs=fs, nperseg=min(256, max(32, len(x)//8)), scaling="spectrum")
    # Combine by normalizing and averaging
    pxxn = (pxx - pxx.min()) / (pxx.max() - pxx.min() + 1e-12)
    wpxxn = (wpxx - wpxx.min()) / (wpxx.max() - wpxx.min() + 1e-12)
    cf = np.linspace(0, 1, len(freqs))
    common = np.interp(cf, np.linspace(0, 1, len(wf)), wpxxn)
    score = 0.5 * pxxn + 0.5 * common
    # Find peaks excluding zero frequency
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(score, distance=max(2, len(score)//100))
    peaks = [p for p in peaks if freqs[p] > 0]
    tops = sorted(peaks, key=lambda i: score[i], reverse=True)[:max_peaks]
    return [(float(freqs[i]), float(score[i])) for i in tops]

def acf_periods(x: np.ndarray, fs: float, top_k: int = 3) -> List[float]:
    """
    Suggest candidate periods from ACF peaks (in samples), converted to 'time units' using fs.
    Returns a list of candidate periods in **time units**.
    """
    nlags = min(2000, max(100, len(x)//2))
    acv = sm_acf(x, nlags=nlags, fft=True)
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(acv[1:])  # skip lag 0
    if len(peaks) == 0:
        return []
    # Convert peaks (in lags) to periods (in time units): lag / fs
    periods = [(p+1)/fs for p in peaks]  # +1 because we removed lag 0
    # Take the strongest (by acf value)
    idx = np.argsort(acv[1:][peaks])[::-1][:top_k]
    return [float(periods[i]) for i in idx]

def suggest_patch_lengths(dominant_freqs: List[float], unit: str, fs: float) -> List[int]:
    """
    Convert top frequencies to patch lengths in samples (rounded), plus simple multiples/halves.
    We suggest:
      - period_in_samples = round(fs / f)
      - {round(period), 2*period, period//2} where valid
    """
    suggestions = set()
    for f in dominant_freqs:
        if f <= 0:
            continue
        period_samples = int(round(fs / f))
        if period_samples >= 2:
            suggestions.add(period_samples)
            suggestions.add(period_samples * 2)
            suggestions.add(max(2, period_samples // 2))
    return sorted(suggestions)

def analyze_one_series(df: pd.DataFrame, series_name: str, time_unit: str, detrend_method: str, max_peaks: int) -> Dict:
    s, fs, unit = regularize_series(df[df["cols"] == series_name][["date", "data", "cols"]].drop(columns=["cols"]), time_unit, allow_resample=True)
    x = remove_trend(s, detrend_method).astype(float).values
    x = x - np.nanmean(x)
    x = np.nan_to_num(x)
    if np.allclose(x, 0):
        return {
            "series": series_name, "unit": unit, "fs(samples_per_unit)": fs,
            "nyquist(cycles_per_unit)": fs/2.0, "dominant_freqs(cycles_per_unit)": [],
            "dominant_periods(time_units)": [], "acf_periods(time_units)": [],
            "suggested_patch_lengths(samples)": [], "n_samples": len(s)
        }
    peaks = spectral_peaks(x, fs=fs, max_peaks=max_peaks)
    dom_freqs = [f for (f,score) in peaks]
    dom_periods = [1.0/f for f in dom_freqs if f>0]
    acf_p = acf_periods(x, fs=fs, top_k=min(3, max_peaks))
    patch_suggestions = suggest_patch_lengths(dom_freqs, unit=unit, fs=fs)
    return {
        "series": series_name,
        "unit": unit,
        "fs(samples_per_unit)": fs,
        "nyquist(cycles_per_unit)": fs/2.0,
        "dominant_freqs(cycles_per_unit)": dom_freqs,
        "dominant_periods(time_units)": dom_periods,
        "acf_periods(time_units)": acf_p,
        "suggested_patch_lengths(samples)": patch_suggestions,
        "n_samples": len(s)
    }

def plot_spectrum(x: np.ndarray, fs: float, outpath: Path, title: str):
    _maybe_import_matplotlib()
    import matplotlib.pyplot as plt
    freqs, pxx = signal.periodogram(x, fs=fs, scaling="spectrum", detrend="constant", window="hann")
    plt.figure(figsize=(8,4.5))
    plt.plot(freqs, pxx)
    plt.xlabel("Frequency (cycles per unit time)")
    plt.ylabel("Power")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def analyze_file(csv_path: Path, outdir: Path, time_unit: str = "auto", detrend_method: str = "median", max_peaks: int = 5, make_plots: bool = False) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"date","data","cols"}
    if not required.issubset(df.columns):
        raise ValueError(f"{csv_path} is not in TFB long format. Expected columns {required}, got {set(df.columns)}")
    # Ensure types
    # 'data'
    df["data"] = pd.to_numeric(df["data"], errors="coerce")
    df = df.dropna(subset=["data"])
    # For each series
    series_names = sorted(df["cols"].unique().tolist())
    results = []
    fig_dir = outdir / "figs" / csv_path.stem
    fig_dir.mkdir(parents=True, exist_ok=True)
    for name in series_names:
        res = analyze_one_series(df, name, time_unit, detrend_method, max_peaks)
        results.append(res)
        if make_plots and len(df[df["cols"]==name])>8:
            # Build spectrum on detrended data again for plotting
            s, fs, unit = regularize_series(df[df["cols"] == name][["date", "data", "cols"]].drop(columns=["cols"]), time_unit, allow_resample=True)
            x = remove_trend(s, detrend_method).astype(float).values
            x = x - np.nanmean(x)
            x = np.nan_to_num(x)
            if not np.allclose(x, 0):
                plot_spectrum(x, fs, fig_dir / f"{name}_spectrum.png", f"{csv_path.name} :: {name}")
    # Save summary
    summary = pd.DataFrame(results)
    summary_path = outdir / f"{csv_path.stem}_frequency_summary.csv"
    summary.to_csv(summary_path, index=False)
    return summary

def analyze_input(input_path: str, outdir: str, time_unit: str = "auto", detrend_method: str = "median", max_peaks: int = 5, make_plots: bool = False) -> None:
    input_path = Path(input_path)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    all_summaries = []
    if input_path.is_dir():
        csvs = sorted(list(input_path.glob("*.csv")))
        if not csvs:
            print(f"No CSV files found in {input_path}")
            return
        for p in csvs:
            print(f"[INFO] Analyzing {p.name} ...")
            summary = analyze_file(p, outdir, time_unit, detrend_method, max_peaks, make_plots)
            summary.insert(0, "file", p.name)
            all_summaries.append(summary)
    else:
        if input_path.suffix.lower() != ".csv":
            print("Please provide a CSV file or a directory of CSVs.")
            return
        print(f"[INFO] Analyzing {input_path.name} ...")
        summary = analyze_file(input_path, outdir, time_unit, detrend_method, max_peaks, make_plots)
        summary.insert(0, "file", input_path.name)
        all_summaries.append(summary)

    if all_summaries:
        big = pd.concat(all_summaries, ignore_index=True)
        big.to_csv(Path(outdir) / "ALL_frequency_summary.csv", index=False)
        print(f"[OK] Wrote combined summary to {Path(outdir) / 'ALL_frequency_summary.csv'}")

def main():
    parser = argparse.ArgumentParser(description="Frequency analysis for TFB-format time series CSVs.")
    parser.add_argument("--input", required=True, help="CSV file in TFB long format, or a directory containing such CSVs.")
    parser.add_argument("--outdir", default="./freq_report", help="Output directory for CSV summaries and optional figures.")
    parser.add_argument("--time_unit", default="auto", choices=["auto","second","minute","hour","day","step"], help="Interpretation of the time unit. 'auto' tries to infer sensible unit from timestamps.")
    parser.add_argument("--detrend", default="median", choices=["none","median","linear","ema"], help="Detrending method before spectral analysis.")
    parser.add_argument("--max-peaks", type=int, default=5, help="Maximum number of dominant spectral peaks to report.")
    parser.add_argument("--plots", action="store_true", help="If set, saves periodogram plots per series.")
    args = parser.parse_args()
    analyze_input(args.input, args.outdir, args.time_unit, args.detrend, args.max_peaks, args.plots)

if __name__ == "__main__":
    main()
