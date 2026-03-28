#!/usr/bin/env python3
import argparse
import csv
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rc("font", size=16)
plt.rc("axes", titlesize=18, labelsize=17)
plt.rc("xtick", labelsize=14)
plt.rc("ytick", labelsize=14)
plt.rc("legend", fontsize=14)


ANNOTATION_OFFSETS = {
    "ETTh1": (10, -48),
    "ETTm1": (10, -48),
    "Solar": (10, -48),
    "Weather": (10, -48),
}


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    path: str


DEFAULT_DATASETS: Tuple[DatasetConfig, ...] = (
    DatasetConfig("ETTh1", "dataset/forecasting/ETTh1.csv"),
    DatasetConfig("ETTm1", "dataset/forecasting/ETTm1.csv"),
    DatasetConfig("Solar", "dataset/forecasting/Solar_ci64_every2.csv"),
    DatasetConfig("Weather", "dataset/forecasting/Weather.csv"),
)


def _parse_dataset_items(items: Sequence[str]) -> Tuple[DatasetConfig, ...]:
    if not items:
        return DEFAULT_DATASETS
    parsed: List[DatasetConfig] = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"--dataset must use NAME=PATH format, got: {item}")
        name, path = item.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError(f"--dataset must use NAME=PATH format, got: {item}")
        parsed.append(DatasetConfig(name=name, path=path))
    return tuple(parsed)


def _load_numeric_frame(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "cols" in df.columns and "data" in df.columns:
        index_col = "date" if "date" in df.columns else None
        df = df.pivot(index=index_col, columns="cols", values="data").reset_index()
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.empty:
        raise ValueError(f"No numeric columns found in {path}")
    return numeric_df


def _channel_spectrum(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    centered = values - np.mean(values)
    fft_vals = np.fft.rfft(centered)
    freqs = np.fft.rfftfreq(values.size, d=1.0)
    mask = freqs > 0.0
    freqs = freqs[mask]
    magnitudes = np.abs(fft_vals[mask])
    if freqs.size == 0:
        raise ValueError("No positive FFT frequencies available.")
    return freqs, magnitudes


def _aggregate_dataset_spectrum(frame: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, int]:
    spectra_by_length: Dict[int, List[np.ndarray]] = {}
    freqs_by_length: Dict[int, np.ndarray] = {}

    for col in frame.columns:
        series = frame[col].dropna().to_numpy(dtype=float)
        if series.size < 4:
            continue
        try:
            freqs, magnitudes = _channel_spectrum(series)
        except ValueError:
            continue
        spectra_by_length.setdefault(series.size, []).append(magnitudes)
        freqs_by_length.setdefault(series.size, freqs)

    if not spectra_by_length:
        raise ValueError("No usable numeric channels for FFT aggregation.")

    best_len, spectra = max(spectra_by_length.items(), key=lambda item: len(item[1]))
    stacked = np.stack(spectra, axis=0)
    return freqs_by_length[best_len], stacked.mean(axis=0), len(spectra)


def _find_peak_index(freqs: np.ndarray, magnitudes: np.ndarray, min_peak_freq: float) -> int:
    if min_peak_freq < 0.0:
        raise ValueError("min_peak_freq must be >= 0.")
    candidate_mask = freqs >= min_peak_freq
    if not np.any(candidate_mask):
        return int(np.argmax(magnitudes))
    candidate_indices = np.where(candidate_mask)[0]
    return int(candidate_indices[np.argmax(magnitudes[candidate_indices])])


def _top_frequency_strings(freqs: np.ndarray, magnitudes: np.ndarray, top_n: int = 5) -> List[str]:
    order = np.argsort(magnitudes)[::-1][:top_n]
    return [f"{float(freqs[idx]):.5f}" for idx in order]


def _write_summary_csv(path: str, rows: Iterable[Dict[str, object]]) -> None:
    columns = [
        "dataset",
        "source_path",
        "n_channels_used",
        "dominant_freq",
        "dominant_period_steps",
        "dominant_magnitude",
        "min_peak_freq",
        "top_5_freqs",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_figure(
    datasets: Sequence[DatasetConfig],
    output_path: str,
    summary_path: str,
    min_peak_freq: float,
) -> List[Dict[str, object]]:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    axes_list = list(axes.flatten())
    summary_rows: List[Dict[str, object]] = []

    for ax, config in zip(axes_list, datasets):
        frame = _load_numeric_frame(config.path)
        freqs, magnitudes, n_channels = _aggregate_dataset_spectrum(frame)
        peak_idx = _find_peak_index(freqs, magnitudes, min_peak_freq=min_peak_freq)
        peak_freq = float(freqs[peak_idx])
        peak_mag = float(magnitudes[peak_idx])
        peak_period = 1.0 / peak_freq

        ax.plot(freqs, magnitudes, color="#1f4e79", linewidth=1.3)
        ax.axvline(peak_freq, color="#b22222", linestyle="--", linewidth=1.0)
        ax.scatter([peak_freq], [peak_mag], color="#b22222", s=38, zorder=3)
        ax.annotate(
            f"f_d={peak_freq:.5f}\nP={peak_period:.1f}",
            xy=(peak_freq, peak_mag),
            xytext=ANNOTATION_OFFSETS.get(config.name, (10, 10)),
            textcoords="offset points",
            fontsize=13,
            color="#b22222",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "none", "alpha": 0.85},
        )
        ax.set_title(f"{config.name} (channels={n_channels})", fontsize=18)
        ax.set_xlim(0.0, 0.5)
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)

        summary_rows.append(
            {
                "dataset": config.name,
                "source_path": config.path,
                "n_channels_used": n_channels,
                "dominant_freq": f"{peak_freq:.5f}",
                "dominant_period_steps": f"{peak_period:.2f}",
                "dominant_magnitude": f"{peak_mag:.6f}",
                "min_peak_freq": f"{min_peak_freq:.5f}",
                "top_5_freqs": ",".join(_top_frequency_strings(freqs, magnitudes)),
            }
        )

    for ax in axes_list[len(datasets) :]:
        ax.axis("off")

    fig.supxlabel("Frequency (cycles / time step)")
    fig.supylabel("Magnitude")
    fig.suptitle("Raw Dataset Frequency Spectra for RoPE Base Selection", fontsize=20)
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    _write_summary_csv(summary_path, summary_rows)
    return summary_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot raw FFT magnitude spectra for the datasets used in the RoPE base analysis."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Dataset mapping in NAME=PATH format. Repeatable. Defaults to ETTh1, ETTm1, Solar_ci64_every2, Weather.",
    )
    parser.add_argument(
        "--output",
        default="result/rope_k_analysis/dataset_frequency_spectra.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--summary-csv",
        default="result/rope_k_analysis/dataset_frequency_spectra_summary.csv",
        help="Output CSV path for dominant frequencies.",
    )
    parser.add_argument(
        "--min-peak-freq",
        type=float,
        default=1.0 / 336.0,
        help="Ignore frequencies below this threshold when selecting the highlighted peak. The full raw spectrum is still plotted.",
    )
    args = parser.parse_args()

    datasets = _parse_dataset_items(args.dataset)
    summary_rows = build_figure(
        datasets=datasets,
        output_path=args.output,
        summary_path=args.summary_csv,
        min_peak_freq=args.min_peak_freq,
    )

    print(f"Saved figure: {os.path.abspath(args.output)}")
    print(f"Saved summary: {os.path.abspath(args.summary_csv)}")
    for row in summary_rows:
        print(
            f"{row['dataset']}: dominant_freq={row['dominant_freq']} "
            f"period={row['dominant_period_steps']} min_peak_freq={row['min_peak_freq']} "
            f"top_5={row['top_5_freqs']}"
        )


if __name__ == "__main__":
    main()
