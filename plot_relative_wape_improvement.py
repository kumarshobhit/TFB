#!/usr/bin/env python3
"""Plot relative WAPE improvements for calibrated vs control PE settings."""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter


DATASET_ORDER = ["ETTh1", "ETTm1", "Weather", "Solar"]
RANDOM_LABELS = ["random500", "random5000", "random50000"]
METHOD_CONFIG = {
    "RoPE": {
        "color": "#1f77b4",
        "summary_path": "result/rope_seed_analysis/summary.csv",
        "figure_path": "figures/wape_relative_improvement_rope.png",
        "json_path": "figures/wape_relative_improvement_rope_summary.json",
    },
    "SineSPE": {
        "color": "#ff7f0e",
        "summary_path": "result/sine_seed_analysis/summary.csv",
        "figure_path": "figures/wape_relative_improvement_sine.png",
        "json_path": "figures/wape_relative_improvement_sine_summary.json",
    },
}
BAR_COLORS = {
    "Best Random": "#9aa0a6",
    "Calibrated Best": "#2ca02c",
}
DISPLAY_LABELS = {
    "SineSPE": "SinePE",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot relative WAPE improvements over the default base frequency."
    )
    parser.add_argument(
        "--rope-summary",
        default=METHOD_CONFIG["RoPE"]["summary_path"],
        help="CSV with RoPE multi-seed summary values.",
    )
    parser.add_argument(
        "--sine-summary",
        default=METHOD_CONFIG["SineSPE"]["summary_path"],
        help="CSV with SineSPE multi-seed summary values.",
    )
    parser.add_argument(
        "--rope-output",
        default=METHOD_CONFIG["RoPE"]["figure_path"],
        help="Output PNG path for the RoPE figure.",
    )
    parser.add_argument(
        "--sine-output",
        default=METHOD_CONFIG["SineSPE"]["figure_path"],
        help="Output PNG path for the SineSPE figure.",
    )
    parser.add_argument(
        "--rope-summary-json",
        default=METHOD_CONFIG["RoPE"]["json_path"],
        help="Output JSON summary path for the RoPE figure.",
    )
    parser.add_argument(
        "--sine-summary-json",
        default=METHOD_CONFIG["SineSPE"]["json_path"],
        help="Output JSON summary path for the SineSPE figure.",
    )
    return parser.parse_args()


def compute_relative_improvement(default_wape: float, variant_wape: float) -> float:
    return (default_wape - variant_wape) / default_wape * 100.0


def format_random_label(base_label: str) -> str:
    mapping = {
        "random500": "base=500",
        "random5000": "base=5000",
        "random50000": "base=50000",
    }
    return mapping.get(base_label, base_label)


def load_method_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"dataset", "base_label", "mean_wape", "std_wape", "base_freq", "k"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    return df.copy()


def select_default_best_random_best(df: pd.DataFrame) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    for dataset in DATASET_ORDER:
        dataset_df = df[df["dataset"] == dataset].copy()
        if dataset_df.empty:
            raise ValueError(f"Dataset {dataset} missing from summary table")

        default_row = dataset_df.loc[dataset_df["base_label"] == "default"]
        best_row = dataset_df.loc[dataset_df["base_label"] == "best"]
        random_rows = dataset_df.loc[dataset_df["base_label"].isin(RANDOM_LABELS)].copy()
        if default_row.empty or best_row.empty or random_rows.empty:
            raise ValueError(f"Dataset {dataset} is missing default, best, or random rows")

        default = default_row.iloc[0]
        best = best_row.iloc[0]
        best_random = random_rows.loc[random_rows["mean_wape"].idxmin()]

        records.append(
            {
                "dataset": dataset,
                "default_mean_wape": float(default["mean_wape"]),
                "default_std_wape": float(default["std_wape"]),
                "best_random_label": str(best_random["base_label"]),
                "best_random_pretty_label": format_random_label(str(best_random["base_label"])),
                "best_random_base_freq": float(best_random["base_freq"]),
                "best_random_mean_wape": float(best_random["mean_wape"]),
                "best_random_std_wape": float(best_random["std_wape"]),
                "best_random_k": None if pd.isna(best_random["k"]) else int(best_random["k"]),
                "best_random_relative_improvement_pct": compute_relative_improvement(
                    float(default["mean_wape"]), float(best_random["mean_wape"])
                ),
                "calibrated_best_base_freq": float(best["base_freq"]),
                "calibrated_best_mean_wape": float(best["mean_wape"]),
                "calibrated_best_std_wape": float(best["std_wape"]),
                "calibrated_best_k": None if pd.isna(best["k"]) else int(best["k"]),
                "calibrated_best_relative_improvement_pct": compute_relative_improvement(
                    float(default["mean_wape"]), float(best["mean_wape"])
                ),
            }
        )
    return pd.DataFrame(records)


def annotate_bars(ax: plt.Axes, bars, values: List[float]) -> None:
    for bar, value in zip(bars, values):
        x = bar.get_x() + bar.get_width() / 2.0
        y = float(bar.get_height())
        if value >= 0:
            ax.text(
                x,
                y + 0.10,
                f"{value:+.2f}%",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
                color="#333333",
            )
        else:
            ax.text(
                x,
                y - 0.12,
                f"{value:+.2f}%",
                ha="center",
                va="top",
                fontsize=11,
                fontweight="bold",
                color="#333333",
            )


def compute_y_limits(random_values: List[float], calibrated_values: List[float]) -> Tuple[float, float]:
    all_values = random_values + calibrated_values + [0.0]
    ymin = min(all_values)
    ymax = max(all_values)
    lower_pad = max(0.25, 0.18 * max(1.0, abs(ymin)))
    upper_pad = max(0.55, 0.16 * max(1.0, abs(ymax)))
    return ymin - lower_pad, ymax + upper_pad


def plot_method_figure(method_name: str, df: pd.DataFrame, output_path: Path) -> None:
    plt.rc("font", size=14)
    plt.rc("axes", titlesize=19, labelsize=15)
    plt.rc("xtick", labelsize=13)
    plt.rc("ytick", labelsize=13)
    plt.rc("legend", fontsize=12)

    x = np.arange(len(DATASET_ORDER))
    width = 0.34
    random_values = df["best_random_relative_improvement_pct"].tolist()
    calibrated_values = df["calibrated_best_relative_improvement_pct"].tolist()
    ymin, ymax = compute_y_limits(random_values, calibrated_values)

    fig, ax = plt.subplots(figsize=(12.4, 7.4))
    bars_random = ax.bar(
        x - width / 2.0,
        random_values,
        width=width,
        label="Best Random",
        color=BAR_COLORS["Best Random"],
        edgecolor="#ffffff",
        linewidth=1.0,
        alpha=0.95,
        hatch="//",
    )
    bars_calibrated = ax.bar(
        x + width / 2.0,
        calibrated_values,
        width=width,
        label="Calibrated Best",
        color=METHOD_CONFIG[method_name]["color"],
        edgecolor="#ffffff",
        linewidth=1.0,
        alpha=0.95,
    )
    annotate_bars(ax, bars_random, random_values)
    annotate_bars(ax, bars_calibrated, calibrated_values)

    ax.axhline(0.0, color="#555555", linewidth=1.2)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(x)
    ax.set_xticklabels(DATASET_ORDER)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Relative WAPE Improvement vs Default (%)")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{val:.0f}%"))
    display_name = DISPLAY_LABELS.get(method_name, method_name)
    ax.set_title(f"{display_name}: Relative WAPE Improvement Over Default Base", pad=20)
    ax.grid(axis="y", alpha=0.18, linewidth=0.7, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),
        ncol=2,
        frameon=False,
        columnspacing=1.8,
        handlelength=2.0,
    )

    fig.text(
        0.5,
        0.01,
        "Gray hatched bars show the best random base among {500, 5000, 50000}; colored bars show the calibrated best. Values use 3-seed mean WAPE; higher is better.",
        ha="center",
        va="bottom",
        fontsize=12,
        color="#4c4c4c",
    )
    fig.tight_layout(rect=[0.02, 0.07, 0.98, 0.92])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_method_summary(method_name: str, df: pd.DataFrame, output_path: Path) -> None:
    payload: List[Dict[str, object]] = []
    for row in df.itertuples(index=False):
        payload.append(
            {
                "dataset": row.dataset,
                "method": method_name,
                "default_mean_wape": float(row.default_mean_wape),
                "default_std_wape": float(row.default_std_wape),
                "best_random": {
                    "label": row.best_random_label,
                    "pretty_label": row.best_random_pretty_label,
                    "base_freq": float(row.best_random_base_freq),
                    "k": None if pd.isna(row.best_random_k) else int(row.best_random_k),
                    "mean_wape": float(row.best_random_mean_wape),
                    "std_wape": float(row.best_random_std_wape),
                    "relative_improvement_pct": float(row.best_random_relative_improvement_pct),
                },
                "calibrated_best": {
                    "base_freq": float(row.calibrated_best_base_freq),
                    "k": None if pd.isna(row.calibrated_best_k) else int(row.calibrated_best_k),
                    "mean_wape": float(row.calibrated_best_mean_wape),
                    "std_wape": float(row.calibrated_best_std_wape),
                    "relative_improvement_pct": float(row.calibrated_best_relative_improvement_pct),
                },
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def build_all_tables(args: argparse.Namespace) -> List[Tuple[str, pd.DataFrame, Path, Path]]:
    rope_df = select_default_best_random_best(load_method_table(Path(args.rope_summary)))
    sine_df = select_default_best_random_best(load_method_table(Path(args.sine_summary)))
    return [
        ("RoPE", rope_df, Path(args.rope_output), Path(args.rope_summary_json)),
        ("SineSPE", sine_df, Path(args.sine_output), Path(args.sine_summary_json)),
    ]


def main() -> None:
    args = parse_args()
    for method_name, df, fig_path, json_path in build_all_tables(args):
        plot_method_figure(method_name, df, fig_path)
        write_method_summary(method_name, df, json_path)
        print(f"Saved {method_name} figure: {fig_path.resolve()}")
        print(f"Saved {method_name} summary: {json_path.resolve()}")


if __name__ == "__main__":
    main()
