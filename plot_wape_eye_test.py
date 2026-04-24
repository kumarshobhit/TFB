"""Build report-facing qualitative forecast comparison figures.

The script can generate a single eye-test plot from explicit archive paths or
run in batch mode for the built-in RoPE comparisons used in the report and
appendix.
"""

import argparse
import base64
import io
import json
import pickle
import tarfile
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EYE_TEST_FONT_SIZES = {
    "font": 16,
    "axes_title": 18,
    "axes_label": 17,
    "xtick": 14,
    "ytick": 14,
    "legend": 14,
    "annotation": 12.5,
}

DEFAULT_COLOR = "#1f77b4"
CALIBRATED_COLOR = "#ff7f0e"
GROUND_TRUTH_COLOR = "black"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an interpretability figure that compares a default and calibrated "
            "forecasting run using saved rolling-forecast predictions."
        )
    )
    parser.add_argument(
        "--default-archive",
        default="result/ETTm1_seed2021/TST_rotary/TST_rope_10000_1.tar.gz",
        help="Path to the default run archive (.tar.gz).",
    )
    parser.add_argument(
        "--calibrated-archive",
        default="result/ETTm1_seed2021/TST_rotary/TST_rope_233.295_1.tar.gz",
        help="Path to the calibrated run archive (.tar.gz).",
    )
    parser.add_argument(
        "--default-feature-metrics",
        default="result/ETTm1_seed2021/TST_rotary/TST_per_feature_metrics_rope_10000.csv",
        help="CSV with per-feature metrics for the default run.",
    )
    parser.add_argument(
        "--calibrated-feature-metrics",
        default="result/ETTm1_seed2021/TST_rotary/TST_per_feature_metrics_rope_233.295_1.csv",
        help="CSV with per-feature metrics for the calibrated run.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=168,
        help="Number of continuous test timestamps to include in the eye-test window.",
    )
    parser.add_argument(
        "--top-k-features",
        type=int,
        default=3,
        help="Evaluate the top K feature-level WAPE gains before choosing the best window.",
    )
    parser.add_argument(
        "--output-dir",
        default="result/eye_test_ettm1_seed2021",
        help="Directory where the figure and summary files will be written.",
    )
    parser.add_argument(
        "--output-prefix",
        default="rope_233295_vs_10000",
        help="Filename prefix for generated outputs.",
    )
    parser.add_argument(
        "--batch-mode",
        action="store_true",
        help="Generate a set of eye-test plots for the built-in multi-dataset rotary comparisons.",
    )
    parser.add_argument(
        "--batch-output-dir",
        default="result/eye_test_rotary_batch",
        help="Root directory for multi-dataset outputs when --batch-mode is used.",
    )
    return parser.parse_args()


def canonical_base_label(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def format_base_for_filename(value: float) -> str:
    return canonical_base_label(value)


def built_in_batch_folder_candidates() -> Dict[str, List[str]]:
    return {
        "ETTm1": [
            "result/ETTm1_seed2021/TST_rotary",
            "result/ETTm1_seed2022/TST_rotary",
            "result/ETTm1_seed2023/TST_rotary",
            "result/ETTm1_22ndfeb/TST_rotary",
        ],
        "ETTh1": [
            "result/ETTh1_seed2022/TST_rotary",
            "result/ETTh1_seed2023/TST_rotary",
            "result/ETTh1_22ndfeb/TST_rotary",
        ],
        "Weather": [
            "result/Weather_seed2021/TST_rotary",
            "result/Weather_seed2022/TST_rotary",
            "result/Weather_seed2023/TST_rotary",
            "result/Weather_22ndfeb/TST_rotary",
        ],
        "Solar": [
            "result/Solar_seed2021/TST_rotary_ci64_every2",
            "result/Solar_seed2022/TST_rotary_ci64_every2",
            "result/Solar_seed2023/TST_rotary_ci64_every2",
            "result/Solar_22ndfeb/TST_rotary_ci64_every2",
        ],
    }


def read_best_vs_default_table(root: Path) -> pd.DataFrame:
    path = root / "result/rope_k_analysis/rope_k_best_vs_default.csv"
    df = pd.read_csv(path)
    needed = {"dataset", "best_wape", "default_wape"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    return df


def extract_file_suffix(path: Path, prefix: str, suffix: str) -> str:
    name = path.name
    if not name.startswith(prefix) or not name.endswith(suffix):
        raise ValueError(f"Unexpected file naming pattern: {path}")
    return name[len(prefix) : -len(suffix)]


def choose_matching_file(
    folder: Path,
    stem_prefix: str,
    preferred_suffixes: Sequence[str],
    suffix: str,
) -> Optional[Path]:
    candidates = sorted(folder.glob(f"{stem_prefix}*{suffix}"))
    if not candidates:
        return None
    name_to_path = {extract_file_suffix(path, stem_prefix, suffix): path for path in candidates}
    for pref in preferred_suffixes:
        if pref in name_to_path:
            return name_to_path[pref]
    for path in candidates:
        token = extract_file_suffix(path, stem_prefix, suffix)
        if token.startswith(preferred_suffixes[0]):
            return path
    return candidates[0]


def resolve_dataset_entry(
    root: Path,
    dataset: str,
    best_base: float,
) -> Dict[str, str]:
    folder_candidates = built_in_batch_folder_candidates().get(dataset, [])
    if not folder_candidates:
        raise ValueError(f"No folder candidates configured for dataset {dataset}")

    best_label = format_base_for_filename(best_base)
    preferred_cal_suffixes = (best_label, f"{best_label}_1", f"{best_label}_2")
    preferred_default_suffixes = ("10000", "10000_1", "10000_2")

    for rel_folder in folder_candidates:
        folder = root / rel_folder
        if not folder.exists():
            continue

        default_archive = choose_matching_file(folder, "TST_rope_", preferred_default_suffixes, ".tar.gz")
        calibrated_archive = choose_matching_file(
            folder, "TST_rope_", preferred_cal_suffixes, ".tar.gz"
        )
        if default_archive is None or calibrated_archive is None:
            continue

        default_suffix = extract_file_suffix(default_archive, "TST_rope_", ".tar.gz")
        calibrated_suffix = extract_file_suffix(calibrated_archive, "TST_rope_", ".tar.gz")
        if not calibrated_suffix.startswith(best_label):
            continue
        default_metrics = choose_matching_file(
            folder,
            "TST_per_feature_metrics_rope_",
            (default_suffix, "10000", "10000_1"),
            ".csv",
        )
        calibrated_metrics = choose_matching_file(
            folder,
            "TST_per_feature_metrics_rope_",
            (calibrated_suffix, best_label, f"{best_label}_1", f"{best_label}_2"),
            ".csv",
        )
        if default_metrics is None or calibrated_metrics is None:
            continue
        metric_suffix = extract_file_suffix(
            calibrated_metrics, "TST_per_feature_metrics_rope_", ".csv"
        )
        if not metric_suffix.startswith(best_label):
            continue

        return {
            "dataset": dataset,
            "folder": str(folder),
            "default_archive": str(default_archive),
            "calibrated_archive": str(calibrated_archive),
            "default_feature_metrics": str(default_metrics),
            "calibrated_feature_metrics": str(calibrated_metrics),
            "default_suffix": default_suffix,
            "calibrated_suffix": calibrated_suffix,
        }

    raise ValueError(
        f"Could not find a matched default-vs-calibrated artifact set for {dataset} with base {best_label}"
    )


def build_batch_entries(root: Path) -> List[Dict[str, str]]:
    df = read_best_vs_default_table(root)
    best_base_map = {
        "ETTm1": 233.295,
        "ETTh1": 45200.1,
        "Weather": 131.291,
        "Solar": 131.291,
    }
    entries = []
    for row in df.itertuples(index=False):
        dataset = str(row.dataset)
        if dataset not in best_base_map:
            continue
        entry = resolve_dataset_entry(root, dataset, best_base_map[dataset])
        entry["best_wape"] = float(row.best_wape)
        entry["default_wape_reported"] = float(row.default_wape)
        entries.append(entry)
    return entries


def read_single_member_csv_from_tar(tar_path: Path) -> pd.DataFrame:
    with tarfile.open(tar_path, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.isfile()]
        if len(members) != 1:
            raise ValueError(f"Expected exactly one file in {tar_path}, found {len(members)}")
        extracted = tar.extractfile(members[0])
        if extracted is None:
            raise ValueError(f"Could not read archive payload from {tar_path}")
        raw = extracted.read()
    return pd.read_csv(io.BytesIO(raw))


def decode_artifact(encoded: str):
    if not isinstance(encoded, str) or not encoded:
        raise ValueError("Missing encoded artifact data")
    return pickle.loads(base64.b64decode(encoded))


def json_load(value: str) -> Dict:
    if not isinstance(value, str) or not value:
        return {}
    return json.loads(value)


def verify_runs_match(default_row: pd.Series, calibrated_row: pd.Series) -> Dict[str, object]:
    if default_row["model_name"] != calibrated_row["model_name"]:
        raise ValueError("Model names differ between runs")
    if default_row["file_name"] != calibrated_row["file_name"]:
        raise ValueError("Datasets differ between runs")

    default_strategy = json_load(default_row["strategy_args"])
    calibrated_strategy = json_load(calibrated_row["strategy_args"])
    if default_strategy != calibrated_strategy:
        raise ValueError("Strategy args differ between runs")

    default_params = json_load(default_row["model_params"])
    calibrated_params = json_load(calibrated_row["model_params"])
    default_base = default_params.pop("base_freq", None)
    calibrated_base = calibrated_params.pop("base_freq", None)
    if default_base is None:
        default_base = 10000.0
    if calibrated_base is None:
        calibrated_base = 10000.0
    if default_params != calibrated_params:
        raise ValueError("Model params differ by more than base_freq")

    return {
        "dataset": default_row["file_name"],
        "model_name": default_row["model_name"],
        "strategy_args": default_strategy,
        "model_params_without_base_freq": default_params,
        "default_base_freq": default_base,
        "calibrated_base_freq": calibrated_base,
    }


def load_feature_metric_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = {"feature_idx", "feature_name", "wape"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")
    return df[["feature_idx", "feature_name", "wape"]].copy()


def select_candidate_features(
    default_metrics: pd.DataFrame,
    calibrated_metrics: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    merged = default_metrics.merge(
        calibrated_metrics,
        on=["feature_idx", "feature_name"],
        suffixes=("_default", "_calibrated"),
    )
    merged["delta_wape"] = merged["wape_default"] - merged["wape_calibrated"]
    merged = merged.sort_values(
        ["delta_wape", "wape_default"], ascending=[False, False]
    ).reset_index(drop=True)
    return merged.head(top_k).copy()


def normalize_decoded_obj(decoded_obj):
    if isinstance(decoded_obj, np.ndarray):
        if decoded_obj.ndim == 3:
            return decoded_obj.astype(float, copy=False)
        if decoded_obj.ndim == 1 and len(decoded_obj) > 0 and isinstance(decoded_obj[0], pd.DataFrame):
            return [frame.copy() for frame in decoded_obj.tolist()]
        raise TypeError(f"Unsupported ndarray artifact shape: {decoded_obj.shape!r}")

    if isinstance(decoded_obj, list):
        frames = decoded_obj
    elif isinstance(decoded_obj, pd.DataFrame):
        frames = [decoded_obj]
    else:
        raise TypeError(f"Unsupported decoded artifact type: {type(decoded_obj)!r}")

    out = []
    for frame in frames:
        if not isinstance(frame, pd.DataFrame):
            frame = pd.DataFrame(frame)
        if frame.empty:
            continue
        out.append(frame.copy())
    if not out:
        raise ValueError("Decoded artifact produced no usable frames")
    return out


def aggregate_continuous_series(
    actual_frames,
    pred_frames,
    feature_idx: int,
    feature_name: str,
) -> pd.DataFrame:
    if isinstance(actual_frames, np.ndarray) and isinstance(pred_frames, np.ndarray):
        if actual_frames.shape != pred_frames.shape:
            raise ValueError("Actual and prediction arrays do not match in shape")
        if actual_frames.ndim != 3:
            raise ValueError(f"Expected 3D rolling arrays, got shape {actual_frames.shape}")
        if feature_idx >= actual_frames.shape[2]:
            raise IndexError(f"Feature index {feature_idx} is out of bounds")

        num_rollings, horizon, _ = actual_frames.shape
        series_len = num_rollings + horizon - 1
        sum_actual = np.zeros(series_len, dtype=float)
        sum_pred = np.zeros(series_len, dtype=float)
        counts = np.zeros(series_len, dtype=float)

        actual_feature = actual_frames[:, :, feature_idx]
        pred_feature = pred_frames[:, :, feature_idx]
        base_idx = np.arange(num_rollings)
        for offset in range(horizon):
            target_idx = base_idx + offset
            sum_actual[target_idx] += actual_feature[:, offset]
            sum_pred[target_idx] += pred_feature[:, offset]
            counts[target_idx] += 1.0

        actual = sum_actual / counts
        pred = sum_pred / counts
        return pd.DataFrame(
            {
                "timestamp": np.arange(series_len),
                "actual": actual,
                "pred": pred,
                "feature_name": feature_name,
            }
        )

    if len(actual_frames) != len(pred_frames):
        raise ValueError("Actual and prediction frame counts do not match")

    rows: List[Tuple[object, float, float]] = []
    for actual_df, pred_df in zip(actual_frames, pred_frames):
        if feature_idx >= actual_df.shape[1] or feature_idx >= pred_df.shape[1]:
            raise IndexError(f"Feature index {feature_idx} is out of bounds")
        actual_col = pd.to_numeric(actual_df.iloc[:, feature_idx], errors="coerce")
        pred_col = pd.to_numeric(pred_df.iloc[:, feature_idx], errors="coerce")
        for idx, actual_val, pred_val in zip(actual_df.index, actual_col, pred_col):
            if pd.isna(actual_val) or pd.isna(pred_val):
                continue
            rows.append((idx, float(actual_val), float(pred_val)))

    merged = pd.DataFrame(rows, columns=["timestamp", "actual", "pred"])
    if merged.empty:
        raise ValueError("No valid rows were decoded for the requested feature")

    merged["timestamp_dt"] = pd.to_datetime(merged["timestamp"], errors="coerce")
    sort_key = "timestamp_dt" if merged["timestamp_dt"].notna().any() else "timestamp"
    grouped = (
        merged.groupby("timestamp", sort=False)[["actual", "pred"]]
        .mean()
        .reset_index()
    )

    ts_lookup = (
        merged[["timestamp", sort_key]]
        .drop_duplicates(subset=["timestamp"])
        .set_index("timestamp")[sort_key]
    )
    grouped[sort_key] = grouped["timestamp"].map(ts_lookup)
    grouped = grouped.sort_values(sort_key).reset_index(drop=True)
    grouped["feature_name"] = feature_name
    return grouped[["timestamp", "actual", "pred", "feature_name"]]


def build_comparison_frame(
    default_actual,
    default_pred,
    calibrated_actual,
    calibrated_pred,
    feature_idx: int,
    feature_name: str,
) -> pd.DataFrame:
    default_series = aggregate_continuous_series(
        default_actual, default_pred, feature_idx, feature_name
    )
    calibrated_series = aggregate_continuous_series(
        calibrated_actual, calibrated_pred, feature_idx, feature_name
    )

    merged = default_series.merge(
        calibrated_series[["timestamp", "actual", "pred"]],
        on="timestamp",
        suffixes=("_default", "_calibrated"),
    )
    merged["actual"] = merged[["actual_default", "actual_calibrated"]].mean(axis=1)
    merged["pred_default"] = merged["pred_default"]
    merged["pred_calibrated"] = merged["pred_calibrated"]
    merged["abs_err_default"] = np.abs(merged["pred_default"] - merged["actual"])
    merged["abs_err_calibrated"] = np.abs(merged["pred_calibrated"] - merged["actual"])
    merged["improvement"] = merged["abs_err_default"] - merged["abs_err_calibrated"]
    merged["feature_name"] = feature_name
    return merged[
        [
            "timestamp",
            "feature_name",
            "actual",
            "pred_default",
            "pred_calibrated",
            "abs_err_default",
            "abs_err_calibrated",
            "improvement",
        ]
    ]


def compute_peakiness(actual: pd.Series) -> float:
    if len(actual) < 3:
        return 0.0
    amplitude = float(actual.max() - actual.min())
    slope = float(np.max(np.abs(np.diff(actual.to_numpy()))))
    return amplitude + slope


def choose_best_window(series_df: pd.DataFrame, window_size: int) -> Tuple[int, int, Dict[str, float]]:
    if len(series_df) < window_size:
        raise ValueError(
            f"Series length {len(series_df)} is shorter than window size {window_size}"
        )

    improvement = series_df["improvement"].to_numpy()
    actual = series_df["actual"]
    improvement_sums = np.convolve(improvement, np.ones(window_size), mode="valid")
    peakiness_scores = np.array(
        [compute_peakiness(actual.iloc[i : i + window_size]) for i in range(len(improvement_sums))]
    )

    best_improvement = float(np.max(improvement_sums))
    improvement_threshold = best_improvement - max(1e-8, abs(best_improvement) * 0.01)
    candidate_indices = np.flatnonzero(improvement_sums >= improvement_threshold)
    best_idx = int(
        max(candidate_indices, key=lambda i: (peakiness_scores[i], improvement_sums[i], -i))
    )
    return best_idx, best_idx + window_size, {
        "window_improvement_sum": float(improvement_sums[best_idx]),
        "window_peakiness": float(peakiness_scores[best_idx]),
    }


def choose_feature_and_window(
    candidate_features: pd.DataFrame,
    default_actual: Sequence[pd.DataFrame],
    default_pred: Sequence[pd.DataFrame],
    calibrated_actual: Sequence[pd.DataFrame],
    calibrated_pred: Sequence[pd.DataFrame],
    window_size: int,
) -> Tuple[pd.Series, pd.DataFrame, int, int, Dict[str, float]]:
    best_payload = None
    for _, feature_row in candidate_features.iterrows():
        feature_idx = int(feature_row["feature_idx"])
        comparison = build_comparison_frame(
            default_actual,
            default_pred,
            calibrated_actual,
            calibrated_pred,
            feature_idx,
            str(feature_row["feature_name"]),
        )
        start, end, window_stats = choose_best_window(comparison, window_size)
        ranking = (
            window_stats["window_improvement_sum"],
            window_stats["window_peakiness"],
            float(feature_row["delta_wape"]),
        )
        if best_payload is None or ranking > best_payload[0]:
            best_payload = (ranking, feature_row, comparison, start, end, window_stats)

    if best_payload is None:
        raise ValueError("No feature/window candidate could be selected")
    _, feature_row, comparison, start, end, window_stats = best_payload
    return feature_row, comparison, start, end, window_stats


def humanize_feature_name(name: str) -> str:
    mapping = {
        "HUFL": "High Useful Load",
        "MUFL": "Middle Useful Load",
        "LULL": "Low Useful Load",
        "sh (g/kg)": "Specific Humidity",
    }
    return mapping.get(name, name)


def apply_eye_test_style() -> None:
    plt.rc("font", size=EYE_TEST_FONT_SIZES["font"])
    plt.rc("axes", titlesize=EYE_TEST_FONT_SIZES["axes_title"], labelsize=EYE_TEST_FONT_SIZES["axes_label"])
    plt.rc("xtick", labelsize=EYE_TEST_FONT_SIZES["xtick"])
    plt.rc("ytick", labelsize=EYE_TEST_FONT_SIZES["ytick"])
    plt.rc("legend", fontsize=EYE_TEST_FONT_SIZES["legend"])


def build_detail_line(dataset_label: str, feature_name: str, run_info: Dict[str, object], start: int, end: int) -> str:
    return (
        f"{dataset_label}: feature={humanize_feature_name(feature_name)}, "
        f"seed={run_info['strategy_args'].get('seed')}, "
        f"horizon={run_info['strategy_args'].get('horizon')}, "
        f"window=[{start}, {end})"
    )


def choose_weather_combined_window(comparison: pd.DataFrame, window_size: int) -> Tuple[int, int]:
    actual = comparison["actual"].to_numpy()
    improvement = comparison["improvement"].to_numpy()
    candidates: List[Tuple[float, float, int]] = []
    for start in range(len(comparison) - window_size + 1):
        seg = actual[start : start + window_size]
        first25_range = float(seg[:25].max() - seg[:25].min())
        smooth90 = float(np.percentile(np.abs(np.diff(seg)), 90))
        score = float(improvement[start : start + window_size].sum())
        if score > 500 and first25_range < 250 and smooth90 < 40:
            candidates.append((smooth90, -score, start))
    if not candidates:
        return 1372, 1372 + window_size
    candidates.sort()
    best_start = candidates[0][2]
    return best_start, best_start + window_size


def plot_eye_test_axes(
    ax_main: plt.Axes,
    ax_err: plt.Axes,
    window_df: pd.DataFrame,
    feature_row: pd.Series,
    run_info: Dict[str, object],
    start: int,
    end: int,
    show_legend: bool,
) -> None:
    x = np.arange(len(window_df))
    dataset_label = str(run_info["dataset"]).replace(".csv", "")
    feature_label = humanize_feature_name(str(feature_row["feature_name"]))
    ax_main.plot(x, window_df["actual"], color=GROUND_TRUTH_COLOR, linewidth=2.8, label="Ground Truth")
    ax_main.plot(
        x,
        window_df["pred_default"],
        color=DEFAULT_COLOR,
        linewidth=2.1,
        linestyle="--",
        alpha=0.9,
        label=f"Default ({run_info['default_base_freq']:.0f})",
    )
    ax_main.plot(
        x,
        window_df["pred_calibrated"],
        color=CALIBRATED_COLOR,
        linewidth=2.3,
        alpha=0.95,
        label=f"Calibrated ({run_info['calibrated_base_freq']:.3f})",
    )
    ax_main.set_title(f"{dataset_label} — {feature_label}")
    ax_main.set_ylabel("Value")
    ax_main.grid(alpha=0.2, linewidth=0.6)
    ax_main.spines["top"].set_visible(False)
    ax_main.spines["right"].set_visible(False)
    if show_legend:
        if dataset_label in {"ETTh1", "ETTm1"}:
            ax_main.legend(loc="lower left", frameon=False)
        else:
            ax_main.legend(loc="upper left", frameon=False)

    metadata_text = (
        f"{dataset_label}\n"
        f"WAPE default={feature_row['wape_default']:.3f}\n"
        f"WAPE calibrated={feature_row['wape_calibrated']:.3f}\n"
        f"delta={feature_row['delta_wape']:.3f}"
    )
    box_x = 0.99
    box_y = 0.02
    va = "bottom"
    ha = "right"
    if dataset_label == "ETTh1":
        box_x = 0.98
        box_y = 0.10
        ha = "right"
        va = "bottom"
    elif dataset_label == "Weather":
        box_x = 0.02
        ha = "left"
        va = "top"

    ax_main.text(
        box_x,
        box_y,
        metadata_text,
        transform=ax_main.transAxes,
        ha=ha,
        va=va,
        fontsize=11.0,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "alpha": 0.7, "edgecolor": "#cccccc"},
    )

    ax_err.plot(
        x,
        window_df["abs_err_default"],
        color=DEFAULT_COLOR,
        linewidth=1.8,
        linestyle="--",
        label="|Default - Truth|",
    )
    ax_err.plot(
        x,
        window_df["abs_err_calibrated"],
        color=CALIBRATED_COLOR,
        linewidth=1.9,
        label="|Calibrated - Truth|",
    )
    ax_err.fill_between(
        x,
        window_df["abs_err_default"],
        window_df["abs_err_calibrated"],
        where=window_df["abs_err_default"] >= window_df["abs_err_calibrated"],
        color=CALIBRATED_COLOR,
        alpha=0.12,
    )
    ax_err.set_ylabel("Absolute Error")
    ax_err.grid(alpha=0.2, linewidth=0.6)
    ax_err.spines["top"].set_visible(False)
    ax_err.spines["right"].set_visible(False)
    if show_legend:
        ax_err.legend(loc="upper right", frameon=False)


def render_eye_test_figure(
    window_df: pd.DataFrame,
    feature_row: pd.Series,
    run_info: Dict[str, object],
    start: int,
    end: int,
    output_path: Path,
    dpi: int = 220,
) -> None:
    apply_eye_test_style()
    fig, (ax_main, ax_err) = plt.subplots(
        2,
        1,
        figsize=(14.5, 8.6),
        sharex=True,
        gridspec_kw={"height_ratios": [3.5, 1.3]},
    )
    plot_eye_test_axes(
        ax_main=ax_main,
        ax_err=ax_err,
        window_df=window_df,
        feature_row=feature_row,
        run_info=run_info,
        start=start,
        end=end,
        show_legend=True,
    )
    ax_err.set_xlabel("Time step (test window)")
    fig.tight_layout(rect=[0, 0.03, 1, 0.99], h_pad=1.4)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_outputs(
    output_dir: Path,
    output_prefix: str,
    comparison: pd.DataFrame,
    feature_row: pd.Series,
    run_info: Dict[str, object],
    start: int,
    end: int,
    window_stats: Dict[str, float],
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    window_df = comparison.iloc[start:end].reset_index(drop=True)
    fig_path = output_dir / f"{output_prefix}.png"
    render_eye_test_figure(
        window_df=window_df,
        feature_row=feature_row,
        run_info=run_info,
        start=start,
        end=end,
        output_path=fig_path,
        dpi=300,
    )

    window_csv_path = output_dir / f"{output_prefix}_window.csv"
    window_df.to_csv(window_csv_path, index=False)

    summary = {
        "dataset": run_info["dataset"],
        "model_name": run_info["model_name"],
        "feature_idx": int(feature_row["feature_idx"]),
        "feature_name": str(feature_row["feature_name"]),
        "wape_default": float(feature_row["wape_default"]),
        "wape_calibrated": float(feature_row["wape_calibrated"]),
        "delta_wape": float(feature_row["delta_wape"]),
        "default_base_freq": float(run_info["default_base_freq"]),
        "calibrated_base_freq": float(run_info["calibrated_base_freq"]),
        "window_size": int(len(window_df)),
        "window_start": int(start),
        "window_end": int(end),
        "window_improvement_sum": float(window_stats["window_improvement_sum"]),
        "window_peakiness": float(window_stats["window_peakiness"]),
        "strategy_args": run_info["strategy_args"],
        "model_params_without_base_freq": run_info["model_params_without_base_freq"],
        "default_archive_path": run_info.get("default_archive_path"),
        "calibrated_archive_path": run_info.get("calibrated_archive_path"),
        "default_feature_metrics_path": run_info.get("default_feature_metrics_path"),
        "calibrated_feature_metrics_path": run_info.get("calibrated_feature_metrics_path"),
        "artifact_folder": run_info.get("artifact_folder"),
    }
    summary_path = output_dir / f"{output_prefix}_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    return {
        "figure": fig_path,
        "window_csv": window_csv_path,
        "summary_json": summary_path,
    }


def save_combined_eye_test_figure(batch_output_dir: Path, rows: List[Dict[str, object]], skipped: List[Dict[str, object]]) -> Dict[str, Path]:
    combined_path = batch_output_dir / "combined_eye_test.png"
    summary_path = batch_output_dir / "combined_eye_test_summary.json"

    fig, axes = plt.subplots(
        nrows=len(rows) * 2,
        ncols=1,
        figsize=(14, 4.2 * len(rows)),
        sharex=False,
        gridspec_kw={"height_ratios": [3.2, 1.2] * len(rows)},
    )
    axes = np.atleast_1d(axes)

    for idx, row in enumerate(rows):
        ax_main = axes[2 * idx]
        ax_err = axes[2 * idx + 1]
        feature_row = pd.Series(row["feature_row"])
        run_info = row["run_info"]
        start = row["window_start"]
        end = row["window_end"]
        if row["dataset"] == "Weather":
            start, end = choose_weather_combined_window(row["comparison"], row["window_size"])
        window_df = row["comparison"].iloc[start:end].reset_index(drop=True)
        plot_eye_test_axes(
            ax_main=ax_main,
            ax_err=ax_err,
            window_df=window_df,
            feature_row=feature_row,
            run_info=run_info,
            start=start,
            end=end,
            show_legend=(idx == 0),
        )
        row["combined_window_start"] = start
        row["combined_window_end"] = end

    axes[-1].set_xlabel("Continuous test timestamp within selected window")
    fig.tight_layout(rect=[0, 0.03, 1, 0.99])
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(combined_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    combined_summary = {
        "combined_figure_path": str(combined_path),
        "datasets": [
            {
                "dataset": row["dataset"],
                "feature_name": humanize_feature_name(row["feature_name"]),
                "window_start": row["combined_window_start"],
                "window_end": row["combined_window_end"],
                "summary_json": row["summary_json"],
            }
            for row in rows
        ],
        "skipped": skipped,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(combined_summary, f, indent=2, sort_keys=True)
    return {"combined_figure": combined_path, "combined_summary": summary_path}


def save_etth1_k_sweep_figure(root: Path) -> Path:
    source_path = root / "result/ETTh1_22ndfeb/TST_rotary/k_vs_wape_mapped.csv"
    df = pd.read_csv(source_path)
    curve = df[df["k"].notna()].copy()
    curve["k"] = curve["k"].astype(int)
    baseline = df[df["x_label"] == "baseline_10000"].iloc[0]
    best_row = curve.loc[curve["wape"].idxmin()]
    k3_row = curve.loc[curve["k"] == 3].iloc[0]

    figures_dir = root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / "k_sweep_etth1.png"

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(curve["k"], curve["wape"], marker="o", linewidth=2.0, color=CALIBRATED_COLOR)
    ax.axvline(best_row["k"], linestyle="--", color=DEFAULT_COLOR, linewidth=1.5)
    ax.axhline(baseline["wape"], linestyle="--", color="#666666", linewidth=1.3)
    ax.annotate(
        f"best k={int(best_row['k'])}\nWAPE={best_row['wape']:.3f}",
        xy=(best_row["k"], best_row["wape"]),
        xytext=(best_row["k"] + 0.35, best_row["wape"] + 0.12),
        arrowprops={"arrowstyle": "->", "color": "#444444", "lw": 1.0},
        fontsize=10,
        ha="left",
        va="bottom",
    )
    ax.annotate(
        f"k=3\nWAPE={k3_row['wape']:.2f}",
        xy=(k3_row["k"], k3_row["wape"]),
        xytext=(k3_row["k"] + 0.35, k3_row["wape"] - 0.10),
        arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 1.0},
        fontsize=9.5,
        ha="left",
        va="top",
    )
    ax.text(
        0.98,
        0.03,
        f"Default RoPE WAPE = {baseline['wape']:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.92, "edgecolor": "#cccccc"},
    )
    ax.set_title("Effect of ladder index k on WAPE - ETTh1")
    ax.set_xlabel("Ladder index k")
    ax.set_ylabel("WAPE")
    ax.set_xticks(curve["k"].tolist())
    ax.grid(alpha=0.2, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def run_single_plot(
    default_archive: Path,
    calibrated_archive: Path,
    default_metrics_path: Path,
    calibrated_metrics_path: Path,
    output_dir: Path,
    output_prefix: str,
    top_k_features: int,
    window_size: int,
    extra_run_info: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    default_result = read_single_member_csv_from_tar(default_archive)
    calibrated_result = read_single_member_csv_from_tar(calibrated_archive)
    if len(default_result) != 1 or len(calibrated_result) != 1:
        raise ValueError("Each archive is expected to contain a single benchmark row")

    default_row = default_result.iloc[0]
    calibrated_row = calibrated_result.iloc[0]
    run_info = verify_runs_match(default_row, calibrated_row)
    run_info.update(
        {
            "default_archive_path": str(default_archive),
            "calibrated_archive_path": str(calibrated_archive),
            "default_feature_metrics_path": str(default_metrics_path),
            "calibrated_feature_metrics_path": str(calibrated_metrics_path),
            "artifact_folder": str(default_archive.parent),
        }
    )
    if extra_run_info:
        run_info.update(extra_run_info)

    default_actual = normalize_decoded_obj(decode_artifact(default_row["actual_data"]))
    default_pred = normalize_decoded_obj(decode_artifact(default_row["inference_data"]))
    calibrated_actual = normalize_decoded_obj(decode_artifact(calibrated_row["actual_data"]))
    calibrated_pred = normalize_decoded_obj(decode_artifact(calibrated_row["inference_data"]))

    default_metrics = load_feature_metric_table(default_metrics_path)
    calibrated_metrics = load_feature_metric_table(calibrated_metrics_path)
    candidates = select_candidate_features(
        default_metrics, calibrated_metrics, top_k=top_k_features
    )

    feature_row, comparison, start, end, window_stats = choose_feature_and_window(
        candidates,
        default_actual,
        default_pred,
        calibrated_actual,
        calibrated_pred,
        window_size=window_size,
    )
    if run_info["dataset"].replace(".csv", "") == "Weather":
        start, end = choose_weather_combined_window(comparison, window_size)
        weather_window = comparison.iloc[start:end]
        window_stats = {
            "window_improvement_sum": float(weather_window["improvement"].sum()),
            "window_peakiness": float(compute_peakiness(weather_window["actual"])),
        }
    outputs = save_outputs(
        output_dir,
        output_prefix,
        comparison,
        feature_row,
        run_info,
        start,
        end,
        window_stats,
    )
    window_df = comparison.iloc[start:end].reset_index(drop=True)

    return {
        "dataset": run_info["dataset"].replace(".csv", ""),
        "feature_name": str(feature_row["feature_name"]),
        "wape_default": float(feature_row["wape_default"]),
        "wape_calibrated": float(feature_row["wape_calibrated"]),
        "delta_wape": float(feature_row["delta_wape"]),
        "window_start": int(start),
        "window_end": int(end),
        "figure": str(outputs["figure"]),
        "window_csv": str(outputs["window_csv"]),
        "summary_json": str(outputs["summary_json"]),
        "window_df": window_df,
        "comparison": comparison,
        "feature_row": feature_row.to_dict(),
        "run_info": run_info,
        "window_size": int(window_size),
    }


def main() -> None:
    args = parse_args()
    root = Path.cwd()

    if args.batch_mode:
        batch_output_dir = Path(args.batch_output_dir)
        batch_output_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        skipped = []
        for entry in build_batch_entries(root):
            dataset = entry["dataset"]
            try:
                result = run_single_plot(
                    default_archive=Path(entry["default_archive"]),
                    calibrated_archive=Path(entry["calibrated_archive"]),
                    default_metrics_path=Path(entry["default_feature_metrics"]),
                    calibrated_metrics_path=Path(entry["calibrated_feature_metrics"]),
                    output_dir=batch_output_dir / dataset,
                    output_prefix=f"{dataset.lower()}_rope_eye_test",
                    top_k_features=args.top_k_features,
                    window_size=args.window_size,
                    extra_run_info={
                        "resolved_folder": entry["folder"],
                        "resolved_default_suffix": entry["default_suffix"],
                        "resolved_calibrated_suffix": entry["calibrated_suffix"],
                        "reported_best_wape": entry.get("best_wape"),
                        "reported_default_wape": entry.get("default_wape_reported"),
                    },
                )
            except Exception as exc:
                skipped.append(
                    {
                        "dataset": dataset,
                        "folder": entry["folder"],
                        "default_archive": entry["default_archive"],
                        "calibrated_archive": entry["calibrated_archive"],
                        "reason": str(exc),
                    }
                )
                print(f"Skipped {dataset}: {exc}")
                continue
            rows.append(result)
            print(
                f"{dataset}: feature={result['feature_name']} "
                f"delta_wape={result['delta_wape']:.3f} "
                f"window=[{result['window_start']}, {result['window_end']})"
            )
            print(f"Saved figure: {result['figure']}")

        index_path = batch_output_dir / "index.csv"
        pd.DataFrame(
            [
                {
                    "dataset": row["dataset"],
                    "feature_name": row["feature_name"],
                    "wape_default": row["wape_default"],
                    "wape_calibrated": row["wape_calibrated"],
                    "delta_wape": row["delta_wape"],
                    "window_start": row["window_start"],
                    "window_end": row["window_end"],
                    "figure": row["figure"],
                    "window_csv": row["window_csv"],
                    "summary_json": row["summary_json"],
                }
                for row in rows
            ]
        ).to_csv(index_path, index=False)
        print(f"Saved batch index: {index_path}")
        if skipped:
            skipped_path = batch_output_dir / "skipped.csv"
            pd.DataFrame(skipped).to_csv(skipped_path, index=False)
            print(f"Saved skipped report: {skipped_path}")
        combined_outputs = save_combined_eye_test_figure(batch_output_dir, rows, skipped)
        print(f"Saved combined figure: {combined_outputs['combined_figure']}")
        print(f"Saved combined summary: {combined_outputs['combined_summary']}")
        k_sweep_path = save_etth1_k_sweep_figure(root)
        print(f"Saved ETTh1 k sweep figure: {k_sweep_path}")
        return

    result = run_single_plot(
        default_archive=Path(args.default_archive),
        calibrated_archive=Path(args.calibrated_archive),
        default_metrics_path=Path(args.default_feature_metrics),
        calibrated_metrics_path=Path(args.calibrated_feature_metrics),
        output_dir=Path(args.output_dir),
        output_prefix=args.output_prefix,
        top_k_features=args.top_k_features,
        window_size=args.window_size,
    )
    print("Selected feature:", result["feature_name"])
    print(
        "Feature WAPE delta:",
        f"{result['wape_default']:.3f} -> {result['wape_calibrated']:.3f}",
        f"(delta={result['delta_wape']:.3f})",
    )
    print("Selected window:", f"[{result['window_start']}, {result['window_end']})")
    print("Saved figure:", result["figure"])
    print("Saved window CSV:", result["window_csv"])
    print("Saved summary JSON:", result["summary_json"])


if __name__ == "__main__":
    main()
