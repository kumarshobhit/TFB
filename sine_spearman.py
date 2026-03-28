#!/usr/bin/env python3
import argparse
import csv
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple


def _safe_float(value: str) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _rankdata(values: Sequence[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def _pearson(x: Sequence[float], y: Sequence[float]) -> float:
    n = len(x)
    if n < 2:
        return float("nan")
    mx = sum(x) / n
    my = sum(y) / n
    vx = sum((v - mx) ** 2 for v in x)
    vy = sum((v - my) ** 2 for v in y)
    if vx == 0.0 or vy == 0.0:
        return float("nan")
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    return cov / math.sqrt(vx * vy)


def spearman_rho(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y):
        raise ValueError("x and y must have same length")
    if len(x) < 2:
        return float("nan")
    return _pearson(_rankdata(x), _rankdata(y))


def _extract_wape(report_path: str) -> Optional[float]:
    with open(report_path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = set(reader.fieldnames or [])
        if "metric_name" not in fieldnames:
            return None

        fixed_cols = {"strategy_args", "metric_name"}
        score_cols = [c for c in (reader.fieldnames or []) if c not in fixed_cols]
        if not score_cols:
            return None
        score_col = score_cols[0]

        for row in reader:
            metric = str(row.get("metric_name", "")).strip().lower()
            if metric == "wape":
                return _safe_float(row.get(score_col, ""))
    return None


def _mean_cosine(feature_path: str) -> Optional[float]:
    values: List[float] = []
    with open(feature_path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if "cosine_sim" not in set(reader.fieldnames or []):
            return None
        for row in reader:
            cosine = _safe_float(row.get("cosine_sim", ""))
            if cosine is not None:
                values.append(cosine)
    if not values:
        return None
    return sum(values) / len(values)


def _suffixes_in_dir(folder: str) -> List[str]:
    suffixes = set()
    for filename in os.listdir(folder):
        if filename.startswith("test_report_sinespe_") and filename.endswith(".csv"):
            suffixes.add(filename[len("test_report_sinespe_") : -len(".csv")])
    return sorted(suffixes)


def collect_dataset_rows(folder: str, dataset_name: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for suffix in _suffixes_in_dir(folder):
        report_path = os.path.join(folder, f"test_report_sinespe_{suffix}.csv")
        feature_path = os.path.join(folder, f"TST_per_feature_metrics_sinespe_{suffix}.csv")
        if not os.path.isfile(report_path) or not os.path.isfile(feature_path):
            print(f"[skip] missing pair for suffix={suffix}")
            continue

        wape = _extract_wape(report_path)
        mean_cosine = _mean_cosine(feature_path)
        if wape is None or mean_cosine is None:
            print(f"[skip] invalid data for suffix={suffix}")
            continue

        base_freq = _safe_float(suffix)
        rows.append(
            {
                "dataset": dataset_name,
                "run_tag": suffix,
                "base_freq": base_freq,
                "mean_cosine_sim": mean_cosine,
                "wape": wape,
            }
        )
    rows.sort(
        key=lambda r: (
            r["base_freq"] if r["base_freq"] is not None else float("inf"),
            str(r["run_tag"]),
        )
    )
    return rows


def build_pairwise_delta_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    delta_rows: List[Dict[str, object]] = []
    sorted_rows = sorted(
        rows,
        key=lambda r: (
            r["base_freq"] if r["base_freq"] is not None else float("inf"),
            str(r["run_tag"]),
        ),
    )
    for i in range(len(sorted_rows)):
        for j in range(i + 1, len(sorted_rows)):
            left = sorted_rows[i]
            right = sorted_rows[j]
            delta_rows.append(
                {
                    "dataset": left["dataset"],
                    "run_tag_left": left["run_tag"],
                    "run_tag_right": right["run_tag"],
                    "delta_mean_cosine_sim": float(right["mean_cosine_sim"]) - float(left["mean_cosine_sim"]),
                    "delta_wape": float(right["wape"]) - float(left["wape"]),
                }
            )
    return delta_rows


def write_rows(path: str, rows: List[Dict[str, object]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["dataset", "run_tag", "base_freq", "mean_cosine_sim", "wape"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_delta_rows(path: str, rows: List[Dict[str, object]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "dataset",
                "run_tag_left",
                "run_tag_right",
                "delta_mean_cosine_sim",
                "delta_wape",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dataset-level Spearman for SineSPE runs: mean cosine_sim vs WAPE."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="result/ETTh1_10thmar/TST_sine",
        help="Folder containing test_report_sinespe_*.csv and TST_per_feature_metrics_sinespe_*.csv",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ETTh1",
        help="Dataset name used in printed summary/output rows.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="result/sine_spearman_10thmar_dataset_level.csv",
        help="CSV path for per-run dataset-level points.",
    )
    parser.add_argument(
        "--output_delta_csv",
        type=str,
        default="result/sine_spearman_10thmar_dataset_level_delta.csv",
        help="CSV path for pairwise delta dataset-level points.",
    )
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    if not os.path.isdir(input_dir):
        raise ValueError(f"input_dir does not exist or is not a directory: {input_dir}")

    rows = collect_dataset_rows(input_dir, args.dataset)
    if len(rows) < 2:
        raise ValueError("Need at least 2 valid runs to compute Spearman correlation.")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    write_rows(args.output_csv, rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_delta_csv)), exist_ok=True)

    x = [float(r["mean_cosine_sim"]) for r in rows]
    y = [float(r["wape"]) for r in rows]
    rho_abs = spearman_rho(x, y)

    delta_rows = build_pairwise_delta_rows(rows)
    write_delta_rows(args.output_delta_csv, delta_rows)
    dx = [float(r["delta_mean_cosine_sim"]) for r in delta_rows]
    dy = [float(r["delta_wape"]) for r in delta_rows]
    rho_delta = spearman_rho(dx, dy) if len(delta_rows) >= 2 else float("nan")

    print(f"Dataset: {args.dataset}")
    print(f"Input dir: {input_dir}")
    print(f"Runs used: {len(rows)}")
    print(
        f"Dataset-level Spearman(mean_cosine_sim, wape): {rho_abs:.6f}"
        if not math.isnan(rho_abs)
        else "Dataset-level Spearman(mean_cosine_sim, wape): nan"
    )
    print(
        f"Dataset-level Delta Spearman(delta_mean_cosine_sim, delta_wape): {rho_delta:.6f}"
        if not math.isnan(rho_delta)
        else "Dataset-level Delta Spearman(delta_mean_cosine_sim, delta_wape): nan"
    )
    print(f"Saved per-run CSV: {os.path.abspath(args.output_csv)}")
    print(f"Saved pairwise delta CSV: {os.path.abspath(args.output_delta_csv)}")
    print("")
    print(f"{'run_tag':<12} {'base_freq':>12} {'mean_cosine':>14} {'wape':>12}")
    print("-" * 54)
    for row in rows:
        base_freq = row["base_freq"]
        base_txt = "nan" if base_freq is None else f"{float(base_freq):.6g}"
        print(
            f"{str(row['run_tag']):<12} {base_txt:>12} "
            f"{float(row['mean_cosine_sim']):>14.6f} {float(row['wape']):>12.6f}"
        )


if __name__ == "__main__":
    main()
