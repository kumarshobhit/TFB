#!/usr/bin/env python3
"""Collate reduced SinePE k-screening runs into a compact CSV summary."""
import argparse
import csv
import math
import os
import re
from statistics import mean
from typing import Dict, List, Optional, Tuple

from rope_base import compute_rope_base_list


def _safe_float(value: str) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_key_value(items: List[str], flag_name: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"{flag_name} must use NAME=VALUE, got: {item}")
        key, value = item.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def _extract_base_from_name(filename: str, default_base: float) -> Optional[float]:
    if filename == "test_report_sinespe_default.csv":
        return default_base
    match = re.match(r"^test_report_sinespe_([0-9eE+\-.]+?)(?:_\d+)?\.csv$", filename)
    if not match:
        return None
    return _safe_float(match.group(1))


def _extract_wape(report_path: str) -> Optional[float]:
    with open(report_path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        score_cols = [c for c in (reader.fieldnames or []) if c not in {"strategy_args", "metric_name"}]
        if not score_cols:
            return None
        score_col = score_cols[0]
        for row in reader:
            if str(row.get("metric_name", "")).strip().lower() == "wape":
                return _safe_float(row.get(score_col, ""))
    return None


def _find_matching_base(observed: float, expected: Dict[int, float], rtol: float) -> Optional[int]:
    best_k = None
    best_diff = math.inf
    for k, target in expected.items():
        diff = abs(observed - target)
        tol = max(abs(target) * rtol, 1e-12)
        if diff <= tol and diff < best_diff:
            best_diff = diff
            best_k = k
    return best_k


def _collect_rows(
    dataset: str,
    folder: str,
    dom_freq: float,
    default_base: float,
    d_model: int,
    n_heads: int,
    rtol: float,
) -> List[Dict[str, object]]:
    expected = {
        item["k"]: item["base_k"]
        for item in compute_rope_base_list(dom_freq, d_model=d_model, n_heads=n_heads)
        if item["k"] in {1, 2, 4, 7}
    }
    rows: List[Dict[str, object]] = []
    if not os.path.isdir(folder):
        return rows

    for fn in sorted(os.listdir(folder)):
        base = _extract_base_from_name(fn, default_base)
        if base is None:
            continue
        report_path = os.path.join(folder, fn)
        wape = _extract_wape(report_path)
        if wape is None:
            continue
        if abs(base - default_base) <= max(abs(default_base) * rtol, 1e-12):
            k_label = "default"
            k_value = ""
        else:
            matched_k = _find_matching_base(base, expected, rtol)
            if matched_k is None:
                continue
            k_label = f"k={matched_k}"
            k_value = matched_k
        rows.append(
            {
                "dataset": dataset,
                "k_label": k_label,
                "k": k_value,
                "base_freq": base,
                "wape": wape,
                "report_file": fn,
            }
        )
    return rows


def _write_csv(path: str, fieldnames: List[str], rows: List[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collate SineSPE reduced-k run-level WAPE results.")
    parser.add_argument("--dataset", action="append", default=[], help="NAME=PATH to a SineSPE result folder.")
    parser.add_argument("--freq", action="append", default=[], help="NAME=DOM_FREQ for that dataset.")
    parser.add_argument("--default_base", type=float, default=10000.0)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--base_match_rtol", type=float, default=1e-4)
    parser.add_argument("--output_dir", default="result/sine_k_analysis")
    args = parser.parse_args()

    dataset_paths = _parse_key_value(args.dataset, "--dataset")
    freq_map_raw = _parse_key_value(args.freq, "--freq")
    freq_map = {key: float(value) for key, value in freq_map_raw.items()}

    missing_freq = sorted(set(dataset_paths) - set(freq_map))
    if missing_freq:
        raise ValueError(f"Missing --freq entries for datasets: {', '.join(missing_freq)}")

    runlevel_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    for dataset, folder in dataset_paths.items():
        rows = _collect_rows(
            dataset=dataset,
            folder=folder,
            dom_freq=freq_map[dataset],
            default_base=args.default_base,
            d_model=args.d_model,
            n_heads=args.n_heads,
            rtol=args.base_match_rtol,
        )
        rows.sort(key=lambda row: (row["k_label"] != "default", row["k"] if row["k"] != "" else -1))
        runlevel_rows.extend(rows)

        grouped: Dict[str, List[float]] = {}
        for row in rows:
            grouped.setdefault(str(row["k_label"]), []).append(float(row["wape"]))
        for k_label, values in grouped.items():
            base_freq = next(float(row["base_freq"]) for row in rows if row["k_label"] == k_label)
            k_value = next(row["k"] for row in rows if row["k_label"] == k_label)
            summary_rows.append(
                {
                    "dataset": dataset,
                    "k_label": k_label,
                    "k": k_value,
                    "base_freq": base_freq,
                    "n_runs": len(values),
                    "mean_wape": mean(values),
                }
            )

    _write_csv(
        os.path.join(args.output_dir, "runlevel.csv"),
        ["dataset", "k_label", "k", "base_freq", "wape", "report_file"],
        runlevel_rows,
    )
    _write_csv(
        os.path.join(args.output_dir, "dataset_summary.csv"),
        ["dataset", "k_label", "k", "base_freq", "n_runs", "mean_wape"],
        summary_rows,
    )


if __name__ == "__main__":
    main()
