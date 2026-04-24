#!/usr/bin/env python3
"""Summarize seed-2021 k-screening coverage for RoPE or SinePE runs.

This helper is used to collate which calibrated `k` values were actually run
for the screening stage and to recover their WAPE values from saved report
files.
"""
import argparse
import csv
import math
import os
import re
from typing import Dict, List, Optional, Sequence

from rope_base import compute_rope_base_list


def _safe_float(value: str) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_kv(items: Sequence[str], value_cast, label: str) -> Dict[str, object]:
    parsed: Dict[str, object] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"{label} must use NAME=VALUE format, got: {item}")
        name, raw_value = item.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"{label} has empty NAME: {item}")
        parsed[name] = value_cast(raw_value.strip())
    return parsed


def _report_pattern(encoding: str) -> re.Pattern[str]:
    if encoding == "rope":
        return re.compile(r"^test_report_rope_([0-9eE+\-.]+?)(?:_\d+)?\.csv$")
    if encoding == "sinespe":
        return re.compile(r"^test_report_sinespe_([0-9eE+\-.]+?)(?:_\d+)?\.csv$")
    raise ValueError(f"Unsupported encoding: {encoding}")


def _extract_base_from_name(filename: str, encoding: str, default_base: float) -> Optional[float]:
    if encoding == "sinespe" and filename == "test_report_sinespe_default.csv":
        return default_base
    match = _report_pattern(encoding).match(filename)
    if not match:
        return None
    return _safe_float(match.group(1))


def _extract_wape(report_path: str) -> Optional[float]:
    with open(report_path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        if "metric_name" not in fieldnames:
            return None
        score_cols = [c for c in fieldnames if c not in {"strategy_args", "metric_name"}]
        if not score_cols:
            return None
        score_col = score_cols[0]
        for row in reader:
            if str(row.get("metric_name", "")).strip().lower() == "wape":
                return _safe_float(row.get(score_col, ""))
    return None


def _list_reports(folder: str, encoding: str, default_base: float) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    if not os.path.isdir(folder):
        return out
    for fn in os.listdir(folder):
        base = _extract_base_from_name(fn, encoding, default_base)
        if base is None:
            continue
        path = os.path.join(folder, fn)
        wape = _extract_wape(path)
        if wape is None:
            continue
        out.append({"base_freq": base, "report_file": fn, "wape": wape, "path": path})
    return out


def _write_csv(path: str, fieldnames: List[str], rows: List[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize seed-2021 k screening coverage for RoPE or SinePE.")
    parser.add_argument("--encoding", choices=["rope", "sinespe"], required=True)
    parser.add_argument("--dataset", action="append", default=[], help="Dataset folder mapping in NAME=PATH format.")
    parser.add_argument("--freq", action="append", default=[], help="Dominant frequency mapping in NAME=VALUE format.")
    parser.add_argument("--default-base", type=float, default=10000.0)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--base-match-rtol", type=float, default=1e-4)
    parser.add_argument("--skip-large-k1", action="store_true")
    parser.add_argument("--large-k1-threshold", type=float, default=1e8)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    dataset_paths = _parse_kv(args.dataset, str, "--dataset")
    dataset_freqs = _parse_kv(args.freq, float, "--freq")
    if sorted(dataset_paths) != sorted(dataset_freqs):
        missing = sorted(set(dataset_paths) ^ set(dataset_freqs))
        raise ValueError(f"Dataset/frequency mismatch: {missing}")

    runlevel_rows: List[Dict[str, object]] = []
    coverage_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    for dataset in sorted(dataset_paths):
        folder = os.path.abspath(str(dataset_paths[dataset]))
        freq = float(dataset_freqs[dataset])
        theoretical = {item["k"]: item["base_k"] for item in compute_rope_base_list(freq, d_model=args.d_model, n_heads=args.n_heads) if 1 <= int(item["k"]) <= 7}
        reports = _list_reports(folder, args.encoding, args.default_base)

        coverage: Dict[str, object] = {"dataset": dataset, "encoding": args.encoding, "default_present": False}
        for k in range(1, 8):
            coverage[f"k{k}_present"] = False

        default_wape: Optional[float] = None
        best_row: Optional[Dict[str, object]] = None

        for report in reports:
            base_freq = float(report["base_freq"])
            if math.isclose(base_freq, args.default_base, rel_tol=args.base_match_rtol, abs_tol=1e-12):
                coverage["default_present"] = True
                default_wape = float(report["wape"])
                runlevel_rows.append(
                    {
                        "dataset": dataset,
                        "encoding": args.encoding,
                        "k": "",
                        "base_freq": base_freq,
                        "wape": report["wape"],
                        "report_file": report["report_file"],
                    }
                )
                continue

            matched_k: Optional[int] = None
            for k, expected_base in theoretical.items():
                tol = max(abs(expected_base) * args.base_match_rtol, 1e-12)
                if abs(base_freq - expected_base) <= tol:
                    matched_k = int(k)
                    break
            if matched_k is None:
                continue
            if args.skip_large_k1 and matched_k == 1 and base_freq >= args.large_k1_threshold:
                continue

            coverage[f"k{matched_k}_present"] = True
            row = {
                "dataset": dataset,
                "encoding": args.encoding,
                "k": matched_k,
                "base_freq": base_freq,
                "wape": report["wape"],
                "report_file": report["report_file"],
            }
            runlevel_rows.append(row)
            if best_row is None or float(report["wape"]) < float(best_row["wape"]):
                best_row = row

        coverage_rows.append(coverage)
        if best_row is not None:
            summary_rows.append(
                {
                    "dataset": dataset,
                    "encoding": args.encoding,
                    "n_k_available": sum(1 for k in range(1, 8) if coverage[f"k{k}_present"]),
                    "best_k": best_row["k"],
                    "best_base_freq": best_row["base_freq"],
                    "best_wape": best_row["wape"],
                    "default_wape": default_wape,
                }
            )

    runlevel_rows.sort(key=lambda row: (row["dataset"], row["k"] == "", row["k"] if row["k"] != "" else 999))
    _write_csv(
        os.path.join(args.output_dir, "runlevel.csv"),
        ["dataset", "encoding", "k", "base_freq", "wape", "report_file"],
        runlevel_rows,
    )
    _write_csv(
        os.path.join(args.output_dir, "coverage.csv"),
        ["dataset", "encoding", "default_present", "k1_present", "k2_present", "k3_present", "k4_present", "k5_present", "k6_present", "k7_present"],
        coverage_rows,
    )
    _write_csv(
        os.path.join(args.output_dir, "dataset_summary.csv"),
        ["dataset", "encoding", "n_k_available", "best_k", "best_base_freq", "best_wape", "default_wape"],
        summary_rows,
    )


if __name__ == "__main__":
    main()
