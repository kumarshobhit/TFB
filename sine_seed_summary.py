#!/usr/bin/env python3
import argparse
import csv
import os
import re
from collections import defaultdict
from statistics import mean, stdev
from typing import Dict, Iterable, List, Optional, Tuple

from rope_base import compute_rope_base_list


def _safe_float(value: str) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_seed_folder(items: List[str]) -> Dict[Tuple[str, int], str]:
    out: Dict[Tuple[str, int], str] = {}
    for item in items:
        if "=" not in item or ":" not in item.split("=", 1)[0]:
            raise ValueError(f"--seed-folder must use DATASET:SEED=PATH, got: {item}")
        lhs, path = item.split("=", 1)
        dataset, seed_text = lhs.split(":", 1)
        out[(dataset.strip(), int(seed_text.strip()))] = path.strip()
    return out


def _parse_labeled_bases(items: List[str], flag_name: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"{flag_name} must use LABEL=VALUE, got: {item}")
        label, value = item.split("=", 1)
        label = label.strip()
        base = _safe_float(value.strip())
        if not label or base is None:
            raise ValueError(f"{flag_name} must use LABEL=NUMERIC_VALUE, got: {item}")
        out[label] = base
    return out


def _extract_base_from_report_name(filename: str, default_base: float) -> Optional[float]:
    if filename == "test_report_sinespe_default.csv":
        return default_base
    match = re.match(r"^test_report_sinespe_([0-9eE+\-.]+?)(?:_\d+)?\.csv$", filename)
    if not match:
        return None
    return _safe_float(match.group(1))


def _list_report_files(folder: str, default_base: float) -> List[Tuple[float, str]]:
    out: List[Tuple[float, str]] = []
    if not os.path.isdir(folder):
        return out
    for fn in os.listdir(folder):
        base = _extract_base_from_report_name(fn, default_base)
        if base is None:
            continue
        out.append((base, os.path.join(folder, fn)))
    return out


def _find_matching_report(base_files: Iterable[Tuple[float, str]], target: float, rtol: float) -> Optional[str]:
    base_files = list(base_files)
    if not base_files:
        return None
    tol = max(abs(target) * rtol, 1e-12)
    matching = [(base, path) for base, path in base_files if abs(base - target) <= tol]
    if matching:
        return max(matching, key=lambda item: os.path.getmtime(item[1]))[1]
    best_base, best_path = min(base_files, key=lambda item: abs(item[0] - target))
    if abs(best_base - target) <= tol:
        return best_path
    return None


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


def _sample_std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    return stdev(values)


def _write_csv(path: str, rows: List[Dict[str, object]], columns: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_readme(path: str) -> None:
    content = """# SinePE Seed Analysis

This folder stores seeded run-level comparisons for the reduced SinePE experiment.

## Files

- `runlevel.csv`
  - one row per `(dataset, seed, base_label)`
  - `wape` is taken from `test_report_sinespe_*.csv`

- `summary.csv`
  - mean and sample standard deviation of WAPE across seeds
  - lower `mean_wape` is better

- `delta_summary.csv`
  - seed-wise comparison against default `base=10000`
  - negative delta means the named base is better than default

## Notes

- This analysis uses the reduced paper-facing base set only.
- Giant k=1-style SinePE bases are intentionally excluded from the main summary.
"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)


def _parse_dataset_frequencies(items: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--freq must use DATASET=VALUE, got: {item}")
        dataset, value = item.split("=", 1)
        freq = _safe_float(value.strip())
        if not dataset.strip() or freq is None:
            raise ValueError(f"--freq must use DATASET=NUMERIC_VALUE, got: {item}")
        out[dataset.strip()] = freq
    return out


def _map_base_to_k(base_freq: float, freq: Optional[float], d_model: int, n_heads: int, rtol: float) -> str:
    if freq is None:
        return ""
    ladder = compute_rope_base_list(freq, d_model=d_model, n_heads=n_heads)
    if not ladder:
        return ""
    tol = max(abs(base_freq) * rtol, 1e-12)
    for item in ladder:
        if abs(item["base_k"] - base_freq) <= tol:
            return str(item["k"])
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize SinePE run-level WAPE across seeds.")
    parser.add_argument("--seed-folder", action="append", default=[], help="Dataset-seed folder in DATASET:SEED=PATH format.")
    parser.add_argument("--best-base", action="append", default=[], help="Dataset best base in DATASET=VALUE format.")
    parser.add_argument("--freq", action="append", default=[], help="Dataset dominant frequency in DATASET=VALUE format.")
    parser.add_argument("--default-base", type=float, default=10000.0)
    parser.add_argument("--extra-base", action="append", default=[], help="Additional labeled control base in LABEL=VALUE format.")
    parser.add_argument("--base-match-rtol", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--output-dir", default="result/sine_seed_analysis")
    args = parser.parse_args()

    if not args.seed_folder:
        raise ValueError("At least one --seed-folder is required.")

    seed_folders = _parse_seed_folder(args.seed_folder)
    best_bases = _parse_labeled_bases(args.best_base, "--best-base")
    dataset_freqs = _parse_dataset_frequencies(args.freq)
    extra_bases = _parse_labeled_bases(args.extra_base, "--extra-base")

    os.makedirs(args.output_dir, exist_ok=True)

    runlevel_rows: List[Dict[str, object]] = []
    grouped: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    row_lookup: Dict[Tuple[str, int, str], float] = {}

    for (dataset, seed), folder in sorted(seed_folders.items()):
        if dataset not in best_bases:
            print(f"[skip] missing best-base entry for dataset {dataset}")
            continue
        base_map = {"best": best_bases[dataset], "default": args.default_base}
        base_map.update(extra_bases)
        report_files = _list_report_files(folder, args.default_base)
        if not report_files:
            print(f"[skip] no report files found in {folder}")
            continue

        for label, base_freq in base_map.items():
            report_path = _find_matching_report(report_files, base_freq, args.base_match_rtol)
            if report_path is None:
                print(f"[skip] {dataset} seed={seed}: missing report for {label} base={base_freq}")
                continue
            wape = _extract_wape(report_path)
            if wape is None:
                print(f"[skip] {dataset} seed={seed}: could not parse WAPE from {report_path}")
                continue
            runlevel_rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "base_label": label,
                    "k": _map_base_to_k(base_freq, dataset_freqs.get(dataset), args.d_model, args.n_heads, args.base_match_rtol),
                    "base_freq": base_freq,
                    "wape": wape,
                }
            )
            grouped[(dataset, label)].append(wape)
            row_lookup[(dataset, seed, label)] = wape

    summary_rows: List[Dict[str, object]] = []
    for (dataset, label), values in sorted(grouped.items()):
        base_map = {"best": best_bases[dataset], "default": args.default_base, **extra_bases}
        summary_rows.append(
            {
                "dataset": dataset,
                "base_label": label,
                "k": _map_base_to_k(base_map[label], dataset_freqs.get(dataset), args.d_model, args.n_heads, args.base_match_rtol),
                "base_freq": base_map[label],
                "n_seeds": len(values),
                "mean_wape": mean(values),
                "std_wape": _sample_std(values),
            }
        )

    delta_rows: List[Dict[str, object]] = []
    candidate_labels = ["best"] + list(extra_bases.keys())
    for dataset in sorted({dataset for dataset, _seed in seed_folders}):
        for lhs_label in candidate_labels:
            deltas: List[float] = []
            for (_dataset, seed) in sorted(seed_folders):
                if _dataset != dataset:
                    continue
                lhs = row_lookup.get((dataset, seed, lhs_label))
                rhs = row_lookup.get((dataset, seed, "default"))
                if lhs is None or rhs is None:
                    continue
                deltas.append(lhs - rhs)
            if deltas:
                delta_rows.append(
                    {
                        "dataset": dataset,
                        "comparison": f"{lhs_label}_minus_default",
                        "n_seeds": len(deltas),
                        "mean_delta_wape": mean(deltas),
                        "std_delta_wape": _sample_std(deltas),
                    }
                )

    _write_csv(
        os.path.join(args.output_dir, "runlevel.csv"),
        runlevel_rows,
        ["dataset", "seed", "base_label", "k", "base_freq", "wape"],
    )
    _write_csv(
        os.path.join(args.output_dir, "summary.csv"),
        summary_rows,
        ["dataset", "base_label", "k", "base_freq", "n_seeds", "mean_wape", "std_wape"],
    )
    _write_csv(
        os.path.join(args.output_dir, "delta_summary.csv"),
        delta_rows,
        ["dataset", "comparison", "n_seeds", "mean_delta_wape", "std_delta_wape"],
    )
    _write_readme(os.path.join(args.output_dir, "README.md"))


if __name__ == "__main__":
    main()
