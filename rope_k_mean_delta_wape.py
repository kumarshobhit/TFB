#!/usr/bin/env python3
import argparse
import csv
import os
from collections import Counter, defaultdict
from statistics import mean
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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


def _extract_wape(report_path: str) -> Optional[float]:
    with open(report_path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        if "metric_name" not in fieldnames:
            return None

        fixed_cols = {"strategy_args", "metric_name"}
        score_cols = [c for c in fieldnames if c not in fixed_cols]
        if not score_cols:
            return None
        score_col = score_cols[0]

        for row in reader:
            if str(row.get("metric_name", "")).strip().lower() == "wape":
                return _safe_float(row.get(score_col, ""))
    return None


def _list_base_files(folder: str, prefix: str) -> List[Tuple[float, str]]:
    out: List[Tuple[float, str]] = []
    if not os.path.isdir(folder):
        return out
    for fn in os.listdir(folder):
        if not fn.startswith(prefix) or not fn.endswith(".csv"):
            continue
        raw = fn[len(prefix) : -len(".csv")]
        val = _safe_float(raw)
        if val is None:
            print(f"[skip] unparseable base in filename: {os.path.join(folder, fn)}")
            continue
        out.append((val, os.path.join(folder, fn)))
    return out


def _find_matching_file(base_files: Iterable[Tuple[float, str]], target: float, rtol: float) -> Optional[Tuple[float, str]]:
    base_files = list(base_files)
    if not base_files:
        return None
    best_val, best_path = min(base_files, key=lambda item: abs(item[0] - target))
    tol = max(abs(target) * rtol, 1e-12)
    if abs(best_val - target) <= tol:
        return best_val, best_path
    return None


def _load_feature_metrics(path: str) -> Dict[str, Dict[str, float]]:
    rows: Dict[str, Dict[str, float]] = {}
    with open(path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        needed = {"feature_name", "wape"}
        if not needed.issubset(set(reader.fieldnames or [])):
            return rows
        for row in reader:
            feature_name = str(row.get("feature_name", "")).strip()
            wape = _safe_float(row.get("wape", ""))
            if not feature_name or wape is None:
                continue
            rows[feature_name] = {"wape": wape}
    return rows


def _map_observed_bases_to_k(
    observed_bases: Iterable[float],
    freq: float,
    d_model: int,
    n_heads: int,
    rtol: float,
) -> Dict[int, float]:
    theoretical = compute_rope_base_list(freq, d_model=d_model, n_heads=n_heads)
    observed = list(observed_bases)
    mapped: Dict[int, float] = {}
    for item in theoretical:
        k = int(item["k"])
        base_k = float(item["base_k"])
        if not observed:
            continue
        closest = min(observed, key=lambda value: abs(value - base_k))
        tol = max(abs(base_k) * rtol, 1e-12)
        if abs(closest - base_k) <= tol:
            mapped[k] = closest
    return mapped


def _write_csv(path: str, rows: List[Dict[str, object]], columns: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute per-k mean delta WAPE relative to default base=10000 in a separate analysis folder."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Dataset mapping in NAME=PATH format. Repeatable.",
    )
    parser.add_argument(
        "--freq",
        action="append",
        default=[],
        help="Dominant frequency mapping in NAME=VALUE format. Repeatable. If omitted, use fft_analysis first-column common frequency.",
    )
    parser.add_argument("--default_base", type=float, default=10000.0)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--base_match_rtol", type=float, default=1e-4)
    parser.add_argument("--output_dir", type=str, default="result/rope_k_analysis")
    args = parser.parse_args()

    if not args.dataset:
        raise ValueError("--dataset is required and repeatable.")

    dataset_paths = _parse_kv(args.dataset, str, "--dataset")
    dataset_freqs = _parse_kv(args.freq, float, "--freq") if args.freq else {}

    def _dataset_fft_filename(name: str) -> str:
        if name == "Solar":
            return "Solar_ci64_every2.csv"
        return f"{name}.csv"

    def _infer_freq_from_fft(name: str) -> float:
        fft_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fft_analysis", _dataset_fft_filename(name))
        if not os.path.isfile(fft_path):
            raise ValueError(f"Missing fft_analysis file for dataset {name}: {fft_path}")
        counter: Counter[float] = Counter()
        with open(fft_path, "r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            freq_col = "Freq 1 (Cycles/Step)"
            if freq_col not in (reader.fieldnames or []):
                raise ValueError(f"Missing '{freq_col}' column in {fft_path}")
            for row in reader:
                value = _safe_float(row.get(freq_col, ""))
                if value is not None:
                    counter[value] += 1
        if not counter:
            raise ValueError(f"No valid dominant frequencies found in {fft_path}")
        return counter.most_common(1)[0][0]

    for dataset in dataset_paths:
        if dataset not in dataset_freqs:
            dataset_freqs[dataset] = _infer_freq_from_fft(dataset)

    os.makedirs(args.output_dir, exist_ok=True)

    feature_rows: List[Dict[str, object]] = []
    dataset_rows: List[Dict[str, object]] = []
    overall_buckets: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: {"feature_deltas": [], "dataset_means": [], "datasets": []})
    runlevel_rows: List[Dict[str, object]] = []

    for dataset in sorted(dataset_paths):
        folder = os.path.abspath(str(dataset_paths[dataset]))
        if not os.path.isdir(folder):
            print(f"[skip] dataset folder not found: {folder}")
            continue

        feature_files = _list_base_files(folder, "TST_per_feature_metrics_rope_")
        report_files = _list_base_files(folder, "test_report_rope_")

        baseline_feature = _find_matching_file(feature_files, args.default_base, args.base_match_rtol)
        baseline_report = _find_matching_file(report_files, args.default_base, args.base_match_rtol)
        if baseline_feature is None or baseline_report is None:
            print(f"[skip] {dataset}: missing default base={args.default_base:g} baseline files")
            continue

        observed_feature_bases = [value for value, _ in feature_files if abs(value - args.default_base) > max(args.default_base * args.base_match_rtol, 1e-12)]
        mapped_k_to_base = _map_observed_bases_to_k(
            observed_bases=observed_feature_bases,
            freq=float(dataset_freqs[dataset]),
            d_model=args.d_model,
            n_heads=args.n_heads,
            rtol=args.base_match_rtol,
        )

        baseline_feature_rows = _load_feature_metrics(baseline_feature[1])
        baseline_wape = _extract_wape(baseline_report[1])
        if baseline_wape is None:
            print(f"[skip] {dataset}: could not parse baseline run-level WAPE")
            continue

        for k in sorted(mapped_k_to_base):
            base_freq = mapped_k_to_base[k]
            current_feature = _find_matching_file(feature_files, base_freq, args.base_match_rtol)
            current_report = _find_matching_file(report_files, base_freq, args.base_match_rtol)
            if current_feature is None or current_report is None:
                print(f"[skip] {dataset}: missing files for k={k}, base={base_freq}")
                continue

            current_feature_rows = _load_feature_metrics(current_feature[1])
            common_features = sorted(set(baseline_feature_rows) & set(current_feature_rows))
            if not common_features:
                print(f"[skip] {dataset}: no overlapping feature rows for k={k}")
                continue

            deltas: List[float] = []
            improved = 0
            for feature_name in common_features:
                wape_k = current_feature_rows[feature_name]["wape"]
                wape_10000 = baseline_feature_rows[feature_name]["wape"]
                delta_wape = wape_k - wape_10000
                deltas.append(delta_wape)
                if delta_wape < 0:
                    improved += 1
                feature_rows.append(
                    {
                        "dataset": dataset,
                        "k": k,
                        "base_freq": base_freq,
                        "feature_name": feature_name,
                        "wape_k": wape_k,
                        "wape_10000": wape_10000,
                        "delta_wape": delta_wape,
                    }
                )

            mean_delta = mean(deltas)
            pct_improved = 100.0 * improved / len(deltas)
            dataset_rows.append(
                {
                    "dataset": dataset,
                    "k": k,
                    "base_freq": base_freq,
                    "n_features": len(deltas),
                    "mean_delta_wape": mean_delta,
                    "pct_features_improved": pct_improved,
                    "summary_scope": "dataset",
                }
            )
            overall_buckets[k]["feature_deltas"].extend(deltas)
            overall_buckets[k]["dataset_means"].append(mean_delta)
            overall_buckets[k]["datasets"].append(dataset)

            current_wape = _extract_wape(current_report[1])
            if current_wape is not None:
                runlevel_rows.append(
                    {
                        "dataset": dataset,
                        "k": k,
                        "base_freq": base_freq,
                        "wape_k": current_wape,
                        "wape_10000": baseline_wape,
                        "delta_wape_runlevel": current_wape - baseline_wape,
                    }
                )

    for k in sorted(overall_buckets):
        feature_deltas = overall_buckets[k]["feature_deltas"]
        dataset_means = overall_buckets[k]["dataset_means"]
        dataset_rows.append(
            {
                "dataset": "pooled",
                "k": k,
                "base_freq": "",
                "n_features": len(feature_deltas),
                "mean_delta_wape": mean(feature_deltas),
                "pct_features_improved": "",
                "summary_scope": "overall_feature_weighted",
            }
        )
        dataset_rows.append(
            {
                "dataset": "pooled",
                "k": k,
                "base_freq": "",
                "n_features": len(dataset_means),
                "mean_delta_wape": mean(dataset_means),
                "pct_features_improved": "",
                "summary_scope": "overall_dataset_equal",
            }
        )

    _write_csv(
        os.path.join(args.output_dir, "featurelevel.csv"),
        feature_rows,
        ["dataset", "k", "base_freq", "feature_name", "wape_k", "wape_10000", "delta_wape"],
    )
    _write_csv(
        os.path.join(args.output_dir, "dataset_summary.csv"),
        dataset_rows,
        ["dataset", "k", "base_freq", "n_features", "mean_delta_wape", "pct_features_improved", "summary_scope"],
    )
    _write_csv(
        os.path.join(args.output_dir, "runlevel.csv"),
        runlevel_rows,
        ["dataset", "k", "base_freq", "wape_k", "wape_10000", "delta_wape_runlevel"],
    )

    print(f"Saved analysis folder: {os.path.abspath(args.output_dir)}")
    print(f"featurelevel rows: {len(feature_rows)}")
    print(f"dataset summary rows: {len(dataset_rows)}")
    print(f"runlevel rows: {len(runlevel_rows)}")


if __name__ == "__main__":
    main()
