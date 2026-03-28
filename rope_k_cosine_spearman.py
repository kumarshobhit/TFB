#!/usr/bin/env python3
import argparse
import csv
import os
from collections import Counter, defaultdict
from statistics import mean
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from scipy.stats import spearmanr

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
        needed = {"feature_name", "wape", "cosine_sim"}
        if not needed.issubset(set(reader.fieldnames or [])):
            return rows
        for row in reader:
            feature_name = str(row.get("feature_name", "")).strip()
            wape = _safe_float(row.get("wape", ""))
            cosine = _safe_float(row.get("cosine_sim", ""))
            if not feature_name or wape is None or cosine is None:
                continue
            rows[feature_name] = {"wape": wape, "cosine_sim": cosine}
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


def _spearman(xs: List[float], ys: List[float]) -> Tuple[float, float]:
    if len(xs) < 2:
        return float("nan"), float("nan")
    rho, p_value = spearmanr(xs, ys)
    return float(rho), float(p_value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute per-k cosine-vs-delta-WAPE Spearman analysis in a separate output folder."
    )
    parser.add_argument("--dataset", action="append", default=[], help="Dataset mapping in NAME=PATH format. Repeatable.")
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
    parser.add_argument("--output_dir", type=str, default="result/rope_k_cosine_spearman")
    args = parser.parse_args()

    if not args.dataset:
        raise ValueError("--dataset is required and repeatable.")

    dataset_paths = _parse_kv(args.dataset, str, "--dataset")
    dataset_freqs = _parse_kv(args.freq, float, "--freq") if args.freq else {}
    for dataset in dataset_paths:
        if dataset not in dataset_freqs:
            dataset_freqs[dataset] = _infer_freq_from_fft(dataset)

    os.makedirs(args.output_dir, exist_ok=True)

    feature_rows: List[Dict[str, object]] = []
    by_k_dataset: Dict[int, Dict[str, List[Dict[str, object]]]] = defaultdict(lambda: defaultdict(list))

    for dataset in sorted(dataset_paths):
        folder = os.path.abspath(str(dataset_paths[dataset]))
        if not os.path.isdir(folder):
            print(f"[skip] dataset folder not found: {folder}")
            continue

        feature_files = _list_base_files(folder, "TST_per_feature_metrics_rope_")
        baseline_feature = _find_matching_file(feature_files, args.default_base, args.base_match_rtol)
        if baseline_feature is None:
            print(f"[skip] {dataset}: missing default base={args.default_base:g} feature file")
            continue

        observed_feature_bases = [value for value, _ in feature_files if abs(value - args.default_base) > max(args.default_base * args.base_match_rtol, 1e-12)]
        mapped_k_to_base = _map_observed_bases_to_k(
            observed_bases=observed_feature_bases,
            freq=float(dataset_freqs[dataset]),
            d_model=args.d_model,
            n_heads=args.n_heads,
            rtol=args.base_match_rtol,
        )

        baseline_rows = _load_feature_metrics(baseline_feature[1])
        for k in sorted(mapped_k_to_base):
            base_freq = mapped_k_to_base[k]
            current_feature = _find_matching_file(feature_files, base_freq, args.base_match_rtol)
            if current_feature is None:
                print(f"[skip] {dataset}: missing feature file for k={k}, base={base_freq}")
                continue

            current_rows = _load_feature_metrics(current_feature[1])
            common_features = sorted(set(baseline_rows) & set(current_rows))
            if not common_features:
                print(f"[skip] {dataset}: no overlapping features for k={k}")
                continue

            local_rows: List[Dict[str, object]] = []
            for feature_name in common_features:
                wape_k = current_rows[feature_name]["wape"]
                wape_10000 = baseline_rows[feature_name]["wape"]
                cos_k = current_rows[feature_name]["cosine_sim"]
                cos_10000 = baseline_rows[feature_name]["cosine_sim"]
                row = {
                    "dataset": dataset,
                    "k": k,
                    "base_freq": base_freq,
                    "feature_name": feature_name,
                    "wape_k": wape_k,
                    "wape_10000": wape_10000,
                    "delta_wape": wape_k - wape_10000,
                    "cos_k": cos_k,
                    "cos_10000": cos_10000,
                    "delta_cos": cos_k - cos_10000,
                }
                local_rows.append(row)

            mean_delta_wape = mean(float(r["delta_wape"]) for r in local_rows)
            mean_delta_cos = mean(float(r["delta_cos"]) for r in local_rows)
            for row in local_rows:
                row["delta_wape_centered"] = float(row["delta_wape"]) - mean_delta_wape
                row["delta_cos_centered"] = float(row["delta_cos"]) - mean_delta_cos
                feature_rows.append(row)
                by_k_dataset[k][dataset].append(row)

    spearman_rows: List[Dict[str, object]] = []
    abs_spearman_rows: List[Dict[str, object]] = []
    for k in sorted(by_k_dataset):
        pooled_raw_rows: List[Dict[str, object]] = []
        pooled_centered_rows: List[Dict[str, object]] = []

        for dataset in sorted(by_k_dataset[k]):
            rows = by_k_dataset[k][dataset]
            pooled_raw_rows.extend(rows)
            pooled_centered_rows.extend(rows)

            xs = [float(r["delta_cos"]) for r in rows]
            ys = [float(r["delta_wape"]) for r in rows]
            rho, p_value = _spearman(xs, ys)
            spearman_rows.append(
                {
                    "dataset": dataset,
                    "k": k,
                    "n_features": len(rows),
                    "spearman_rho": rho,
                    "p_value": p_value,
                    "analysis_scope": "dataset",
                }
            )

            abs_xs = [float(r["cos_k"]) for r in rows]
            abs_ys = [float(r["wape_k"]) for r in rows]
            abs_rho, abs_p_value = _spearman(abs_xs, abs_ys)
            abs_spearman_rows.append(
                {
                    "dataset": dataset,
                    "k": k,
                    "n_features": len(rows),
                    "spearman_rho": abs_rho,
                    "p_value": abs_p_value,
                    "analysis_scope": "dataset",
                }
            )

        xs_raw = [float(r["delta_cos"]) for r in pooled_raw_rows]
        ys_raw = [float(r["delta_wape"]) for r in pooled_raw_rows]
        rho_raw, p_raw = _spearman(xs_raw, ys_raw)
        spearman_rows.append(
            {
                "dataset": "pooled",
                "k": k,
                "n_features": len(pooled_raw_rows),
                "spearman_rho": rho_raw,
                "p_value": p_raw,
                "analysis_scope": "pooled_raw",
            }
        )

        xs_ctr = [float(r["delta_cos_centered"]) for r in pooled_centered_rows]
        ys_ctr = [float(r["delta_wape_centered"]) for r in pooled_centered_rows]
        rho_ctr, p_ctr = _spearman(xs_ctr, ys_ctr)
        spearman_rows.append(
            {
                "dataset": "pooled",
                "k": k,
                "n_features": len(pooled_centered_rows),
                "spearman_rho": rho_ctr,
                "p_value": p_ctr,
                "analysis_scope": "pooled_centered",
            }
        )

        abs_xs_raw = [float(r["cos_k"]) for r in pooled_raw_rows]
        abs_ys_raw = [float(r["wape_k"]) for r in pooled_raw_rows]
        abs_rho_raw, abs_p_raw = _spearman(abs_xs_raw, abs_ys_raw)
        abs_spearman_rows.append(
            {
                "dataset": "pooled",
                "k": k,
                "n_features": len(pooled_raw_rows),
                "spearman_rho": abs_rho_raw,
                "p_value": abs_p_raw,
                "analysis_scope": "pooled_raw",
            }
        )

        abs_xs_ctr: List[float] = []
        abs_ys_ctr: List[float] = []
        for dataset in sorted(by_k_dataset[k]):
            rows = by_k_dataset[k][dataset]
            mean_cos_k = mean(float(r["cos_k"]) for r in rows)
            mean_wape_k = mean(float(r["wape_k"]) for r in rows)
            abs_xs_ctr.extend(float(r["cos_k"]) - mean_cos_k for r in rows)
            abs_ys_ctr.extend(float(r["wape_k"]) - mean_wape_k for r in rows)
        abs_rho_ctr, abs_p_ctr = _spearman(abs_xs_ctr, abs_ys_ctr)
        abs_spearman_rows.append(
            {
                "dataset": "pooled",
                "k": k,
                "n_features": len(pooled_raw_rows),
                "spearman_rho": abs_rho_ctr,
                "p_value": abs_p_ctr,
                "analysis_scope": "pooled_centered",
            }
        )

    _write_csv(
        os.path.join(args.output_dir, "featurelevel.csv"),
        feature_rows,
        [
            "dataset",
            "k",
            "base_freq",
            "feature_name",
            "wape_k",
            "wape_10000",
            "delta_wape",
            "cos_k",
            "cos_10000",
            "delta_cos",
            "delta_wape_centered",
            "delta_cos_centered",
        ],
    )
    _write_csv(
        os.path.join(args.output_dir, "dataset_spearman.csv"),
        spearman_rows,
        ["dataset", "k", "n_features", "spearman_rho", "p_value", "analysis_scope"],
    )
    _write_csv(
        os.path.join(args.output_dir, "abs_spearman.csv"),
        abs_spearman_rows,
        ["dataset", "k", "n_features", "spearman_rho", "p_value", "analysis_scope"],
    )

    readme_path = os.path.join(args.output_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as fh:
        fh.write(
            "# RoPE k Cosine/Spearman Analysis\n\n"
            "This folder contains cosine-based Spearman analysis for RoPE k runs,\n"
            "relative to the default baseline `base_freq=10000`.\n\n"
            "## Files\n\n"
            "- `featurelevel.csv`\n"
            "  - Per `(dataset, k, feature)` row.\n"
            "  - Includes raw deltas and within-dataset centered deltas.\n\n"
            "- `dataset_spearman.csv`\n"
            "  - Delta-based Spearman using `delta_cos` vs `delta_wape`.\n"
            "  - `analysis_scope=dataset`: dataset-wise Spearman for one `k`.\n"
            "  - `analysis_scope=pooled_raw`: pooled Spearman across all datasets for one `k`.\n"
            "  - `analysis_scope=pooled_centered`: pooled Spearman after centering within each dataset for one `k`.\n\n"
            "- `abs_spearman.csv`\n"
            "  - Absolute-cosine Spearman using `cos_k` vs `wape_k`.\n"
            "  - Uses the same `analysis_scope` values as `dataset_spearman.csv`.\n\n"
            "## Interpretation\n\n"
            "- `dataset_spearman.csv`:\n"
            "  - `rho < 0`: larger cosine increase is associated with lower delta WAPE (improvement).\n"
            "  - `rho > 0`: larger cosine increase is associated with higher delta WAPE (worsening).\n"
            "- `abs_spearman.csv`:\n"
            "  - `rho < 0`: higher absolute cosine at this `k` is associated with lower WAPE.\n"
            "  - `rho > 0`: higher absolute cosine at this `k` is associated with higher WAPE.\n"
            "- `rho ~ 0`: weak or no monotonic relation.\n\n"
            "## Notes\n\n"
            "- Dataset-wise `n` is small for ETTh1 and ETTm1, so those p-values can be noisy.\n"
            "- `pooled_raw` can be influenced by dataset scale and Solar's larger feature count.\n"
            "- `pooled_centered` reduces cross-dataset bias and better reflects within-dataset association.\n"
            "- Delta-based analysis remains the more decision-relevant view; absolute analysis is descriptive only.\n"
        )

    print(f"Saved analysis folder: {os.path.abspath(args.output_dir)}")
    print(f"featurelevel rows: {len(feature_rows)}")
    print(f"dataset spearman rows: {len(spearman_rows)}")
    print(f"absolute spearman rows: {len(abs_spearman_rows)}")


if __name__ == "__main__":
    main()
