#!/usr/bin/env python3
import argparse
import csv
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass
class FeaturePoint:
    dataset: str
    feature_name: str
    base_freq: float
    wape: float
    cosine_sim: float


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
    rx = _rankdata(x)
    ry = _rankdata(y)
    return _pearson(rx, ry)


def permutation_p_value(
    x: Sequence[float],
    y: Sequence[float],
    observed_rho: float,
    n_perm: int,
    seed: int,
) -> float:
    if len(x) < 3 or math.isnan(observed_rho):
        return float("nan")
    rng = random.Random(seed)
    y_perm = list(y)
    extreme = 0
    abs_obs = abs(observed_rho)
    for _ in range(n_perm):
        rng.shuffle(y_perm)
        rho_perm = spearman_rho(x, y_perm)
        if not math.isnan(rho_perm) and abs(rho_perm) >= abs_obs:
            extreme += 1
    return (extreme + 1) / (n_perm + 1)


def load_feature_points(dataset: str, dataset_dir: str) -> List[FeaturePoint]:
    rows: List[FeaturePoint] = []
    for filename in os.listdir(dataset_dir):
        if not filename.startswith("TST_per_feature_metrics_rope_") or not filename.endswith(".csv"):
            continue
        base_raw = filename[len("TST_per_feature_metrics_rope_") : -len(".csv")]
        base_freq = _safe_float(base_raw)
        if base_freq is None:
            print(f"[skip] invalid base in filename: {filename}")
            continue
        path = os.path.join(dataset_dir, filename)
        with open(path, "r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            needed = {"feature_name", "wape", "cosine_sim"}
            if not needed.issubset(set(reader.fieldnames or [])):
                print(f"[skip] missing required columns in {path}")
                continue
            for row in reader:
                feature_name = str(row.get("feature_name", "")).strip()
                wape = _safe_float(row.get("wape", ""))
                cosine = _safe_float(row.get("cosine_sim", ""))
                if not feature_name or wape is None or cosine is None:
                    continue
                rows.append(
                    FeaturePoint(
                        dataset=dataset,
                        feature_name=feature_name,
                        base_freq=base_freq,
                        wape=wape,
                        cosine_sim=cosine,
                    )
                )
    rows.sort(key=lambda r: (r.dataset, r.feature_name, r.base_freq))
    return rows


def build_pairwise_delta_points(points: Iterable[FeaturePoint]) -> Dict[str, List[Tuple[float, float]]]:
    grouped: Dict[Tuple[str, str], List[FeaturePoint]] = defaultdict(list)
    for p in points:
        grouped[(p.dataset, p.feature_name)].append(p)

    out: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for (dataset, _feature), arr in grouped.items():
        arr = sorted(arr, key=lambda r: r.base_freq)
        for i in range(len(arr)):
            for j in range(i + 1, len(arr)):
                delta_cos = arr[j].cosine_sim - arr[i].cosine_sim
                delta_wape = arr[j].wape - arr[i].wape
                out[dataset].append((delta_cos, delta_wape))
                out["pooled"].append((delta_cos, delta_wape))
    return out


def build_abs_points(points: Iterable[FeaturePoint]) -> Dict[str, List[Tuple[float, float]]]:
    out: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for p in points:
        out[p.dataset].append((p.cosine_sim, p.wape))
        out["pooled"].append((p.cosine_sim, p.wape))
    return out


def compute_metric_rows(
    metric_type: str,
    by_dataset: Dict[str, List[Tuple[float, float]]],
    n_perm: int,
    seed: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for dataset in sorted(by_dataset.keys(), key=lambda d: (d == "pooled", d)):
        pairs = by_dataset[dataset]
        xs = [x for x, _ in pairs]
        ys = [y for _, y in pairs]
        rho = spearman_rho(xs, ys) if len(xs) >= 2 else float("nan")
        p_value = permutation_p_value(xs, ys, rho, n_perm=n_perm, seed=seed) if len(xs) >= 3 else float("nan")
        rows.append(
            {
                "dataset": dataset,
                "n_points": len(xs),
                "spearman_rho": rho,
                "p_value": p_value,
                "metric_type": metric_type,
            }
        )
    return rows


def write_rows(path: str, rows: List[Dict[str, object]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["dataset", "n_points", "spearman_rho", "p_value", "metric_type"],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def print_summary(title: str, rows: List[Dict[str, object]]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    ranked = sorted(
        rows,
        key=lambda r: float(r["spearman_rho"]) if not math.isnan(float(r["spearman_rho"])) else float("inf"),
    )
    for r in ranked:
        rho = float(r["spearman_rho"])
        p = float(r["p_value"])
        rho_txt = "nan" if math.isnan(rho) else f"{rho:.6f}"
        p_txt = "nan" if math.isnan(p) else f"{p:.6f}"
        print(f"{r['dataset']:<8} n={int(r['n_points']):<4} rho={rho_txt:<10} p={p_txt}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 1 RoPE Spearman analysis: absolute and pairwise delta correlations."
    )
    parser.add_argument(
        "--result_root",
        type=str,
        default="result",
        help="Root folder containing dataset result folders.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["ETTh1_22ndfeb", "ETTm1_22ndfeb", "Weather_22ndfeb"],
        help="Dataset result folders under result_root.",
    )
    parser.add_argument(
        "--subdir",
        type=str,
        default="TST_rotary",
        help="Subfolder inside each dataset folder where RoPE files are stored.",
    )
    parser.add_argument(
        "--output_abs",
        type=str,
        default="result/rope_spearman_abs.csv",
        help="Output CSV for absolute cosine vs wape Spearman.",
    )
    parser.add_argument(
        "--output_delta",
        type=str,
        default="result/rope_spearman_delta.csv",
        help="Output CSV for pairwise delta cosine vs delta wape Spearman.",
    )
    parser.add_argument(
        "--n_perm",
        type=int,
        default=5000,
        help="Number of permutations for two-sided p-value estimation.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for permutation test.")
    args = parser.parse_args()

    all_points: List[FeaturePoint] = []
    for dataset_folder in args.datasets:
        dataset_path = os.path.join(args.result_root, dataset_folder, args.subdir)
        dataset_name = dataset_folder.replace("_22ndfeb", "")
        if not os.path.isdir(dataset_path):
            print(f"[skip] dataset path not found: {dataset_path}")
            continue
        pts = load_feature_points(dataset_name, dataset_path)
        if not pts:
            print(f"[skip] no valid RoPE feature rows in {dataset_path}")
            continue
        all_points.extend(pts)

    if not all_points:
        raise ValueError("No feature points parsed. Check dataset paths and file formats.")

    abs_points = build_abs_points(all_points)
    delta_points = build_pairwise_delta_points(all_points)

    abs_rows = compute_metric_rows("abs", abs_points, n_perm=args.n_perm, seed=args.seed)
    delta_rows = compute_metric_rows("delta", delta_points, n_perm=args.n_perm, seed=args.seed)

    os.makedirs(os.path.dirname(args.output_abs) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.output_delta) or ".", exist_ok=True)
    write_rows(args.output_abs, abs_rows)
    write_rows(args.output_delta, delta_rows)

    print(f"Saved absolute Spearman CSV: {os.path.abspath(args.output_abs)}")
    print(f"Saved delta Spearman CSV:    {os.path.abspath(args.output_delta)}")
    print(f"Permutation test: n_perm={args.n_perm}, seed={args.seed}")
    print_summary("Absolute cosine_sim vs wape (Spearman)", abs_rows)
    print_summary("Pairwise delta_cos vs delta_wape (Spearman)", delta_rows)


if __name__ == "__main__":
    main()
