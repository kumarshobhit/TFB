#!/usr/bin/env python3
import argparse
import csv
import math
import os
from dataclasses import dataclass
from statistics import median
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass
class DatasetConfig:
    dataset: str
    folder: str
    k2_base: float
    default_base: float


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
    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    return _pearson(_rankdata(x), _rankdata(y))


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
    for fn in os.listdir(folder):
        if not fn.startswith(prefix) or not fn.endswith(".csv"):
            continue
        raw = fn[len(prefix) : -len(".csv")]
        val = _safe_float(raw)
        if val is None:
            continue
        out.append((val, os.path.join(folder, fn)))
    return out


def _find_nearest_file(base_files: List[Tuple[float, str]], target: float, rtol: float = 1e-4) -> Optional[str]:
    if not base_files:
        return None
    best_val, best_path = min(base_files, key=lambda t: abs(t[0] - target))
    tol = max(rtol * abs(target), 1e-12)
    if abs(best_val - target) <= tol:
        return best_path
    return None


def _load_feature_map(path: str) -> Dict[str, Tuple[float, float]]:
    out: Dict[str, Tuple[float, float]] = {}
    with open(path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        needed = {"feature_name", "wape", "cosine_sim"}
        if not needed.issubset(set(reader.fieldnames or [])):
            return out
        for row in reader:
            feature_name = str(row.get("feature_name", "")).strip()
            wape = _safe_float(row.get("wape", ""))
            cosine = _safe_float(row.get("cosine_sim", ""))
            if not feature_name or wape is None or cosine is None:
                continue
            out[feature_name] = (wape, cosine)
    return out


def write_csv(path: str, rows: List[Dict], columns: List[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def verdict_from_deltas(deltas: Iterable[float]) -> str:
    vals = list(deltas)
    if not vals:
        return "mixed"
    pos = sum(1 for v in vals if v > 0)
    neg = sum(1 for v in vals if v < 0)
    if neg > 0 and pos == 0:
        return "k=2 better"
    if pos > 0 and neg == 0:
        return "k=2 worse"
    return "mixed"


def main() -> None:
    parser = argparse.ArgumentParser(description="Short experiment: compare k=2 base against default base=10000.")
    parser.add_argument("--result_root", type=str, default="result", help="Root result directory.")
    parser.add_argument("--default_base", type=float, default=10000.0, help="Default baseline base_freq.")
    parser.add_argument(
        "--output_runlevel",
        type=str,
        default="result/rope_k2_vs_10000_runlevel.csv",
        help="Output CSV for run-level comparison.",
    )
    parser.add_argument(
        "--output_featurelevel",
        type=str,
        default="result/rope_k2_vs_10000_featurelevel.csv",
        help="Output CSV for feature-level comparison.",
    )
    parser.add_argument(
        "--output_summary",
        type=str,
        default="result/rope_k2_vs_10000_summary.csv",
        help="Output CSV for per-dataset and pooled summary.",
    )
    args = parser.parse_args()

    cfgs = [
        DatasetConfig("ETTh1", os.path.join(args.result_root, "ETTh1_22ndfeb", "TST_rotary"), 212.60314535583788, args.default_base),
        DatasetConfig("ETTm1", os.path.join(args.result_root, "ETTm1_22ndfeb", "TST_rotary"), 54426.4052110945, args.default_base),
        DatasetConfig("Weather", os.path.join(args.result_root, "Weather_22ndfeb", "TST_rotary"), 17237.391423477242, args.default_base),
        DatasetConfig("Solar", os.path.join(args.result_root, "Solar_22ndfeb", "TST_rotary_ci64_every2"), 17237.391423477242, args.default_base),
    ]

    run_rows: List[Dict] = []
    feat_rows: List[Dict] = []
    summary_rows: List[Dict] = []
    pooled_delta_wape: List[float] = []
    pooled_delta_cos: List[float] = []

    for cfg in cfgs:
        if not os.path.isdir(cfg.folder):
            print(f"[skip] missing folder: {cfg.folder}")
            continue

        report_files = _list_base_files(cfg.folder, "test_report_rope_")
        feature_files = _list_base_files(cfg.folder, "TST_per_feature_metrics_rope_")
        rep_k2 = _find_nearest_file(report_files, cfg.k2_base)
        rep_def = _find_nearest_file(report_files, cfg.default_base)
        feat_k2 = _find_nearest_file(feature_files, cfg.k2_base)
        feat_def = _find_nearest_file(feature_files, cfg.default_base)

        if rep_k2 is None or rep_def is None or feat_k2 is None or feat_def is None:
            print(f"[skip] {cfg.dataset}: required k2/default files missing")
            continue

        wape_k2 = _extract_wape(rep_k2)
        wape_def = _extract_wape(rep_def)
        if wape_k2 is None or wape_def is None:
            print(f"[skip] {cfg.dataset}: failed to parse report wape")
            continue

        delta_run = wape_k2 - wape_def
        run_rows.append(
            {
                "dataset": cfg.dataset,
                "wape_k2": wape_k2,
                "wape_10000": wape_def,
                "delta_wape_k2_minus_default": delta_run,
                "improves": delta_run < 0.0,
            }
        )

        map_k2 = _load_feature_map(feat_k2)
        map_def = _load_feature_map(feat_def)
        common = sorted(set(map_k2.keys()) & set(map_def.keys()))
        if not common:
            print(f"[skip] {cfg.dataset}: no overlapping features")
            continue

        dw_list: List[float] = []
        dc_list: List[float] = []
        for feature in common:
            w2, c2 = map_k2[feature]
            w0, c0 = map_def[feature]
            dw = w2 - w0
            dc = c2 - c0
            dw_list.append(dw)
            dc_list.append(dc)
            pooled_delta_wape.append(dw)
            pooled_delta_cos.append(dc)
            feat_rows.append(
                {
                    "dataset": cfg.dataset,
                    "feature_name": feature,
                    "wape_k2": w2,
                    "wape_10000": w0,
                    "delta_wape": dw,
                    "cos_k2": c2,
                    "cos_10000": c0,
                    "delta_cos": dc,
                }
            )

        rho = spearman_rho(dc_list, dw_list)
        pct_improved = 100.0 * sum(1 for v in dw_list if v < 0.0) / len(dw_list)
        summary_rows.append(
            {
                "dataset": cfg.dataset,
                "n_features": len(dw_list),
                "median_delta_wape": median(dw_list),
                "mean_delta_wape": sum(dw_list) / len(dw_list),
                "pct_features_improved": pct_improved,
                "spearman_delta_cos_vs_delta_wape": rho,
            }
        )

        print(f"{cfg.dataset}: {verdict_from_deltas(dw_list)}")

    if pooled_delta_wape and pooled_delta_cos:
        summary_rows.append(
            {
                "dataset": "pooled",
                "n_features": len(pooled_delta_wape),
                "median_delta_wape": median(pooled_delta_wape),
                "mean_delta_wape": sum(pooled_delta_wape) / len(pooled_delta_wape),
                "pct_features_improved": 100.0 * sum(1 for v in pooled_delta_wape if v < 0.0) / len(pooled_delta_wape),
                "spearman_delta_cos_vs_delta_wape": spearman_rho(pooled_delta_cos, pooled_delta_wape),
            }
        )

    write_csv(
        args.output_runlevel,
        run_rows,
        ["dataset", "wape_k2", "wape_10000", "delta_wape_k2_minus_default", "improves"],
    )
    write_csv(
        args.output_featurelevel,
        feat_rows,
        ["dataset", "feature_name", "wape_k2", "wape_10000", "delta_wape", "cos_k2", "cos_10000", "delta_cos"],
    )
    write_csv(
        args.output_summary,
        summary_rows,
        [
            "dataset",
            "n_features",
            "median_delta_wape",
            "mean_delta_wape",
            "pct_features_improved",
            "spearman_delta_cos_vs_delta_wape",
        ],
    )

    print(f"Saved run-level CSV:    {os.path.abspath(args.output_runlevel)}")
    print(f"Saved feature-level CSV:{os.path.abspath(args.output_featurelevel)}")
    print(f"Saved summary CSV:      {os.path.abspath(args.output_summary)}")


if __name__ == "__main__":
    main()
