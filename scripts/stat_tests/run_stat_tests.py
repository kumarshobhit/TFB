#!/usr/bin/env python3
"""Run seeded and feature-level statistical tests for RoPE and SinePE.

This script consumes the existing aggregate CSV outputs and dataset-specific
feature metrics, then writes a unified statistical summary to result/stat_tests.
It is intentionally stdlib-only so it can run in the current workspace without
extra Python package dependencies.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = REPO_ROOT / "result"
DEFAULT_OUTPUT_DIR = RESULT_ROOT / "stat_tests"
BOOTSTRAP_SAMPLES = 5000
BOOTSTRAP_SEED = 12345
EPS = 1e-12


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def parse_float(value: str) -> float:
    return float(value) if value not in ("", None) else math.nan


def format_float(value: float) -> str:
    if value is None or math.isnan(value):
        return ""
    return f"{value:.12g}"


def mean(values: Sequence[float]) -> float:
    return statistics.fmean(values)


def median(values: Sequence[float]) -> float:
    return statistics.median(values)


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    samples: int,
    rng: random.Random,
) -> Tuple[float, float]:
    if not values:
        raise ValueError("bootstrap_mean_ci requires at least one value")
    n = len(values)
    bootstrap_means = []
    for _ in range(samples):
        resample = [values[rng.randrange(n)] for _ in range(n)]
        bootstrap_means.append(mean(resample))
    bootstrap_means.sort()
    lower_idx = int(0.025 * (samples - 1))
    upper_idx = int(0.975 * (samples - 1))
    return bootstrap_means[lower_idx], bootstrap_means[upper_idx]


def sign_test_pvalues(values: Sequence[float]) -> Tuple[float, float]:
    wins = sum(value < 0 for value in values)
    losses = sum(value > 0 for value in values)
    n = wins + losses
    if n == 0:
        return 1.0, 1.0
    p_one_sided = sum(math.comb(n, i) for i in range(wins, n + 1)) / (2 ** n)
    smaller_tail = min(wins, losses)
    p_two_sided = min(
        1.0,
        2.0 * sum(math.comb(n, i) for i in range(0, smaller_tail + 1)) / (2 ** n),
    )
    return p_one_sided, p_two_sided


def sign_flip_pvalues(values: Sequence[float]) -> Tuple[float, float]:
    if not values:
        return 1.0, 1.0
    observed = mean(values)
    all_means = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        stat = mean([sign * value for sign, value in zip(signs, values)])
        all_means.append(stat)
    lower = sum(stat <= observed + EPS for stat in all_means) / len(all_means)
    upper = sum(stat >= observed - EPS for stat in all_means) / len(all_means)
    return lower, min(1.0, 2.0 * min(lower, upper))


def benjamini_hochberg(rows: List[Dict[str, object]], p_key: str, q_key: str) -> None:
    if not rows:
        return
    ordered = sorted(enumerate(rows), key=lambda item: float(item[1][p_key]))
    m = len(ordered)
    adjusted = [0.0] * m
    running = 1.0
    for rank in range(m - 1, -1, -1):
        _, row = ordered[rank]
        p_value = float(row[p_key])
        candidate = p_value * m / (rank + 1)
        running = min(running, candidate)
        adjusted[rank] = min(1.0, running)
    for (original_idx, _), q_value in zip(ordered, adjusted):
        rows[original_idx][q_key] = q_value


def decision_label(mean_delta: float, q_value: float) -> str:
    if mean_delta < 0 and q_value < 0.05:
        return "improves_vs_default"
    return "not_significant"


def pooled_decision_label(mean_delta: float, p_one_sided: float, note: str) -> str:
    if mean_delta < 0 and p_one_sided < 0.05:
        return "supports_improvement" if "confirmatory" in note else "exploratory_support"
    return "no_clear_support"


def group_rows(rows: Iterable[Dict[str, str]], *keys: str) -> Dict[Tuple[str, ...], List[Dict[str, str]]]:
    grouped: Dict[Tuple[str, ...], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    return grouped


def compute_seed_tests(
    family: str,
    runlevel_path: Path,
    delta_summary_path: Path,
    rng: random.Random,
) -> List[Dict[str, object]]:
    rows = read_csv_rows(runlevel_path)
    grouped = group_rows(rows, "dataset", "seed")
    deltas_by_dataset_comparison: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for (dataset, seed), seed_rows in grouped.items():
        by_label = {row["base_label"]: row for row in seed_rows}
        if "default" not in by_label:
            raise ValueError(f"Missing default row for {family} dataset={dataset} seed={seed}")
        default_wape = parse_float(by_label["default"]["wape"])
        for base_label, row in by_label.items():
            if base_label == "default":
                continue
            candidate_wape = parse_float(row["wape"])
            deltas_by_dataset_comparison[(dataset, f"{base_label}_minus_default")].append(
                candidate_wape - default_wape
            )

    tests: List[Dict[str, object]] = []
    for (dataset, comparison), deltas in sorted(deltas_by_dataset_comparison.items()):
        p_one_sign, p_two_sign = sign_test_pvalues(deltas)
        p_one_perm, p_two_perm = sign_flip_pvalues(deltas)
        ci_low, ci_high = bootstrap_mean_ci(deltas, samples=BOOTSTRAP_SAMPLES, rng=rng)
        wins = sum(delta < 0 for delta in deltas)
        losses = sum(delta > 0 for delta in deltas)
        tests.append(
            {
                "family": family,
                "dataset": dataset,
                "comparison": comparison,
                "n_units": len(deltas),
                "mean_delta_wape": mean(deltas),
                "median_delta_wape": median(deltas),
                "wins": wins,
                "losses": losses,
                "p_one_sided": p_one_sign,
                "p_two_sided": p_two_sign,
                "p_one_sided_perm": p_one_perm,
                "p_two_sided_perm": p_two_perm,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "notes": "confirmatory seed-level paired comparison vs default",
            }
        )

    benjamini_hochberg(tests, "p_one_sided", "q_value")
    for row in tests:
        row["decision"] = decision_label(float(row["mean_delta_wape"]), float(row["q_value"]))

    validate_seed_summary_matches(family, tests, delta_summary_path)
    return tests


def validate_seed_summary_matches(
    family: str,
    tests: Sequence[Dict[str, object]],
    delta_summary_path: Path,
) -> None:
    expected_rows = read_csv_rows(delta_summary_path)
    expected = {
        (row["dataset"], row["comparison"]): parse_float(row["mean_delta_wape"])
        for row in expected_rows
    }
    for row in tests:
        key = (str(row["dataset"]), str(row["comparison"]))
        if key not in expected:
            raise ValueError(f"{family}: missing expected delta summary row for {key}")
        actual = float(row["mean_delta_wape"])
        if abs(actual - expected[key]) > 1e-9:
            raise ValueError(
                f"{family}: mean delta mismatch for {key}: actual={actual} expected={expected[key]}"
            )


def compute_rope_k_tests(rng: random.Random) -> List[Dict[str, object]]:
    path = RESULT_ROOT / "rope_k_analysis" / "featurelevel.csv"
    rows = read_csv_rows(path)
    grouped = group_rows(rows, "dataset", "k", "base_freq")
    tests: List[Dict[str, object]] = []
    for (dataset, k_value, base_freq), group in sorted(grouped.items()):
        deltas = [parse_float(row["delta_wape"]) for row in group]
        p_one, p_two = sign_test_pvalues(deltas)
        ci_low, ci_high = bootstrap_mean_ci(deltas, samples=BOOTSTRAP_SAMPLES, rng=rng)
        wins = sum(delta < 0 for delta in deltas)
        losses = sum(delta > 0 for delta in deltas)
        tests.append(
            {
                "family": "rope",
                "dataset": dataset,
                "k": int(float(k_value)),
                "base_freq": parse_float(base_freq),
                "n_units": len(deltas),
                "mean_delta_wape": mean(deltas),
                "median_delta_wape": median(deltas),
                "wins": wins,
                "losses": losses,
                "pct_features_improved": 100.0 * wins / len(deltas),
                "p_one_sided": p_one,
                "p_two_sided": p_two,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "notes": "exploratory feature-level paired comparison vs default",
            }
        )

    benjamini_hochberg(tests, "p_one_sided", "q_value")
    for row in tests:
        row["decision"] = decision_label(float(row["mean_delta_wape"]), float(row["q_value"]))

    validate_rope_feature_counts(rows)
    validate_rope_k_example(rows)
    return tests


def validate_rope_feature_counts(rows: Sequence[Dict[str, str]]) -> None:
    expected_counts = {"ETTh1": 7, "ETTm1": 7, "Solar": 64, "Weather": 21}
    grouped = group_rows(rows, "dataset", "k")
    for dataset, expected_count in expected_counts.items():
        matching = [group for (group_dataset, _), group in grouped.items() if group_dataset == dataset]
        if not matching:
            raise ValueError(f"Missing rope k rows for dataset={dataset}")
        for group in matching:
            if len(group) != expected_count:
                raise ValueError(
                    f"Unexpected rope feature count for dataset={dataset}: "
                    f"actual={len(group)} expected={expected_count}"
                )


def validate_rope_k_example(rows: Sequence[Dict[str, str]]) -> None:
    group = [
        parse_float(row["delta_wape"])
        for row in rows
        if row["dataset"] == "ETTh1" and row["k"] == "1"
    ]
    if not group:
        raise ValueError("Missing rope k example rows for ETTh1 k=1")
    expected = -0.614704942523634
    actual = mean(group)
    if abs(actual - expected) > 1e-9:
        raise ValueError(f"Rope k example mismatch: actual={actual} expected={expected}")


def build_sine_feature_rows() -> List[Dict[str, object]]:
    summary_rows = read_csv_rows(RESULT_ROOT / "sine_k_analysis" / "dataset_summary.csv")
    run_rows = read_csv_rows(RESULT_ROOT / "sine_k_analysis" / "runlevel.csv")
    summary_lookup = {
        (row["dataset"], row["k_label"]): {
            "k": row["k"],
            "base_freq": row["base_freq"],
        }
        for row in summary_rows
    }
    run_by_dataset = group_rows(run_rows, "dataset")
    feature_rows: List[Dict[str, object]] = []
    for (dataset,), dataset_rows in sorted(run_by_dataset.items()):
        default_row = next((row for row in dataset_rows if row["k_label"] == "default"), None)
        if default_row is None:
            raise ValueError(f"Missing sine default run row for dataset={dataset}")
        default_report_path = resolve_report_path(
            dataset=dataset,
            report_file=default_row["report_file"],
            expected_wape=parse_float(default_row["wape"]),
        )
        default_feature_path = feature_metrics_path_for_report(default_report_path)
        default_features = read_feature_wape_map(default_feature_path)

        for row in dataset_rows:
            if row["k_label"] == "default":
                continue
            summary = summary_lookup.get((dataset, row["k_label"]))
            if summary is None:
                raise ValueError(f"Missing sine summary row for dataset={dataset} k_label={row['k_label']}")
            report_path = resolve_report_path(
                dataset=dataset,
                report_file=row["report_file"],
                expected_wape=parse_float(row["wape"]),
            )
            feature_path = feature_metrics_path_for_report(report_path)
            feature_map = read_feature_wape_map(feature_path)
            feature_names = sorted(default_features)
            if sorted(feature_map) != feature_names:
                raise ValueError(
                    f"Mismatched sine feature sets for dataset={dataset} report={report_path.name}"
                )
            for feature_name in feature_names:
                wape_default = default_features[feature_name]
                wape_k = feature_map[feature_name]
                feature_rows.append(
                    {
                        "dataset": dataset,
                        "k": int(float(summary["k"])),
                        "base_freq": parse_float(summary["base_freq"]),
                        "feature_name": feature_name,
                        "wape_k": wape_k,
                        "wape_10000": wape_default,
                        "delta_wape": wape_k - wape_default,
                    }
                )
    return feature_rows


def resolve_report_path(dataset: str, report_file: str, expected_wape: float) -> Path:
    matches = list(RESULT_ROOT.rglob(report_file))
    if not matches:
        raise ValueError(f"No matches found for report file {report_file}")
    scored_matches = []
    for path in matches:
        try:
            wape = read_report_wape(path)
        except Exception:
            continue
        if abs(wape - expected_wape) <= 1e-9:
            scored_matches.append(path)
    dataset_matches = [path for path in scored_matches if dataset in str(path)]
    chosen = dataset_matches if dataset_matches else scored_matches
    if len(chosen) != 1:
        raise ValueError(
            f"Ambiguous report resolution for dataset={dataset} file={report_file}: {chosen}"
        )
    return chosen[0]


def read_report_wape(path: Path) -> float:
    rows = read_csv_rows(path)
    for row in rows:
        if row["metric_name"] == "wape":
            value_column = next(key for key in row.keys() if key not in {"strategy_args", "metric_name"})
            return parse_float(row[value_column])
    raise ValueError(f"No wape row found in report {path}")


def feature_metrics_path_for_report(report_path: Path) -> Path:
    filename = report_path.name.replace("test_report_", "TST_per_feature_metrics_")
    feature_path = report_path.with_name(filename)
    if not feature_path.exists():
        raise ValueError(f"Missing feature metrics file for report {report_path}")
    return feature_path


def read_feature_wape_map(path: Path) -> Dict[str, float]:
    rows = read_csv_rows(path)
    return {row["feature_name"]: parse_float(row["wape"]) for row in rows}


def compute_sine_k_tests(rng: random.Random) -> List[Dict[str, object]]:
    rows = build_sine_feature_rows()
    grouped: Dict[Tuple[str, int, float], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset"]), int(row["k"]), float(row["base_freq"]))].append(row)

    tests: List[Dict[str, object]] = []
    for (dataset, k_value, base_freq), group in sorted(grouped.items()):
        deltas = [float(row["delta_wape"]) for row in group]
        p_one, p_two = sign_test_pvalues(deltas)
        ci_low, ci_high = bootstrap_mean_ci(deltas, samples=BOOTSTRAP_SAMPLES, rng=rng)
        wins = sum(delta < 0 for delta in deltas)
        losses = sum(delta > 0 for delta in deltas)
        tests.append(
            {
                "family": "sine",
                "dataset": dataset,
                "k": k_value,
                "base_freq": base_freq,
                "n_units": len(deltas),
                "mean_delta_wape": mean(deltas),
                "median_delta_wape": median(deltas),
                "wins": wins,
                "losses": losses,
                "pct_features_improved": 100.0 * wins / len(deltas),
                "p_one_sided": p_one,
                "p_two_sided": p_two,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "notes": "exploratory feature-level paired comparison vs default",
            }
        )

    benjamini_hochberg(tests, "p_one_sided", "q_value")
    for row in tests:
        row["decision"] = decision_label(float(row["mean_delta_wape"]), float(row["q_value"]))
    validate_sine_k_example(rows)
    return tests


def validate_sine_k_example(rows: Sequence[Dict[str, object]]) -> None:
    group = [
        float(row["delta_wape"])
        for row in rows
        if row["dataset"] == "ETTm1" and int(row["k"]) == 4
    ]
    if not group:
        raise ValueError("Missing sine k example rows for ETTm1 k=4")
    # This validates the reconstructed feature-level table, which is expected
    # to differ numerically from the run-level summary in sine_k_analysis.
    expected = -0.039693211912263236
    actual = mean(group)
    if abs(actual - expected) > 1e-9:
        raise ValueError(f"Sine k example mismatch: actual={actual} expected={expected}")


def compute_pooled_seed_tests(
    seed_tests: Sequence[Dict[str, object]],
    rng: random.Random,
) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for row in seed_tests:
        grouped[(str(row["family"]), str(row["comparison"]))].append(float(row["mean_delta_wape"]))

    pooled_rows: List[Dict[str, object]] = []
    for (family, comparison), values in sorted(grouped.items()):
        p_one, p_two = sign_flip_pvalues(values)
        ci_low, ci_high = bootstrap_mean_ci(values, samples=BOOTSTRAP_SAMPLES, rng=rng)
        pooled_rows.append(
            {
                "family": family,
                "analysis_level": "seed_dataset_equal",
                "target": comparison,
                "n_units": len(values),
                "mean_delta_wape": mean(values),
                "median_delta_wape": median(values),
                "p_one_sided": p_one,
                "p_two_sided": p_two,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "notes": "confirmatory pooled view across dataset means",
                "decision": pooled_decision_label(mean(values), p_one, "confirmatory"),
            }
        )
    return pooled_rows


def compute_pooled_k_tests(
    family: str,
    k_tests: Sequence[Dict[str, object]],
    feature_rows: Sequence[Dict[str, object]],
    rng: random.Random,
) -> List[Dict[str, object]]:
    pooled_rows: List[Dict[str, object]] = []

    feature_grouped: Dict[int, List[float]] = defaultdict(list)
    dataset_grouped: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in feature_rows:
        k_value = int(row["k"])
        delta = float(row["delta_wape"])
        feature_grouped[k_value].append(delta)
        dataset_grouped[k_value][str(row["dataset"])].append(delta)

    for k_value, deltas in sorted(feature_grouped.items()):
        p_one, p_two = sign_test_pvalues(deltas)
        ci_low, ci_high = bootstrap_mean_ci(deltas, samples=BOOTSTRAP_SAMPLES, rng=rng)
        pooled_rows.append(
            {
                "family": family,
                "analysis_level": "k_feature_weighted",
                "target": f"k={k_value}",
                "n_units": len(deltas),
                "mean_delta_wape": mean(deltas),
                "median_delta_wape": median(deltas),
                "p_one_sided": p_one,
                "p_two_sided": p_two,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "notes": "exploratory pooled view across all feature deltas",
                "decision": pooled_decision_label(mean(deltas), p_one, "exploratory"),
            }
        )

    for k_value, by_dataset in sorted(dataset_grouped.items()):
        dataset_means = [mean(deltas) for _, deltas in sorted(by_dataset.items())]
        p_one, p_two = sign_flip_pvalues(dataset_means)
        ci_low, ci_high = bootstrap_mean_ci(dataset_means, samples=BOOTSTRAP_SAMPLES, rng=rng)
        pooled_rows.append(
            {
                "family": family,
                "analysis_level": "k_dataset_equal",
                "target": f"k={k_value}",
                "n_units": len(dataset_means),
                "mean_delta_wape": mean(dataset_means),
                "median_delta_wape": median(dataset_means),
                "p_one_sided": p_one,
                "p_two_sided": p_two,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "notes": "confirmatory-style pooled view across dataset means",
                "decision": pooled_decision_label(mean(dataset_means), p_one, "confirmatory"),
            }
        )

    return pooled_rows


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            formatted = {}
            for field in fieldnames:
                value = row.get(field, "")
                if isinstance(value, float):
                    formatted[field] = format_float(value)
                else:
                    formatted[field] = value
            writer.writerow(formatted)


def write_readme(
    output_dir: Path,
    seed_tests: Sequence[Dict[str, object]],
    k_tests: Sequence[Dict[str, object]],
    pooled_tests: Sequence[Dict[str, object]],
) -> None:
    lines = [
        "# Statistical Tests",
        "",
        "This folder contains confirmatory seed-level tests and exploratory",
        "feature-level `k` tests against the default `base_freq=10000` baseline.",
        "",
        "## Files",
        "",
        "- `seed_tests.csv`: paired seed comparisons from the aggregate seed analyses.",
        "- `k_feature_tests.csv`: paired feature-level `k` comparisons for RoPE and SinePE.",
        "- `pooled_tests.csv`: pooled summaries across datasets and features.",
        "",
        "## Method",
        "",
        "- `delta_wape = wape_candidate - wape_default`.",
        "- Negative delta means the candidate improved over default.",
        "- Seed tests report exact sign-test p-values plus exact sign-flip permutation p-values.",
        "- `k` tests report exact sign-test p-values on feature-level deltas.",
        "- Confidence intervals are bootstrap intervals on the mean delta.",
        "- `q_value` uses Benjamini-Hochberg correction within each family and analysis table.",
        "",
        "## Output sizes",
        "",
        f"- seed tests: {len(seed_tests)} rows",
        f"- k feature tests: {len(k_tests)} rows",
        f"- pooled tests: {len(pooled_tests)} rows",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where statistical test outputs should be written.",
    )
    args = parser.parse_args()

    rng = random.Random(BOOTSTRAP_SEED)

    rope_seed_tests = compute_seed_tests(
        family="rope",
        runlevel_path=RESULT_ROOT / "rope_seed_analysis" / "runlevel.csv",
        delta_summary_path=RESULT_ROOT / "rope_seed_analysis" / "delta_summary.csv",
        rng=rng,
    )
    sine_seed_tests = compute_seed_tests(
        family="sine",
        runlevel_path=RESULT_ROOT / "sine_seed_analysis" / "runlevel.csv",
        delta_summary_path=RESULT_ROOT / "sine_seed_analysis" / "delta_summary.csv",
        rng=rng,
    )
    seed_tests = rope_seed_tests + sine_seed_tests

    rope_feature_rows = [
        {
            "dataset": row["dataset"],
            "k": int(float(row["k"])),
            "delta_wape": parse_float(row["delta_wape"]),
        }
        for row in read_csv_rows(RESULT_ROOT / "rope_k_analysis" / "featurelevel.csv")
    ]
    rope_k_tests = compute_rope_k_tests(rng)
    sine_feature_rows = build_sine_feature_rows()
    sine_k_tests = compute_sine_k_tests(rng)
    k_tests = rope_k_tests + sine_k_tests

    pooled_tests = []
    pooled_tests.extend(compute_pooled_seed_tests(seed_tests, rng))
    pooled_tests.extend(compute_pooled_k_tests("rope", rope_k_tests, rope_feature_rows, rng))
    pooled_tests.extend(compute_pooled_k_tests("sine", sine_k_tests, sine_feature_rows, rng))

    seed_fieldnames = [
        "family",
        "dataset",
        "comparison",
        "n_units",
        "mean_delta_wape",
        "median_delta_wape",
        "wins",
        "losses",
        "p_one_sided",
        "p_two_sided",
        "p_one_sided_perm",
        "p_two_sided_perm",
        "ci95_low",
        "ci95_high",
        "q_value",
        "decision",
        "notes",
    ]
    k_fieldnames = [
        "family",
        "dataset",
        "k",
        "base_freq",
        "n_units",
        "mean_delta_wape",
        "median_delta_wape",
        "wins",
        "losses",
        "pct_features_improved",
        "p_one_sided",
        "p_two_sided",
        "ci95_low",
        "ci95_high",
        "q_value",
        "decision",
        "notes",
    ]
    pooled_fieldnames = [
        "family",
        "analysis_level",
        "target",
        "n_units",
        "mean_delta_wape",
        "median_delta_wape",
        "p_one_sided",
        "p_two_sided",
        "ci95_low",
        "ci95_high",
        "decision",
        "notes",
    ]

    output_dir = args.output_dir
    write_csv(output_dir / "seed_tests.csv", seed_tests, seed_fieldnames)
    write_csv(output_dir / "k_feature_tests.csv", k_tests, k_fieldnames)
    write_csv(output_dir / "pooled_tests.csv", pooled_tests, pooled_fieldnames)
    write_readme(output_dir, seed_tests, k_tests, pooled_tests)

    print(f"Wrote seed tests to {output_dir / 'seed_tests.csv'}")
    print(f"Wrote k feature tests to {output_dir / 'k_feature_tests.csv'}")
    print(f"Wrote pooled tests to {output_dir / 'pooled_tests.csv'}")
    print(f"Wrote README to {output_dir / 'README.md'}")


if __name__ == "__main__":
    main()
