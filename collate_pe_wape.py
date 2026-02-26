#!/usr/bin/env python3
import argparse
import csv
import os
import re
from typing import Dict, List, Optional, Tuple


REPORT_RE = re.compile(r"^test_report_(rope|sinespe)_(.+)\.csv$")


def _parse_filename(filename: str) -> Optional[Tuple[str, str, float]]:
    m = REPORT_RE.match(filename)
    if not m:
        return None
    pe_type, pe_value_raw = m.group(1), m.group(2)
    try:
        pe_value_num = float(pe_value_raw)
    except ValueError:
        pe_value_num = float("nan")
    return pe_type, pe_value_raw, pe_value_num


def _safe_float(value: str) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_wape(report_path: str) -> Optional[float]:
    with open(report_path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        metric_col = "metric_name"
        if metric_col not in fieldnames:
            return None

        fixed_cols = {"strategy_args", "metric_name"}
        score_cols = [c for c in fieldnames if c not in fixed_cols]
        if not score_cols:
            return None
        score_col = score_cols[0]

        for row in reader:
            metric = str(row.get(metric_col, "")).strip().lower()
            if metric == "wape":
                return _safe_float(row.get(score_col, ""))
    return None


def _dataset_group(input_dir: str, report_path: str) -> str:
    rel = os.path.relpath(report_path, input_dir)
    parts = rel.split(os.sep)
    if len(parts) <= 1:
        return os.path.basename(os.path.normpath(input_dir))
    return parts[0]


def collate_reports(input_dir: str) -> List[Dict]:
    rows: List[Dict] = []

    for root, _, files in os.walk(input_dir):
        for filename in files:
            parsed = _parse_filename(filename)
            if parsed is None:
                continue

            pe_type, pe_value_raw, pe_value_num = parsed
            report_path = os.path.join(root, filename)
            wape = _extract_wape(report_path)
            if wape is None:
                print(f"[skip] no valid wape row: {report_path}")
                continue

            rows.append(
                {
                    "pe_type": pe_type,
                    "pe_value_num": pe_value_num,
                    "wape": wape,
                }
            )

    rows.sort(
        key=lambda r: (
            r["wape"],
            r["pe_type"],
            r["pe_value_num"] if r["pe_value_num"] == r["pe_value_num"] else float("inf"),
        )
    )
    return rows


def write_csv(rows: List[Dict], output_csv: str) -> None:
    columns = [
        "pe_type",
        "pe_value_num",
        "wape",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_table(rows: List[Dict]) -> None:
    if not rows:
        print("No matching rows to display.")
        return

    header = ("pe_type", "pe_value_num", "wape")
    print(f"{header[0]:<10} {header[1]:>14} {header[2]:>14}")
    print("-" * 40)
    for r in rows:
        pe_type = str(r["pe_type"])
        pe_val = r["pe_value_num"]
        wape = r["wape"]
        pe_val_txt = f"{pe_val:.6g}" if pe_val == pe_val else "nan"
        wape_txt = f"{wape:.6f}"
        print(f"{pe_type:<10} {pe_val_txt:>14} {wape_txt:>14}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collate new-style test report files into PE-vs-WAPE CSV."
    )
    parser.add_argument("--input_dir", type=str, required=True, help="Folder to scan recursively")
    parser.add_argument(
        "--output_name",
        type=str,
        default="pe_wape_collated.csv",
        help="Output CSV filename (saved inside input_dir)",
    )
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    if not os.path.isdir(input_dir):
        raise ValueError(f"input_dir does not exist or is not a directory: {input_dir}")

    rows = collate_reports(input_dir)
    output_csv = os.path.join(input_dir, args.output_name)
    write_csv(rows, output_csv)

    print(f"Scanned folder: {input_dir}")
    print(f"Matched reports: {len(rows)}")
    print(f"Saved CSV: {output_csv}")
    print_table(rows)


if __name__ == "__main__":
    main()
