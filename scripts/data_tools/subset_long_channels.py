#!/usr/bin/env python3
import argparse
import csv
import os
import re
from typing import List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create deterministic channel subsets for long-format datasets (date,data,cols)."
    )
    parser.add_argument("--input", required=True, help="Input CSV path (long format)")
    parser.add_argument("--output", required=True, help="Output subset CSV path")
    parser.add_argument(
        "--manifest",
        required=True,
        help="Output channel manifest txt path",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=2,
        help="Take every Nth channel after sorting (default: 2)",
    )
    parser.add_argument(
        "--target_count",
        type=int,
        default=64,
        help="Number of selected channels to keep after striding (default: 64)",
    )
    return parser.parse_args()


def channel_sort_key(name: str) -> Tuple[int, str]:
    m = re.search(r"(\d+)$", name)
    if m:
        return (0, int(m.group(1)))
    return (1, name)


def load_unique_channels(input_path: str) -> List[str]:
    channels = set()
    with open(input_path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        required = {"date", "data", "cols"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"Input must contain columns {sorted(required)}; got {reader.fieldnames}"
            )
        for row in reader:
            channels.add(row["cols"])
    return sorted(channels, key=channel_sort_key)


def subset_dataset(input_path: str, output_path: str, selected: set) -> int:
    kept_rows = 0
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(input_path, "r", newline="", encoding="utf-8") as src, open(
        output_path, "w", newline="", encoding="utf-8"
    ) as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            if row["cols"] in selected:
                writer.writerow(row)
                kept_rows += 1
    return kept_rows


def write_manifest(manifest_path: str, original_count: int, selected_ordered: List[str]) -> None:
    os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as fh:
        fh.write(f"original_channel_count: {original_count}\n")
        fh.write(f"selected_channel_count: {len(selected_ordered)}\n")
        fh.write("selection_rule: every_2_then_first_64\n")
        fh.write("selected_channels:\n")
        for ch in selected_ordered:
            fh.write(f"{ch}\n")


def main() -> None:
    args = parse_args()
    if args.step <= 0:
        raise ValueError("--step must be > 0")
    if args.target_count <= 0:
        raise ValueError("--target_count must be > 0")

    channels_sorted = load_unique_channels(args.input)
    selected_ordered = channels_sorted[:: args.step][: args.target_count]
    selected_set = set(selected_ordered)

    kept_rows = subset_dataset(args.input, args.output, selected_set)
    write_manifest(args.manifest, len(channels_sorted), selected_ordered)

    print(f"Input: {args.input}")
    print(f"Original channels: {len(channels_sorted)}")
    print(f"Selected channels: {len(selected_ordered)}")
    print(f"Rows written: {kept_rows}")
    print(f"Output CSV: {args.output}")
    print(f"Manifest: {args.manifest}")


if __name__ == "__main__":
    main()
