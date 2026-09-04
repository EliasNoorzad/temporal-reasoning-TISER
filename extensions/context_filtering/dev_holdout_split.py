"""Create a fixed development and final held-out split of JSONL records."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split evaluation records into development and final held-out sets."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--dev-output", type=Path, required=True)
    parser.add_argument("--heldout-output", type=Path, required=True)
    parser.add_argument("--dev-fraction", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--summary-output", type=Path)
    args = parser.parse_args()
    if not 0.0 < args.dev_fraction < 1.0:
        parser.error("--dev-fraction must be strictly between 0 and 1.")
    return args


def validate_paths(paths: list[Path]) -> None:
    # Prevent an output from overwriting the source or another split.
    for index, path in enumerate(paths):
        for previous in paths[:index]:
            if path.resolve() == previous.resolve() or (
                path.exists() and previous.exists() and path.samefile(previous)
            ):
                raise ValueError("Input, split outputs, and summary must use distinct files.")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at line {line_number} in {path}."
                ) from error
            if not isinstance(record, dict):
                raise TypeError(
                    f"Expected a JSON object at line {line_number} in {path}."
                )
            records.append(record)
    if len(records) < 2:
        raise ValueError("At least two input records are required for two subsets.")
    return records


def record_hash(record: dict[str, Any]) -> str:
    canonical_json = json.dumps(record, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def verify_split(
    records: list[dict[str, Any]],
    development: list[dict[str, Any]],
    heldout: list[dict[str, Any]],
) -> tuple[int, int]:
    if len(development) + len(heldout) != len(records):
        raise ValueError("Split sizes do not add up to the input size.")

    original_counts = Counter(record_hash(record) for record in records)
    development_counts = Counter(record_hash(record) for record in development)
    heldout_counts = Counter(record_hash(record) for record in heldout)

    # Compare multiplicities as well as hashes so duplicate copies cannot be lost.
    if development_counts + heldout_counts != original_counts:
        raise ValueError("Split verification failed: records were lost or changed.")

    duplicate_records = len(records) - len(original_counts)
    overlap = len(set(development_counts).intersection(heldout_counts))
    return duplicate_records, overlap


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    paths = [args.input, args.dev_output, args.heldout_output]
    if args.summary_output is not None:
        paths.append(args.summary_output)
    validate_paths(paths)
    records = load_jsonl(args.input)

    # sklearn's test portion is development here; the larger train portion stays held out.
    heldout, development = train_test_split(
        records,
        test_size=args.dev_fraction,
        random_state=args.seed,
        shuffle=True,
    )
    duplicate_records, overlap = verify_split(records, development, heldout)
    development_fraction = len(development) / len(records)
    heldout_fraction = len(heldout) / len(records)

    print(f"Input examples: {len(records)}")
    print(f"Development examples: {len(development)}")
    print(f"Final held-out examples: {len(heldout)}")
    print(f"Development percentage: {development_fraction * 100.0:.2f}%")
    print(f"Held-out percentage: {heldout_fraction * 100.0:.2f}%")
    print(f"Random seed: {args.seed}")
    print(f"Duplicate records detected in source: {duplicate_records}")
    print(f"Cross-split identical-record overlap: {overlap}")
    if duplicate_records:
        print("Duplicate count is extra copies beyond the first; none were removed.")
    if overlap:
        raise ValueError(
            "Identical source records appear in both subsets. No output files were "
            "written; the dataset was not deduplicated or regrouped."
        )

    write_jsonl(args.dev_output, development)
    write_jsonl(args.heldout_output, heldout)
    print(f"Development output path: {args.dev_output}")
    print(f"Held-out output path: {args.heldout_output}")

    if args.summary_output is not None:
        summary = {
            "input_examples": len(records),
            "development_examples": len(development),
            "heldout_examples": len(heldout),
            "development_fraction_actual": development_fraction,
            "heldout_fraction_actual": heldout_fraction,
            "seed": args.seed,
            "requested_development_fraction": args.dev_fraction,
            "duplicate_records_in_input": duplicate_records,
            "cross_split_identical_record_overlap": overlap,
            "input_path": str(args.input),
            "development_output_path": str(args.dev_output),
            "heldout_output_path": str(args.heldout_output),
            "summary_output_path": str(args.summary_output),
        }
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        with args.summary_output.open("w", encoding="utf-8") as file:
            json.dump(summary, file, ensure_ascii=False, indent=2)
            file.write("\n")
        print(f"Summary output path: {args.summary_output}")


if __name__ == "__main__":
    main()
