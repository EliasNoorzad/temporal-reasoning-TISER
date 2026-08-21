"""Dataset loading utilities for the TISER dataset."""

from __future__ import annotations

import argparse
import json
from typing import Any

from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer

from src.model import MODEL_NAME


DATASET_NAME = "AmazonScience/TISER"


def load_tiser_dataset() -> DatasetDict:
    """Load the official TISER dataset from Hugging Face."""
    dataset = load_dataset(DATASET_NAME)
    if not isinstance(dataset, DatasetDict):
        raise TypeError(f"Expected a DatasetDict, got {type(dataset).__name__}.")
    return dataset


def load_filtered_test_dataset(max_input_tokens: int = 2048) -> Any:
    dataset = load_tiser_dataset()
    if "test" not in dataset:
        raise KeyError("AmazonScience/TISER does not contain a test split.")

    test_dataset = dataset["test"]
    required_columns = {"question_id", "dataset_name", "question", "prompt", "answer"}
    missing_columns = required_columns.difference(test_dataset.column_names)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise KeyError(f"TISER test split is missing required columns: {missing}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Extremely long test prompts are excluded from evaluation. We keep TISER
    # prompts up to 2048 tokens, matching the context range used in our experiments.
    original_test_examples = len(test_dataset)
    valid_indices = []
    for i, example in enumerate(test_dataset):
        prompt = str(example["prompt"])
        prompt_length = len(
            tokenizer(
                prompt,
                add_special_tokens=False,
            )["input_ids"]
        )
        if prompt_length <= max_input_tokens:
            valid_indices.append(i)

    filtered_dataset = test_dataset.select(valid_indices)
    print(f"Original test examples: {original_test_examples}")
    print(f"Test examples <= {max_input_tokens} tokens: {len(filtered_dataset)}")
    print(f"Removed examples: {original_test_examples - len(filtered_dataset)}")
    return filtered_dataset


def _json_default(value: Any) -> str:
    """Make uncommon dataset values printable as JSON."""
    return str(value)


def print_dataset_summary(sample_rows: int = 3) -> None:
    """Print splits, sizes, column names, and sample rows for TISER."""
    dataset = load_tiser_dataset()

    print(f"Dataset: {DATASET_NAME}")
    print("\nAvailable splits:")
    for split_name in dataset.keys():
        print(f"- {split_name}")

    print("\nNumber of examples in each split:")
    for split_name, split_dataset in dataset.items():
        print(f"- {split_name}: {len(split_dataset)}")

    print("\nColumn names:")
    for split_name, split_dataset in dataset.items():
        columns = ", ".join(split_dataset.column_names)
        print(f"- {split_name}: {columns}")

    print(f"\nSample rows ({sample_rows} per split):")
    for split_name, split_dataset in dataset.items():
        count = min(sample_rows, len(split_dataset))
        print(f"\n[{split_name}]")
        for index in range(count):
            row = split_dataset[index]
            print(json.dumps(row, indent=2, ensure_ascii=False, default=_json_default))


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Inspect the AmazonScience/TISER dataset.")
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=3,
        choices=range(2, 4),
        metavar="{2,3}",
        help="Number of sample rows to print per split.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print_dataset_summary(sample_rows=args.sample_rows)
