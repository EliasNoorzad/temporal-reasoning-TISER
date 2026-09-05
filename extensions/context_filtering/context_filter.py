"""Filter context sentences by TF-IDF relevance and measure context-token savings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.utils.validation import check_is_fitted
from tqdm.auto import tqdm
from transformers import AutoTokenizer


SOURCE_ROW_ID_FIELD = "source_row_id"
FILTER_OUTPUT_FIELDS = (
    "context_filter_top_k",
    "original_context_sentences",
    "filtered_context_sentences",
    "original_context_tokens",
    "filtered_context_tokens",
    "context_token_saving_pct",
    "filtered_temporal_context",
)
OUTPUT_FIELDS = (SOURCE_ROW_ID_FIELD, *FILTER_OUTPUT_FIELDS)
SUMMARY_FIELDS = (
    "original_context_sentences",
    "filtered_context_sentences",
    "original_context_tokens",
    "filtered_context_tokens",
    "context_token_saving_pct",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Keep the top-k context sentences ranked by TF-IDF similarity."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-3B-Instruct")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fit-vectorizer", action="store_true")
    mode.add_argument("--vectorizer", type=Path)
    parser.add_argument("--vectorizer-output", type=Path)
    parser.add_argument("--summary-output", type=Path)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1.")
    if args.fit_vectorizer and args.vectorizer_output is None:
        parser.error("--fit-vectorizer requires --vectorizer-output.")
    if not args.fit_vectorizer and args.vectorizer_output is not None:
        parser.error("--vectorizer-output is only allowed with --fit-vectorizer.")
    return args


def validate_paths(paths: list[Path]) -> None:
    # Outputs must not overwrite the source data or a vectorizer being loaded.
    for index, path in enumerate(paths):
        for previous in paths[:index]:
            if path.resolve() == previous.resolve() or (
                path.exists() and previous.exists() and path.samefile(previous)
            ):
                raise ValueError("Input, output, vectorizer, and summary paths must differ.")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(
            tqdm(file, desc="Loading examples", unit="records"), start=1
        ):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at line {line_number} in {path}."
                ) from error
            if not isinstance(record, dict):
                raise TypeError(f"Expected a JSON object at line {line_number}.")
            for field in ("question", "temporal_context"):
                if field not in record:
                    raise KeyError(f"Line {line_number} is missing field: {field}")
                if not isinstance(record[field], str):
                    raise TypeError(f"Line {line_number}: {field} must be a string.")
            collisions = set(FILTER_OUTPUT_FIELDS).intersection(record)
            if collisions:
                raise ValueError(
                    f"Line {line_number} already contains output fields: "
                    f"{', '.join(sorted(collisions))}. Refusing to overwrite them."
                )
            records.append(record)
    if not records:
        raise ValueError("Input JSONL contains no examples.")
    return records


def assign_source_row_ids(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    has_source_row_id = [SOURCE_ROW_ID_FIELD in record for record in records]
    if any(has_source_row_id) and not all(has_source_row_id):
        raise ValueError(
            "source_row_id must be present in every input record or omitted from all records."
        )

    prepared_records = []
    seen_ids = set()
    for row_index, record in enumerate(records):
        source_row_id = record.get(SOURCE_ROW_ID_FIELD, row_index)
        if (
            isinstance(source_row_id, bool)
            or not isinstance(source_row_id, int)
            or source_row_id < 0
        ):
            raise ValueError(
                f"Input record {row_index + 1}: source_row_id must be a "
                "non-negative integer."
            )
        if source_row_id in seen_ids:
            raise ValueError(f"Duplicate source_row_id in input: {source_row_id}")
        seen_ids.add(source_row_id)
        prepared_record = dict(record)
        prepared_record[SOURCE_ROW_ID_FIELD] = source_row_id
        prepared_records.append(prepared_record)
    return prepared_records


def split_context_sentences(text: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", text)
        if sentence.strip()
    ]


def prepare_vectorizer(
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    context_sentences_by_example: list[list[str]],
) -> TfidfVectorizer:
    if not args.fit_vectorizer:
        # Load only trusted joblib files. Held-out text is never used to refit them.
        vectorizer = joblib.load(args.vectorizer)
        if not isinstance(vectorizer, TfidfVectorizer):
            raise TypeError("--vectorizer must contain a fitted TfidfVectorizer.")
        check_is_fitted(vectorizer, attributes=["vocabulary_", "idf_"])
        return vectorizer

    corpus = [record["question"] for record in records]
    corpus.extend(
        sentence
        for sentences in context_sentences_by_example
        for sentence in sentences
    )
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )
    try:
        vectorizer.fit(corpus)
    except ValueError as error:
        raise ValueError(
            "Cannot fit TF-IDF on the current input with the required settings: "
            f"{error}"
        ) from error
    args.vectorizer_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, args.vectorizer_output)
    return vectorizer


def select_context_sentences(
    vectorizer: TfidfVectorizer,
    question: str,
    context_sentences: list[str],
    top_k: int,
) -> list[str]:
    if not context_sentences:
        return []
    vectors = vectorizer.transform([question, *context_sentences])
    similarities = linear_kernel(vectors[0:1], vectors[1:]).ravel()
    ranked_indices = sorted(
        range(len(context_sentences)),
        key=lambda index: (-float(similarities[index]), index),
    )

    # Relevance chooses the sentences; their original indices restore textual order.
    selected_indices = sorted(ranked_indices[:top_k])
    return [context_sentences[index] for index in selected_indices]


def count_context_tokens(tokenizer: Any, text: str) -> int:
    return len(
        tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
    )


def filter_context(
    record: dict[str, Any],
    context_sentences: list[str],
    vectorizer: TfidfVectorizer,
    tokenizer: Any,
    top_k: int,
) -> dict[str, Any]:
    selected_sentences = select_context_sentences(
        vectorizer, record["question"], context_sentences, top_k
    )
    filtered_context = " ".join(selected_sentences)
    original_tokens = count_context_tokens(tokenizer, record["temporal_context"])
    filtered_tokens = count_context_tokens(tokenizer, filtered_context)
    saving_pct = (
        (1.0 - filtered_tokens / original_tokens) * 100.0
        if original_tokens != 0
        else 0.0
    )
    return {
        "context_filter_top_k": top_k,
        "original_context_sentences": len(context_sentences),
        "filtered_context_sentences": len(selected_sentences),
        "original_context_tokens": original_tokens,
        "filtered_context_tokens": filtered_tokens,
        "context_token_saving_pct": saving_pct,
        "filtered_temporal_context": filtered_context,
    }


def main() -> None:
    args = parse_args()
    vectorizer_path = (
        args.vectorizer_output if args.fit_vectorizer else args.vectorizer
    )
    paths = [args.input, args.output, vectorizer_path]
    if args.summary_output is not None:
        paths.append(args.summary_output)
    validate_paths(paths)
    records = assign_source_row_ids(load_jsonl(args.input))
    context_sentences_by_example = [
        split_context_sentences(record["temporal_context"])
        for record in tqdm(records, desc="Splitting context sentences", unit="examples")
    ]
    vectorizer = prepare_vectorizer(args, records, context_sentences_by_example)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    summary_rows = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        for record, sentences in tqdm(
            zip(records, context_sentences_by_example),
            total=len(records),
            desc="Filtering contexts and counting tokens",
            unit="examples",
        ):
            fields = filter_context(record, sentences, vectorizer, tokenizer, args.top_k)
            output_record = dict(record)
            output_record.update(fields)
            file.write(json.dumps(output_record, ensure_ascii=False) + "\n")
            summary_rows.append({field: fields[field] for field in SUMMARY_FIELDS})

    statistics = pd.DataFrame(summary_rows, columns=SUMMARY_FIELDS)
    if args.summary_output is not None:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary = statistics.describe().transpose()
        summary.index.name = "feature"
        summary.to_csv(args.summary_output)

    print(f"Loaded examples: {len(records)}")
    print(f"Top-k: {args.top_k}")
    print(f"TF-IDF vectorizer: {'fitted on current input' if args.fit_vectorizer else 'loaded without refitting'}")
    print(f"TF-IDF vocabulary size: {len(vectorizer.vocabulary_)}")
    means = statistics.mean()
    print(f"Average original context sentences: {means['original_context_sentences']:.2f}")
    print(f"Average filtered context sentences: {means['filtered_context_sentences']:.2f}")
    print(f"Average original context tokens: {means['original_context_tokens']:.2f}")
    print(f"Average filtered context tokens: {means['filtered_context_tokens']:.2f}")
    print(f"Average context-token saving percentage: {means['context_token_saving_pct']:.2f}%")
    print(f"Output JSONL path: {args.output}")
    print(f"Vectorizer path: {vectorizer_path}")
    if args.summary_output is not None:
        print(f"Summary path: {args.summary_output}")


if __name__ == "__main__":
    main()
