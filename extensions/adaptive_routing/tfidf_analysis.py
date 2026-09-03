"""Compute TF-IDF evidence-dispersion features for evaluation records."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from tqdm.auto import tqdm


FEATURE_NAMES = (
    "tfidf_n_sentences",
    "tfidf_concentration",
    "tfidf_effective_evidence_count",
    "tfidf_no_overlap",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute TF-IDF evidence-dispersion features."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(
            tqdm(file, desc="Loading examples", unit="records"),
            start=1,
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
                raise TypeError(
                    f"Expected a JSON object at line {line_number} in {path}."
                )
            missing_fields = {"question", "temporal_context"}.difference(record)
            if missing_fields:
                missing = ", ".join(sorted(missing_fields))
                raise KeyError(f"Line {line_number} is missing fields: {missing}")
            records.append(record)
    return records


def split_context_sentences(text: Any) -> list[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", str(text))
        if sentence.strip()
    ]


def build_corpus(
    records: list[dict[str, Any]],
) -> tuple[list[str], list[str], list[list[str]]]:
    questions = []
    context_sentences_by_example = []

    for record in tqdm(records, desc="Preparing TF-IDF corpus", unit="examples"):
        questions.append(str(record["question"]))
        context_sentences_by_example.append(
            split_context_sentences(record["temporal_context"])
        )

    all_context_sentences = [
        sentence
        for context_sentences in context_sentences_by_example
        for sentence in context_sentences
    ]
    corpus = questions + all_context_sentences
    return corpus, questions, context_sentences_by_example


def create_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )


def compute_features(
    vectorizer: TfidfVectorizer,
    question: str,
    context_sentences: list[str],
) -> dict[str, int | float]:
    sentence_count = len(context_sentences)
    if sentence_count == 0:
        return {
            "tfidf_n_sentences": 0,
            "tfidf_concentration": 0.0,
            "tfidf_effective_evidence_count": 0.0,
            "tfidf_no_overlap": 1,
        }

    vectors = vectorizer.transform([question, *context_sentences])
    similarities = linear_kernel(vectors[0:1], vectors[1:]).ravel()
    total_similarity = float(np.sum(similarities, dtype=np.float64))

    if total_similarity <= 1e-12:
        return {
            "tfidf_n_sentences": sentence_count,
            "tfidf_concentration": 0.0,
            "tfidf_effective_evidence_count": 0.0,
            "tfidf_no_overlap": 1,
        }

    positive_similarities = similarities[similarities > 0].astype(
        np.float64,
        copy=False,
    )
    probabilities = positive_similarities / total_similarity
    entropy = -float(np.sum(probabilities * np.log(probabilities)))

    return {
        "tfidf_n_sentences": sentence_count,
        "tfidf_concentration": float(np.max(similarities) / total_similarity),
        "tfidf_effective_evidence_count": float(np.exp(entropy)),
        "tfidf_no_overlap": 0,
    }


def write_output_jsonl(
    path: Path,
    records: list[dict[str, Any]],
    feature_rows: list[dict[str, int | float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record, features in tqdm(
            zip(records, feature_rows),
            total=len(records),
            desc="Writing TF-IDF features",
            unit="records",
        ):
            output_record = dict(record)
            output_record.update(features)
            file.write(json.dumps(output_record, ensure_ascii=False) + "\n")


def write_summary_csv(
    path: Path,
    feature_rows: list[dict[str, int | float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    feature_frame = pd.DataFrame(feature_rows, columns=FEATURE_NAMES)
    summary = feature_frame.describe().transpose()
    summary.index.name = "feature"
    summary.to_csv(path)


def main() -> None:
    args = parse_args()
    records = load_jsonl(args.input)
    corpus, questions, context_sentences_by_example = build_corpus(records)

    vectorizer = create_vectorizer()
    vectorizer.fit(corpus)

    feature_rows = []
    for question, context_sentences in tqdm(
        zip(questions, context_sentences_by_example),
        total=len(records),
        desc="Computing TF-IDF features",
        unit="examples",
    ):
        feature_rows.append(
            compute_features(vectorizer, question, context_sentences)
        )

    write_output_jsonl(args.output, records, feature_rows)
    if args.summary_output is not None:
        write_summary_csv(args.summary_output, feature_rows)

    no_overlap_count = sum(
        int(features["tfidf_no_overlap"])
        for features in feature_rows
    )
    no_overlap_percentage = (
        100.0 * no_overlap_count / len(records)
        if records
        else 0.0
    )

    print(f"Loaded examples: {len(records)}")
    print(f"Total TF-IDF documents: {len(corpus)}")
    print(f"TF-IDF vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"Output path: {args.output}")
    print(
        f"No-overlap examples: {no_overlap_count} "
        f"({no_overlap_percentage:.2f}%)"
    )


if __name__ == "__main__":
    main()
