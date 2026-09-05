"""Evaluate offline context-filtering policies on aligned full-test results."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import combinations
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


THRESHOLDS = [3, 4, 5, 6, 7, 8, 9, 10, 12]
SOURCE_ROW_ID_FIELD = "source_row_id"
IDENTITY_FIELDS = (
    "question_id",
    "dataset_name",
    "question",
    "temporal_context",
    "gold_answer",
)
COMPLEXITY_FIELD = "tfidf_effective_evidence_count"
FULL_FIELDS = (
    "direct_em",
    "direct_f1",
    "direct_generated_tokens",
    "tiser_em",
    "tiser_f1",
    "tiser_generated_tokens",
)
FILTERED_FIELDS = (
    "filtered_direct_em",
    "filtered_direct_f1",
    "filtered_direct_generated_tokens",
    "direct_filtered_input_tokens",
    "filtered_direct_total_tokens",
    "filtered_tiser_em",
    "filtered_tiser_f1",
    "filtered_tiser_generated_tokens",
    "tiser_filtered_input_tokens",
    "filtered_tiser_total_tokens",
    "direct_full_input_tokens",
    "tiser_full_input_tokens",
)
POLICY_COLUMNS = (
    "threshold_a",
    "threshold_b",
    "threshold_c",
    "top3_examples",
    "top3_pct",
    "top5_examples",
    "top5_pct",
    "top7_examples",
    "top7_pct",
    "full_examples",
    "full_pct",
    "direct_em",
    "direct_f1",
    "direct_em_change_pp_vs_full",
    "direct_f1_change_pp_vs_full",
    "direct_avg_input_tokens",
    "direct_avg_generated_tokens",
    "direct_avg_total_tokens",
    "direct_total_tokens",
    "direct_total_token_saving_pct_vs_full",
    "tiser_em",
    "tiser_f1",
    "tiser_em_change_pp_vs_full",
    "tiser_f1_change_pp_vs_full",
    "tiser_avg_input_tokens",
    "tiser_avg_generated_tokens",
    "tiser_avg_total_tokens",
    "tiser_total_tokens",
    "tiser_total_token_saving_pct_vs_full",
)
BASELINE_COLUMNS = tuple(
    f"{branch}_full_baseline_{metric}"
    for branch in ("direct", "tiser")
    for metric in (
        "em",
        "f1",
        "avg_input_tokens",
        "avg_generated_tokens",
        "avg_total_tokens",
        "total_tokens",
    )
)
PARETO_COLUMNS = (
    "direct_em_pareto",
    "direct_f1_pareto",
    "tiser_em_pareto",
    "tiser_f1_pareto",
)
OUTPUT_COLUMNS = POLICY_COLUMNS + BASELINE_COLUMNS + PARETO_COLUMNS


@dataclass(frozen=True)
class BranchMetrics:
    em: np.ndarray
    f1: np.ndarray
    input_tokens: np.ndarray
    generated_tokens: np.ndarray
    total_tokens: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep complexity-aware context-filtering policies offline."
    )
    parser.add_argument("--complexity-input", type=Path, required=True)
    parser.add_argument("--full-input", type=Path, required=True)
    parser.add_argument("--top3-input", type=Path, required=True)
    parser.add_argument("--top5-input", type=Path, required=True)
    parser.add_argument("--top7-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pareto-output", type=Path, required=True)
    return parser.parse_args()


def load_jsonl(
    path: Path,
    name: str,
    required_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at line {line_number} in {name}: {path}"
                ) from error
            if not isinstance(record, dict):
                raise TypeError(f"Expected a JSON object at line {line_number} in {name}.")
            missing = set(required_fields).difference(record)
            if missing:
                raise KeyError(
                    f"{name} line {line_number} is missing: "
                    f"{', '.join(sorted(missing))}"
                )
            records.append(record)
    if not records:
        raise ValueError(f"{name} contains no records: {path}")
    return records


def source_row_id(record: dict[str, Any], name: str, line_number: int) -> int:
    value = record.get(SOURCE_ROW_ID_FIELD)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            f"{name} line {line_number}: source_row_id must be a non-negative integer."
        )
    return value


def map_by_source_row_id(
    records: list[dict[str, Any]],
    name: str,
) -> tuple[list[int], dict[int, dict[str, Any]]]:
    keys = []
    mapped_records = {}
    for line_number, record in enumerate(records, start=1):
        key = source_row_id(record, name, line_number)
        if key in mapped_records:
            raise ValueError(
                f"Duplicate source_row_id in {name} at line {line_number}: {key}"
            )
        keys.append(key)
        mapped_records[key] = record
    return keys, mapped_records


def canonical_value(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def validate_rowwise_complexity_alignment(
    complexity_records: list[dict[str, Any]],
    full_records: list[dict[str, Any]],
) -> None:
    if len(complexity_records) != len(full_records):
        raise ValueError(
            "Complexity input has no source_row_id and cannot be aligned by row: "
            f"it contains {len(complexity_records)} records, while the full input "
            f"contains {len(full_records)}."
        )
    for row_index, (complexity_record, full_record) in enumerate(
        zip(complexity_records, full_records, strict=True)
    ):
        for field in IDENTITY_FIELDS:
            if field not in complexity_record or field not in full_record:
                continue
            if canonical_value(complexity_record[field]) != canonical_value(
                full_record[field]
            ):
                raise ValueError(
                    "Complexity input cannot be aligned by row because "
                    f"{field} differs at zero-based row {row_index}."
                )


def prepare_complexity_records(
    complexity_records: list[dict[str, Any]],
    full_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    has_source_row_id = [
        SOURCE_ROW_ID_FIELD in record for record in complexity_records
    ]
    if any(has_source_row_id) and not all(has_source_row_id):
        raise ValueError(
            "source_row_id must be present in every complexity record or omitted "
            "from all complexity records."
        )
    if all(has_source_row_id):
        return complexity_records

    validate_rowwise_complexity_alignment(complexity_records, full_records)
    prepared_records = []
    for row_index, record in enumerate(complexity_records):
        prepared_record = dict(record)
        prepared_record[SOURCE_ROW_ID_FIELD] = row_index
        prepared_records.append(prepared_record)
    return prepared_records


def prepare_full_records(
    full_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    has_source_row_id = [SOURCE_ROW_ID_FIELD in record for record in full_records]
    if any(has_source_row_id) and not all(has_source_row_id):
        raise ValueError(
            "source_row_id must be present in every full-input record or omitted "
            "from all full-input records."
        )
    if all(has_source_row_id):
        return full_records

    prepared_records = []
    for row_index, record in enumerate(full_records):
        prepared_record = dict(record)
        prepared_record[SOURCE_ROW_ID_FIELD] = row_index
        prepared_records.append(prepared_record)
    return prepared_records


def validate_key_sets(
    reference: dict[int, dict[str, Any]],
    inputs: dict[str, dict[int, dict[str, Any]]],
) -> None:
    reference_keys = set(reference)
    for name, records in inputs.items():
        current_keys = set(records)
        missing = reference_keys - current_keys
        extra = current_keys - reference_keys
        if missing or extra:
            raise ValueError(
                f"source_row_id mismatch for {name}: {len(missing)} missing and "
                f"{len(extra)} extra IDs. Missing sample: {list(missing)[:3]}; "
                f"extra sample: {list(extra)[:3]}"
            )


def validate_identity_consistency(
    keys: list[int],
    inputs: dict[str, dict[int, dict[str, Any]]],
) -> None:
    for key in keys:
        for field in IDENTITY_FIELDS:
            available = [
                (name, records[key][field])
                for name, records in inputs.items()
                if field in records[key]
            ]
            if len(available) < 2:
                continue
            expected_name, expected_value = available[0]
            expected = canonical_value(expected_value)
            for current_name, current_value in available[1:]:
                if canonical_value(current_value) != expected:
                    raise ValueError(
                        f"Identity mismatch for source_row_id {key}: field {field} "
                        f"differs between {expected_name} and {current_name}."
                    )


def numeric_value(
    record: dict[str, Any],
    field: str,
    source_name: str,
    key: int,
) -> float:
    try:
        value = float(record[field])
    except (TypeError, ValueError) as error:
        raise ValueError(f"{source_name} field {field} is not numeric for key {key}.") from error
    if not math.isfinite(value):
        raise ValueError(f"{source_name} field {field} is not finite for key {key}.")
    if field.endswith(("_em", "_f1")) and not 0.0 <= value <= 1.0:
        raise ValueError(f"{source_name} field {field} must be on the 0-1 scale for key {key}.")
    if "tokens" in field and value < 0.0:
        raise ValueError(f"{source_name} field {field} must be nonnegative for key {key}.")
    return value


def validate_top_k(
    records: dict[int, dict[str, Any]],
    expected_top_k: int,
    source_name: str,
) -> None:
    for key, record in records.items():
        if "context_filter_top_k" in record:
            value = numeric_value(record, "context_filter_top_k", source_name, key)
            if value != expected_top_k:
                raise ValueError(
                    f"{source_name} contains context_filter_top_k={value:g} for key "
                    f"{key}; expected {expected_top_k}."
                )


def values_for(
    keys: list[int],
    records: dict[int, dict[str, Any]],
    field: str,
    source_name: str,
) -> np.ndarray:
    return np.asarray(
        [numeric_value(records[key], field, source_name, key) for key in keys],
        dtype=np.float64,
    )


def resolve_full_input_tokens(
    keys: list[int],
    full_records: dict[int, dict[str, Any]],
    filtered_records: dict[str, dict[int, dict[str, Any]]],
    branch: str,
) -> np.ndarray:
    field = f"{branch}_full_input_tokens"
    resolved = []
    for key in keys:
        filtered_values = [
            numeric_value(records[key], field, name, key)
            for name, records in filtered_records.items()
        ]
        if len(set(filtered_values)) != 1:
            detail = ", ".join(
                f"{name}={value:g}"
                for name, value in zip(filtered_records, filtered_values, strict=True)
            )
            raise ValueError(f"Inconsistent {field} values for key {key}: {detail}")
        value = filtered_values[0]
        if field in full_records[key]:
            full_value = numeric_value(full_records[key], field, "full input", key)
            if full_value != value:
                raise ValueError(
                    f"{field} in full input does not match filtered evaluations "
                    f"for key {key}: full={full_value:g}, filtered={value:g}"
                )
            value = full_value
        resolved.append(value)
    return np.asarray(resolved, dtype=np.float64)


def build_metrics(
    keys: list[int],
    full_records: dict[int, dict[str, Any]],
    filtered_records: dict[str, dict[int, dict[str, Any]]],
) -> dict[str, dict[str, BranchMetrics]]:
    metrics: dict[str, dict[str, BranchMetrics]] = {"full": {}}
    for branch in ("direct", "tiser"):
        input_tokens = resolve_full_input_tokens(
            keys, full_records, filtered_records, branch
        )
        generated_tokens = values_for(
            keys, full_records, f"{branch}_generated_tokens", "full input"
        )
        metrics["full"][branch] = BranchMetrics(
            em=values_for(keys, full_records, f"{branch}_em", "full input"),
            f1=values_for(keys, full_records, f"{branch}_f1", "full input"),
            input_tokens=input_tokens,
            generated_tokens=generated_tokens,
            total_tokens=input_tokens + generated_tokens,
        )

    for setting, records in filtered_records.items():
        metrics[setting] = {}
        for branch in ("direct", "tiser"):
            metrics[setting][branch] = BranchMetrics(
                em=values_for(keys, records, f"filtered_{branch}_em", setting),
                f1=values_for(keys, records, f"filtered_{branch}_f1", setting),
                input_tokens=values_for(
                    keys, records, f"{branch}_filtered_input_tokens", setting
                ),
                generated_tokens=values_for(
                    keys, records, f"filtered_{branch}_generated_tokens", setting
                ),
                total_tokens=values_for(
                    keys, records, f"filtered_{branch}_total_tokens", setting
                ),
            )
    return metrics


def baseline_values(metrics: dict[str, dict[str, BranchMetrics]]) -> dict[str, float]:
    values = {}
    for branch in ("direct", "tiser"):
        branch_metrics = metrics["full"][branch]
        values.update(
            {
                f"{branch}_full_baseline_em": float(branch_metrics.em.mean() * 100.0),
                f"{branch}_full_baseline_f1": float(branch_metrics.f1.mean() * 100.0),
                f"{branch}_full_baseline_avg_input_tokens": float(
                    branch_metrics.input_tokens.mean()
                ),
                f"{branch}_full_baseline_avg_generated_tokens": float(
                    branch_metrics.generated_tokens.mean()
                ),
                f"{branch}_full_baseline_avg_total_tokens": float(
                    branch_metrics.total_tokens.mean()
                ),
                f"{branch}_full_baseline_total_tokens": float(
                    branch_metrics.total_tokens.sum()
                ),
            }
        )
    return values


def selected_values(
    setting_indices: np.ndarray,
    metrics: dict[str, dict[str, BranchMetrics]],
    branch: str,
    field: str,
) -> np.ndarray:
    setting_order = ("top3", "top5", "top7", "full")
    rows = np.vstack(
        [getattr(metrics[setting][branch], field) for setting in setting_order]
    )
    return rows[setting_indices, np.arange(len(setting_indices))]


def evaluate_policies(
    complexity: np.ndarray,
    metrics: dict[str, dict[str, BranchMetrics]],
) -> pd.DataFrame:
    total_examples = len(complexity)
    baselines = baseline_values(metrics)
    rows = []
    for threshold_a, threshold_b, threshold_c in combinations(THRESHOLDS, 3):
        setting_indices = np.select(
            [
                complexity < threshold_a,
                complexity < threshold_b,
                complexity < threshold_c,
            ],
            [0, 1, 2],
            default=3,
        )
        counts = np.bincount(setting_indices, minlength=4)
        row: dict[str, Any] = {
            "threshold_a": threshold_a,
            "threshold_b": threshold_b,
            "threshold_c": threshold_c,
        }
        for setting, count in zip(("top3", "top5", "top7", "full"), counts, strict=True):
            row[f"{setting}_examples"] = int(count)
            row[f"{setting}_pct"] = float(count / total_examples * 100.0)

        for branch in ("direct", "tiser"):
            em = selected_values(setting_indices, metrics, branch, "em")
            f1 = selected_values(setting_indices, metrics, branch, "f1")
            input_tokens = selected_values(
                setting_indices, metrics, branch, "input_tokens"
            )
            generated_tokens = selected_values(
                setting_indices, metrics, branch, "generated_tokens"
            )
            total_tokens = selected_values(
                setting_indices, metrics, branch, "total_tokens"
            )
            em_pct = float(em.mean() * 100.0)
            f1_pct = float(f1.mean() * 100.0)
            routed_total = float(total_tokens.sum())
            baseline_total = baselines[f"{branch}_full_baseline_total_tokens"]
            row.update(
                {
                    f"{branch}_em": em_pct,
                    f"{branch}_f1": f1_pct,
                    f"{branch}_em_change_pp_vs_full": (
                        em_pct - baselines[f"{branch}_full_baseline_em"]
                    ),
                    f"{branch}_f1_change_pp_vs_full": (
                        f1_pct - baselines[f"{branch}_full_baseline_f1"]
                    ),
                    f"{branch}_avg_input_tokens": float(input_tokens.mean()),
                    f"{branch}_avg_generated_tokens": float(
                        generated_tokens.mean()
                    ),
                    f"{branch}_avg_total_tokens": float(total_tokens.mean()),
                    f"{branch}_total_tokens": routed_total,
                    f"{branch}_total_token_saving_pct_vs_full": (
                        100.0 * (1.0 - routed_total / baseline_total)
                        if baseline_total != 0.0
                        else 0.0
                    ),
                }
            )
        row.update(baselines)
        rows.append(row)
    return pd.DataFrame(rows, columns=POLICY_COLUMNS + BASELINE_COLUMNS)


def pareto_flags(scores: np.ndarray, total_tokens: np.ndarray) -> np.ndarray:
    flags = np.ones(len(scores), dtype=bool)
    for index, (score, tokens) in enumerate(zip(scores, total_tokens, strict=True)):
        dominates = (
            (scores >= score)
            & (total_tokens <= tokens)
            & ((scores > score) | (total_tokens < tokens))
        )
        flags[index] = not bool(dominates.any())
    return flags


def add_pareto_flags(sweep: pd.DataFrame) -> pd.DataFrame:
    result = sweep.copy()
    for branch in ("direct", "tiser"):
        tokens = result[f"{branch}_total_tokens"].to_numpy(dtype=np.float64)
        for metric in ("em", "f1"):
            scores = result[f"{branch}_{metric}"].to_numpy(dtype=np.float64)
            result[f"{branch}_{metric}_pareto"] = pareto_flags(scores, tokens)
    return result.loc[:, OUTPUT_COLUMNS]


def print_tiser_em_frontier(sweep: pd.DataFrame) -> None:
    columns = {
        "threshold_a": "a",
        "threshold_b": "b",
        "threshold_c": "c",
        "tiser_em": "TISER EM",
        "tiser_em_change_pp_vs_full": "EM change vs full",
        "tiser_total_token_saving_pct_vs_full": "TISER total-token saving %",
        "top3_pct": "top3 %",
        "top5_pct": "top5 %",
        "top7_pct": "top7 %",
        "full_pct": "full %",
    }
    frontier = sweep.loc[sweep["tiser_em_pareto"], list(columns)].rename(
        columns=columns
    )
    frontier = frontier.sort_values(
        ["TISER total-token saving %", "TISER EM"],
        ascending=[False, False],
        kind="mergesort",
    )
    print("\nTISER EM Pareto frontier")
    print(frontier.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


def validate_paths(paths: list[Path]) -> None:
    for index, path in enumerate(paths):
        for previous in paths[:index]:
            if path.resolve() == previous.resolve() or (
                path.exists() and previous.exists() and path.samefile(previous)
            ):
                raise ValueError("All input and output paths must refer to distinct files.")


def main() -> None:
    args = parse_args()
    validate_paths(
        [
            args.complexity_input,
            args.full_input,
            args.top3_input,
            args.top5_input,
            args.top7_input,
            args.output,
            args.pareto_output,
        ]
    )
    full_record_list = load_jsonl(
        args.full_input,
        "full input",
        FULL_FIELDS,
    )
    full_record_list = prepare_full_records(full_record_list)
    keys, full_records = map_by_source_row_id(full_record_list, "full input")

    complexity_record_list = load_jsonl(
        args.complexity_input,
        "complexity input",
        (COMPLEXITY_FIELD,),
    )
    complexity_record_list = prepare_complexity_records(
        complexity_record_list,
        full_record_list,
    )
    _, complexity_records = map_by_source_row_id(
        complexity_record_list,
        "complexity input",
    )
    filtered_records = {}
    filtered_paths = {
        "top3": args.top3_input,
        "top5": args.top5_input,
        "top7": args.top7_input,
    }
    for setting, path in filtered_paths.items():
        record_list = load_jsonl(
            path,
            setting,
            (SOURCE_ROW_ID_FIELD, *FILTERED_FIELDS),
        )
        _, records = map_by_source_row_id(record_list, setting)
        validate_top_k(records, int(setting.removeprefix("top")), setting)
        filtered_records[setting] = records

    validate_key_sets(
        full_records,
        {"complexity input": complexity_records, **filtered_records},
    )
    validate_identity_consistency(
        keys,
        {
            "complexity input": complexity_records,
            "full input": full_records,
            **filtered_records,
        },
    )
    complexity = values_for(
        keys,
        complexity_records,
        COMPLEXITY_FIELD,
        "complexity input",
    )
    metrics = build_metrics(keys, full_records, filtered_records)
    sweep = add_pareto_flags(evaluate_policies(complexity, metrics))
    sweep = sweep.sort_values(
        ["threshold_a", "threshold_b", "threshold_c"],
        kind="mergesort",
    ).reset_index(drop=True)
    pareto = sweep.loc[sweep.loc[:, PARETO_COLUMNS].any(axis=1)].copy()
    pareto = pareto.sort_values(
        ["tiser_total_tokens", "tiser_em", "threshold_a", "threshold_b", "threshold_c"],
        ascending=[True, False, True, True, True],
        kind="mergesort",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.pareto_output.parent.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(args.output, index=False)
    pareto.to_csv(args.pareto_output, index=False)

    baselines = baseline_values(metrics)
    print(f"Number of aligned examples: {len(keys)}")
    print(f"Number of threshold triples: {len(sweep)}")
    print(
        "Full-context Direct: "
        f"EM {baselines['direct_full_baseline_em']:.2f}%, "
        f"F1 {baselines['direct_full_baseline_f1']:.2f}%, "
        f"total tokens {baselines['direct_full_baseline_total_tokens']:.0f}"
    )
    print(
        "Full-context TISER: "
        f"EM {baselines['tiser_full_baseline_em']:.2f}%, "
        f"F1 {baselines['tiser_full_baseline_f1']:.2f}%, "
        f"total tokens {baselines['tiser_full_baseline_total_tokens']:.0f}"
    )
    for column, label in (
        ("direct_em_pareto", "Direct EM"),
        ("direct_f1_pareto", "Direct F1"),
        ("tiser_em_pareto", "TISER EM"),
        ("tiser_f1_pareto", "TISER F1"),
    ):
        print(f"{label} Pareto policies: {int(sweep[column].sum())}")
    print_tiser_em_frontier(sweep)
    print(f"\nPolicy sweep path: {args.output}")
    print(f"Pareto policies path: {args.pareto_output}")


if __name__ == "__main__":
    main()
