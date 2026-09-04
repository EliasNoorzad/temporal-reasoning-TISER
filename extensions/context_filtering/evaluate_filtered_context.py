"""Evaluate the frozen LoRA model on prefiltered temporal contexts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import fmean
import sys
from typing import Any

import torch
from tqdm.auto import tqdm

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.dataset import load_tiser_dataset
from src.evaluate import (
    format_prompts_for_generation,
    generate_responses_with_metadata,
    load_evaluation_model,
    make_combined_result_record,
    to_percentage,
)
from src.model import MODEL_NAME
from src.prompts import (
    ANSWER_SECTION_MARKER,
    TEMPORAL_CONTEXT_MARKER,
    extract_temporal_context,
    get_prompt_text,
)


REQUIRED_FIELDS = (
    "question", "temporal_context", "filtered_temporal_context", "gold_answer",
    "original_context_tokens", "filtered_context_tokens",
    "context_token_saving_pct", "context_filter_top_k",
)
BRANCH_RESULT_FIELDS = (
    "direct_answer", "direct_em", "direct_f1", "direct_generated_tokens",
    "tiser_raw_response", "tiser_answer", "tiser_answer_extraction_status",
    "tiser_em", "tiser_f1", "tiser_generated_tokens",
)
TOKEN_FIELDS = (
    "direct_full_input_tokens", "direct_filtered_input_tokens",
    "direct_input_token_saving_pct", "tiser_full_input_tokens",
    "tiser_filtered_input_tokens", "tiser_input_token_saving_pct",
    "filtered_direct_total_tokens", "filtered_tiser_total_tokens",
    "full_direct_total_tokens", "full_tiser_total_tokens",
)
OUTPUT_FIELDS = tuple(f"filtered_{field}" for field in BRANCH_RESULT_FIELDS) + TOKEN_FIELDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Direct and TISER using already-filtered contexts."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument(
        "--adapter-path", "--lora-adapter-path", dest="lora_adapter_path", required=True
    )
    parser.add_argument(
        "--model-name", default=MODEL_NAME, choices=(MODEL_NAME,),
        help="Use the same base model as the existing frozen-LoRA evaluator.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--direct-max-new-tokens", type=int, default=128)
    parser.add_argument("--tiser-max-new-tokens", type=int, default=2048)
    parser.add_argument("--device-map", default="auto")
    parser.set_defaults(model_type="lora")
    args = parser.parse_args()
    for name in ("batch_size", "direct_max_new_tokens", "tiser_max_new_tokens"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1.")
    return args


def validate_paths(paths: list[Path]) -> None:
    for index, path in enumerate(paths):
        for previous in paths[:index]:
            if path.resolve() == previous.resolve() or (
                path.exists() and previous.exists() and path.samefile(previous)
            ):
                raise ValueError("Input, output, and summary must use distinct files.")


def load_input_records(path: Path) -> list[dict[str, Any]]:
    # The evaluator's resume reader can repair files; the source here stays read-only.
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at line {line_number} in {path}.") from error
            if not isinstance(record, dict):
                raise TypeError(f"Expected a JSON object at line {line_number}.")
            missing = set(REQUIRED_FIELDS).difference(record)
            if missing:
                raise KeyError(f"Line {line_number} is missing: {', '.join(sorted(missing))}")
            collisions = set(OUTPUT_FIELDS).intersection(record)
            if collisions:
                raise ValueError(
                    f"Line {line_number} already contains filtered-evaluation fields: "
                    f"{', '.join(sorted(collisions))}. Refusing to overwrite them."
                )
            for field in ("question", "temporal_context", "filtered_temporal_context"):
                if not isinstance(record[field], str):
                    raise TypeError(f"Line {line_number}: {field} must be a string.")
            top_k = record["context_filter_top_k"]
            if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
                raise ValueError(f"Line {line_number}: context_filter_top_k must be a positive integer.")
            numeric_fields = [
                "original_context_tokens", "filtered_context_tokens", "context_token_saving_pct",
                "direct_em", "direct_f1", "tiser_em", "tiser_f1",
                "direct_generated_tokens", "tiser_generated_tokens",
            ]
            for field in numeric_fields:
                if field not in record or record[field] is None:
                    if field in REQUIRED_FIELDS:
                        raise ValueError(f"Line {line_number}: {field} must be numeric.")
                    continue
                value = float(record[field])
                if not math.isfinite(value):
                    raise ValueError(f"Line {line_number}: {field} must be finite.")
                if field.endswith("_tokens") and (value < 0 or not value.is_integer()):
                    raise ValueError(f"Line {line_number}: {field} must be a nonnegative token count.")
                if field.endswith(("_em", "_f1")) and not 0.0 <= value <= 1.0:
                    raise ValueError(f"Line {line_number}: {field} must be on the 0-1 scale.")
            records.append(record)
    if not records:
        raise ValueError("Input JSONL contains no examples.")
    return records


def resolve_full_tiser_prompts(records: list[dict[str, Any]]) -> list[str]:
    prompts: list[str | None] = [record.get("prompt") for record in records]
    missing_indices = [index for index, prompt in enumerate(prompts) if prompt is None]
    if missing_indices:
        # Combined result files omit the prompt. Recover the exact official text,
        # without constructing a new TISER template or consulting model correctness.
        wanted = {
            (records[index]["question"], records[index]["temporal_context"].strip())
            for index in missing_indices
        }
        wanted_questions = {question for question, _ in wanted}
        dataset = load_tiser_dataset()
        if "test" not in dataset:
            raise KeyError("AmazonScience/TISER does not contain a test split.")
        candidates: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for example in tqdm(dataset["test"], desc="Recovering original TISER prompts"):
            question = str(example["question"])
            if question not in wanted_questions:
                continue
            key = (question, extract_temporal_context(str(example["prompt"])))
            if key in wanted:
                candidates.setdefault(key, []).append(example)
        for index in missing_indices:
            record = records[index]
            key = (record["question"], record["temporal_context"].strip())
            matches = candidates.get(key, [])
            for field in ("question_id", "dataset_name"):
                if record.get(field) is not None:
                    matches = [
                        example for example in matches
                        if str(example.get(field)) == str(record[field])
                    ]
            unique_prompts = {str(example["prompt"]) for example in matches}
            if len(unique_prompts) != 1:
                raise ValueError(
                    f"Cannot recover one unambiguous official prompt for input record {index + 1}. "
                    "Include its original TISER prompt in the input's 'prompt' field."
                )
            prompts[index] = unique_prompts.pop()
    for index, (record, prompt) in enumerate(zip(records, prompts, strict=True)):
        if not isinstance(prompt, str):
            raise TypeError(f"Record {index + 1}: prompt must be a string.")
        if extract_temporal_context(prompt) != record["temporal_context"].strip():
            raise ValueError(f"Record {index + 1}: original prompt and temporal_context differ.")
    return [str(prompt) for prompt in prompts]


def replace_prompt_context(prompt: str, filtered_context: str) -> str:
    start = prompt.index(TEMPORAL_CONTEXT_MARKER) + len(TEMPORAL_CONTEXT_MARKER)
    end = prompt.index(ANSWER_SECTION_MARKER, start)
    block = prompt[start:end]
    if block.strip() == filtered_context:
        return prompt
    if not block.strip():
        raise ValueError("A nonempty filtered context cannot replace an empty original context.")
    content_start = start + len(block) - len(block.lstrip())
    content_end = start + len(block.rstrip())
    # Preserve every instruction and the whitespace around the context block.
    return prompt[:content_start] + filtered_context + prompt[content_end:]


def build_prompt_pairs(record: dict[str, Any], full_prompt: str) -> dict[str, list[str]]:
    full_example = {"question": record["question"], "prompt": full_prompt}
    filtered_example = {
        "question": record["question"],
        "prompt": replace_prompt_context(full_prompt, record["filtered_temporal_context"]),
    }
    return {
        prompt_type: [get_prompt_text(full_example, prompt_type),
                      get_prompt_text(filtered_example, prompt_type)]
        for prompt_type in ("standard", "tiser")
    }


def count_model_input_tokens(
    tokenizer: Any, prompt_texts: list[str], prompt_type: str,
) -> list[int]:
    formatted = format_prompts_for_generation(tokenizer, prompt_texts, prompt_type)
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        inputs = tokenizer(formatted, return_tensors="pt", padding=True)
    finally:
        tokenizer.padding_side = original_padding_side
    # Use the generation tokenizer defaults, including chat/special tokens,
    # but exclude padding added only to align rows within a batch.
    return [int(count) for count in inputs["attention_mask"].sum(dim=1).tolist()]


def token_saving_percentage(full_tokens: float, filtered_tokens: float) -> float:
    return (1.0 - filtered_tokens / full_tokens) * 100.0 if full_tokens else 0.0


def make_filtered_result(
    record: dict[str, Any], filtered_prompt: str,
    direct_generation: dict[str, Any], tiser_generation: dict[str, Any],
    input_counts: dict[str, tuple[int, int]],
) -> dict[str, Any]:
    example = {
        "question_id": record.get("question_id"),
        "dataset_name": record.get("dataset_name"),
        "question": record["question"], "prompt": filtered_prompt,
        "answer": record["gold_answer"],
    }
    evaluated = make_combined_result_record(example, direct_generation, tiser_generation)
    result = dict(record)
    result.update({f"filtered_{field}": evaluated[field] for field in BRANCH_RESULT_FIELDS})
    for branch in ("direct", "tiser"):
        full_count, filtered_count = input_counts[branch]
        result[f"{branch}_full_input_tokens"] = full_count
        result[f"{branch}_filtered_input_tokens"] = filtered_count
        result[f"{branch}_input_token_saving_pct"] = token_saving_percentage(
            full_count, filtered_count
        )
        result[f"filtered_{branch}_total_tokens"] = (
            filtered_count + result[f"filtered_{branch}_generated_tokens"]
        )
        if record.get(f"{branch}_generated_tokens") is not None:
            result[f"full_{branch}_total_tokens"] = (
                full_count + int(float(record[f"{branch}_generated_tokens"]))
            )
    return result


def summarize_results(records: list[dict[str, Any]]) -> dict[str, Any]:
    top_k_values = sorted({record["context_filter_top_k"] for record in records})
    summary: dict[str, Any] = {
        "examples": len(records),
        "context_filter_top_k": top_k_values[0] if len(top_k_values) == 1 else None,
    }
    if len(top_k_values) != 1:
        summary["context_filter_top_k_values"] = top_k_values
    for field in ("original_context_tokens", "filtered_context_tokens", "context_token_saving_pct"):
        summary[f"avg_{field}"] = fmean(float(record[field]) for record in records)
    for branch in ("direct", "tiser"):
        for metric in ("em", "f1"):
            filtered_field = f"filtered_{branch}_{metric}"
            summary[filtered_field] = to_percentage(fmean(record[filtered_field] for record in records))
            full_field = f"{branch}_{metric}"
            paired = [record for record in records if record.get(full_field) is not None]
            if paired:
                summary[f"full_{branch}_{metric}"] = to_percentage(
                    fmean(float(record[full_field]) for record in paired)
                )
                summary[f"{branch}_{metric}_change_pp"] = to_percentage(
                    fmean(record[filtered_field] - float(record[full_field]) for record in paired)
                )
                summary[f"{branch}_{metric}_comparison_examples"] = len(paired)
        for field in (
            f"filtered_{branch}_generated_tokens", f"{branch}_full_input_tokens",
            f"{branch}_filtered_input_tokens", f"{branch}_input_token_saving_pct",
            f"filtered_{branch}_total_tokens",
        ):
            summary[f"avg_{field}"] = fmean(record[field] for record in records)

        # Optional before/after comparisons use the same records on both sides.
        # This avoids comparing a partial baseline with the entire filtered run.
        full_generated_field = f"{branch}_generated_tokens"
        paired = [record for record in records if record.get(full_generated_field) is not None]
        if paired:
            summary[f"{branch}_token_comparison_examples"] = len(paired)
            summary[f"avg_full_{branch}_generated_tokens"] = fmean(
                float(record[full_generated_field]) for record in paired
            )
            summary[f"avg_paired_filtered_{branch}_generated_tokens"] = fmean(
                record[f"filtered_{branch}_generated_tokens"] for record in paired
            )
            summary[f"{branch}_generated_token_change"] = fmean(
                record[f"filtered_{branch}_generated_tokens"] - float(record[full_generated_field])
                for record in paired
            )
            summary[f"avg_full_{branch}_total_tokens"] = fmean(
                record[f"full_{branch}_total_tokens"] for record in paired
            )
            summary[f"avg_paired_filtered_{branch}_total_tokens"] = fmean(
                record[f"filtered_{branch}_total_tokens"] for record in paired
            )
            full_total = sum(record[f"full_{branch}_total_tokens"] for record in paired)
            filtered_total = sum(record[f"filtered_{branch}_total_tokens"] for record in paired)
            summary[f"{branch}_total_token_saving_pct"] = token_saving_percentage(full_total, filtered_total)
    return summary


def print_summary(summary: dict[str, Any], args: argparse.Namespace) -> None:
    print(f"Evaluated examples: {summary['examples']}")
    print(f"Top-k: {summary['context_filter_top_k'] or summary.get('context_filter_top_k_values')}")
    print(
        f"Average context tokens: {summary['avg_original_context_tokens']:.2f} -> "
        f"{summary['avg_filtered_context_tokens']:.2f}; "
        f"average saving: {summary['avg_context_token_saving_pct']:.2f}%"
    )
    for branch, label in (("direct", "Direct"), ("tiser", "TISER")):
        for metric in ("em", "f1"):
            full = summary.get(f"full_{branch}_{metric}")
            filtered = summary[f"filtered_{branch}_{metric}"]
            if full is None:
                print(f"{label} {metric.upper()}: full unavailable; filtered {filtered:.2f}%")
            else:
                change = summary[f"{branch}_{metric}_change_pp"]
                paired = summary[f"{branch}_{metric}_comparison_examples"]
                print(
                    f"{label} {metric.upper()}: full {full:.2f}%; filtered {filtered:.2f}%; "
                    f"paired change {change:+.2f} pp ({paired} examples)"
                )
        print(
            f"{label} average input tokens: {summary[f'avg_{branch}_full_input_tokens']:.2f} -> "
            f"{summary[f'avg_{branch}_filtered_input_tokens']:.2f}; "
            f"average saving: {summary[f'avg_{branch}_input_token_saving_pct']:.2f}%"
        )
        full_generated = summary.get(f"avg_full_{branch}_generated_tokens")
        if full_generated is not None:
            paired_count = summary[f"{branch}_token_comparison_examples"]
            paired_filtered_generated = summary[f"avg_paired_filtered_{branch}_generated_tokens"]
            full_total = summary[f"avg_full_{branch}_total_tokens"]
            saving = summary[f"{branch}_total_token_saving_pct"]
            paired_filtered_total = summary[f"avg_paired_filtered_{branch}_total_tokens"]
            print(
                f"{label} average generated tokens: {full_generated:.2f} -> "
                f"{paired_filtered_generated:.2f} ({paired_count} paired examples)"
            )
            print(
                f"{label} average total tokens: {full_total:.2f} -> {paired_filtered_total:.2f}; "
                f"total-token saving: {saving:.2f}%"
            )
        else:
            print(f"{label} average filtered generated tokens: {summary[f'avg_filtered_{branch}_generated_tokens']:.2f}")
            print(f"{label} average filtered total tokens: {summary[f'avg_filtered_{branch}_total_tokens']:.2f}")
    print(f"Output JSONL path: {args.output}")
    print(f"Summary JSON path: {args.summary_output}")


def run_evaluation(args: argparse.Namespace) -> None:
    validate_paths([args.input, args.output, args.summary_output])
    records = load_input_records(args.input)
    full_prompts = resolve_full_tiser_prompts(records)
    tokenizer, model = load_evaluation_model(args)
    model.requires_grad_(False)
    model.eval()

    results = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file, tqdm(
        total=len(records), desc="Evaluating filtered Direct and TISER", unit="examples"
    ) as progress:
        for start in range(0, len(records), args.batch_size):
            batch = records[start:start + args.batch_size]
            pairs = [
                build_prompt_pairs(record, prompt)
                for record, prompt in zip(batch, full_prompts[start:start + len(batch)], strict=True)
            ]
            input_counts: list[dict[str, tuple[int, int]]] = [{} for _ in batch]
            generations = {}
            for branch, prompt_type, limit in (
                ("direct", "standard", args.direct_max_new_tokens),
                ("tiser", "tiser", args.tiser_max_new_tokens),
            ):
                full = [pair[prompt_type][0] for pair in pairs]
                filtered = [pair[prompt_type][1] for pair in pairs]
                counts = count_model_input_tokens(tokenizer, full + filtered, prompt_type)
                for index in range(len(batch)):
                    input_counts[index][branch] = (counts[index], counts[index + len(batch)])
                with torch.inference_mode():
                    generations[branch] = generate_responses_with_metadata(
                        model=model, tokenizer=tokenizer, prompt_texts=filtered,
                        prompt_type=prompt_type, max_new_tokens=limit,
                    )
            batch_results = [
                make_filtered_result(record, pair["tiser"][1], direct, tiser, counts)
                for record, pair, direct, tiser, counts in zip(
                    batch, pairs, generations["direct"], generations["tiser"], input_counts,
                    strict=True,
                )
            ]
            for result in batch_results:
                file.write(json.dumps(result, ensure_ascii=False) + "\n")
            file.flush()
            results.extend(batch_results)
            progress.update(len(batch_results))

    summary = summarize_results(results)
    summary["configuration"] = {
        "model_name": args.model_name, "adapter_path": args.lora_adapter_path,
        "batch_size": args.batch_size, "device_map": args.device_map,
        "direct_max_new_tokens": args.direct_max_new_tokens,
        "tiser_max_new_tokens": args.tiser_max_new_tokens,
        "input_path": str(args.input), "output_path": str(args.output),
        "metric_aggregation": "pooled_per_example",
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_output.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print_summary(summary, args)


if __name__ == "__main__":
    run_evaluation(parse_args())
