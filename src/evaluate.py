"""Evaluation pipeline for TISER experiments."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import LogitsProcessor, LogitsProcessorList

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.dataset import load_filtered_test_dataset
from src.model import MODEL_NAME, load_qwen_model
from src.prompts import extract_temporal_context, get_prompt_text


ANSWER_TAG_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
ANSWER_CLOSING_TAG = "</answer>"
DIRECT_LEADING_LABEL_PATTERN = re.compile(
    r"^\s*(?:Answer|Assistant)\s*:\s*",
    re.IGNORECASE,
)
DIRECT_CONTINUATION_PATTERN = re.compile(
    r"^[ \t]*(?:Human|Assistant|Explanation|Reasoning|Rationale|Analysis)\s*:",
    re.IGNORECASE | re.MULTILINE,
)


class AnswerClosingTagLogitsProcessor(LogitsProcessor):
    """Finish each TISER row after its generated answer tag is complete."""

    def __init__(self, tokenizer: Any, prompt_length: int, eos_token_id: int) -> None:
        self.prompt_length = prompt_length
        self.eos_token_id = eos_token_id
        closing_tag_ids = tokenizer(
            ANSWER_CLOSING_TAG,
            add_special_tokens=False,
        )["input_ids"]
        if not closing_tag_ids:
            raise ValueError("The tokenizer produced no tokens for </answer>.")

        # Verify that these IDs represent the exact literal tag before using
        # them as a suffix of the generated continuation.
        decoded_closing_tag = tokenizer.decode(
            closing_tag_ids,
            skip_special_tokens=False,
        )
        if decoded_closing_tag != ANSWER_CLOSING_TAG:
            raise ValueError(
                "The tokenizer cannot represent </answer> as an exact token suffix."
            )

        self.closing_tag_token_ids = closing_tag_ids
        self._closing_tag_tensor: torch.Tensor | None = None

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        generated_length = input_ids.shape[1] - self.prompt_length
        closing_tag_length = len(self.closing_tag_token_ids)
        if generated_length < closing_tag_length:
            return scores

        if (
            self._closing_tag_tensor is None
            or self._closing_tag_tensor.device != input_ids.device
        ):
            self._closing_tag_tensor = input_ids.new_tensor(
                self.closing_tag_token_ids
            )

        # Match only the generated suffix. Completed rows emit EOS while other
        # rows in the batch continue normally.
        completed_rows = torch.all(
            input_ids[:, -closing_tag_length:] == self._closing_tag_tensor,
            dim=1,
        )
        scores[completed_rows, :] = -float("inf")
        scores[completed_rows, self.eos_token_id] = 0.0
        return scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Qwen on the TISER test split.")
    parser.add_argument("--model-type", choices=("base", "lora"), required=True)
    parser.add_argument(
        "--prompt-type",
        choices=("standard", "tiser", "both"),
        required=True,
    )
    parser.add_argument("--lora-adapter-path", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--direct-max-new-tokens", type=int, default=128)
    parser.add_argument("--tiser-max-new-tokens", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device-map", default="auto")
    return parser.parse_args()


def load_evaluation_model(args: argparse.Namespace) -> tuple[Any, torch.nn.Module]:
    bundle = load_qwen_model(MODEL_NAME, device_map=args.device_map)
    tokenizer = bundle.tokenizer
    model = bundle.model

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.model_type == "lora":
        if not args.lora_adapter_path:
            raise ValueError("--lora-adapter-path is required when --model-type lora.")
        # LoRA runs use the same Qwen base model, then attach only the trained
        # adapter weights so the base and fine-tuned conditions stay comparable.
        model = PeftModel.from_pretrained(model, args.lora_adapter_path)

    model.eval()
    return tokenizer, model


def get_model_input_device(model: torch.nn.Module) -> torch.device:
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


def format_prompts_for_generation(
    tokenizer: Any,
    prompt_texts: list[str],
    prompt_type: str,
) -> list[str]:
    if prompt_type != "standard":
        return prompt_texts

    return [
        tokenizer.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": (
                        "Return only the final answer. "
                        "Do not provide reasoning, explanation, or intermediate steps."
                    ),
                },
                {"role": "user", "content": prompt_text},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt_text in prompt_texts
    ]


def generate_responses(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt_texts: list[str],
    prompt_type: str,
    max_new_tokens: int,
) -> list[str]:
    formatted_prompt_texts = format_prompts_for_generation(
        tokenizer,
        prompt_texts,
        prompt_type,
    )

    # Qwen is decoder-only, so left padding keeps the end of each prompt aligned
    # at the point where generation should begin for batched inference.
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        inputs = tokenizer(formatted_prompt_texts, return_tensors="pt", padding=True)
    finally:
        tokenizer.padding_side = original_padding_side

    input_length = inputs["input_ids"].shape[-1]
    input_device = get_model_input_device(model)
    inputs = {name: value.to(input_device) for name, value in inputs.items()}

    generation_args = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
    }
    if tokenizer.eos_token_id is not None:
        generation_args["pad_token_id"] = tokenizer.eos_token_id
        generation_args["eos_token_id"] = tokenizer.eos_token_id
    if prompt_type == "tiser":
        if tokenizer.eos_token_id is None:
            raise ValueError("TISER batched stopping requires an EOS token.")
        # The closing answer tag is ordinary text, not the model's EOS token.
        # The logits processor finishes each completed row without stopping the rest.
        generation_args["logits_processor"] = LogitsProcessorList(
            [
                AnswerClosingTagLogitsProcessor(
                    tokenizer,
                    input_length,
                    tokenizer.eos_token_id,
                )
            ]
        )

    with torch.inference_mode():
        generated_ids = model.generate(**generation_args)

    # With left padding, the padded input length is the prompt boundary for
    # every row, so slicing from there removes both prompt and pad tokens.
    responses = []
    for row_ids in generated_ids:
        new_token_ids = row_ids[input_length:]
        responses.append(tokenizer.decode(new_token_ids, skip_special_tokens=True).strip())
    return responses


def count_generated_tokens(
    generated_row: torch.LongTensor,
    input_length: int,
    eos_token_id: int | None,
    pad_token_id: int | None,
) -> int:
    """Count continuation tokens without prompt padding or trailing special tokens."""
    token_ids = generated_row[input_length:].tolist()

    # The first EOS marks the end of the model's text. Later EOS/pad values are
    # batch padding added while other rows continue generating.
    if eos_token_id is not None and eos_token_id in token_ids:
        token_ids = token_ids[: token_ids.index(eos_token_id)]
    elif pad_token_id is not None:
        while token_ids and token_ids[-1] == pad_token_id:
            token_ids.pop()

    return len(token_ids)


def generate_responses_with_metadata(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt_texts: list[str],
    prompt_type: str,
    max_new_tokens: int,
) -> list[dict[str, str | int]]:
    """Generate a batch and return each response with its generated-token count."""
    if not prompt_texts:
        return []

    formatted_prompt_texts = format_prompts_for_generation(
        tokenizer,
        prompt_texts,
        prompt_type,
    )

    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        inputs = tokenizer(formatted_prompt_texts, return_tensors="pt", padding=True)
    finally:
        tokenizer.padding_side = original_padding_side

    input_length = inputs["input_ids"].shape[-1]
    input_device = get_model_input_device(model)
    inputs = {name: value.to(input_device) for name, value in inputs.items()}

    generation_args = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
    }
    if tokenizer.eos_token_id is not None:
        generation_args["pad_token_id"] = tokenizer.eos_token_id
        generation_args["eos_token_id"] = tokenizer.eos_token_id
    if prompt_type == "tiser":
        if tokenizer.eos_token_id is None:
            raise ValueError("TISER batched stopping requires an EOS token.")
        generation_args["logits_processor"] = LogitsProcessorList(
            [
                AnswerClosingTagLogitsProcessor(
                    tokenizer,
                    input_length,
                    tokenizer.eos_token_id,
                )
            ]
        )

    with torch.inference_mode():
        generated_ids = model.generate(**generation_args)

    responses = []
    for row_ids in generated_ids:
        new_token_ids = row_ids[input_length:]
        responses.append(
            {
                "response": tokenizer.decode(
                    new_token_ids,
                    skip_special_tokens=True,
                ).strip(),
                "generated_tokens": count_generated_tokens(
                    generated_row=row_ids,
                    input_length=input_length,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                ),
            }
        )
    return responses


def extract_direct_answer(raw_response: str) -> str:
    """Extract a short Direct answer without using the gold answer."""
    response = str(raw_response).replace("\r\n", "\n").replace("\r", "\n").strip()
    if not response:
        return ""

    response = DIRECT_LEADING_LABEL_PATTERN.sub("", response, count=1).strip()
    continuation_match = DIRECT_CONTINUATION_PATTERN.search(response)
    if continuation_match:
        response = response[: continuation_match.start()].rstrip()

    # Direct answers in this task are short. A blank line is treated as the
    # boundary before an additional explanation, while multiline answers stay intact.
    return re.split(r"\n[ \t]*\n", response, maxsplit=1)[0].strip()


def extract_prediction(raw_response: str, prompt_type: str) -> tuple[str, str]:
    # TISER output can include reasoning before the final answer. The metric
    # should use only the text inside <answer>...</answer> when those tags exist.
    match = ANSWER_TAG_PATTERN.search(raw_response)
    has_open_tag = bool(re.search(r"<answer>", raw_response, re.IGNORECASE))
    has_close_tag = bool(re.search(r"</answer>", raw_response, re.IGNORECASE))

    if match:
        return match.group(1).strip(), "answer_tag_found"

    if prompt_type == "tiser":
        # Missing or broken tags are kept in the result record for inspection
        # instead of failing the whole evaluation run.
        if has_open_tag or has_close_tag:
            return raw_response.strip(), "malformed_answer_tags"
        return raw_response.strip(), "missing_answer_tags"

    # Standard prompting has no tag contract, so use its conservative text extractor.
    direct_answer = extract_direct_answer(raw_response)
    if has_open_tag or has_close_tag:
        return direct_answer, "malformed_answer_tags"
    return direct_answer, "plain_text"


def normalize_for_metrics(text: Any) -> str:
    # EM and token F1 should not change because a model wrote "Bristol,Connecticut"
    # instead of "Bristol, Connecticut", but punctuation and casing are preserved.
    normalized = " ".join(str(text).strip().split())
    normalized = re.sub(r"\s+([,.:;?!])", r"\1", normalized)
    normalized = re.sub(r"([,.:;?!])(?=[^\s,.:;?!])", r"\1 ", normalized)
    return normalized.strip()


def exact_match(prediction: str, gold_answer: str) -> bool:
    return normalize_for_metrics(prediction) == normalize_for_metrics(gold_answer)


def tokenize_for_f1(text: Any) -> list[str]:
    normalized = normalize_for_metrics(text)
    if not normalized:
        return []
    return normalized.split()


def token_f1(prediction: str, gold_answer: str) -> float:
    # Token F1 gives partial credit when the prediction overlaps with the gold
    # answer, while exact match remains strict after the shared normalization.
    prediction_tokens = tokenize_for_f1(prediction)
    gold_tokens = tokenize_for_f1(gold_answer)

    if not prediction_tokens and not gold_tokens:
        return 1.0
    if not prediction_tokens or not gold_tokens:
        return 0.0

    # Counter preserves repeated tokens, so overlap is counted at token level.
    overlap = Counter(prediction_tokens) & Counter(gold_tokens)
    overlap_count = sum(overlap.values())
    if overlap_count == 0:
        return 0.0

    precision = overlap_count / len(prediction_tokens)
    recall = overlap_count / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def make_result_record(
    example: dict[str, Any],
    prompt_type: str,
    model_type: str,
    raw_response: str,
) -> dict[str, Any]:
    prediction, extraction_status = extract_prediction(raw_response, prompt_type)
    gold_answer = str(example["answer"])
    em = exact_match(prediction, gold_answer)
    f1 = token_f1(prediction, gold_answer)

    return {
        "question_id": example["question_id"],
        "dataset_name": example["dataset_name"],
        "prompt_type": prompt_type,
        "model_type": model_type,
        "gold_answer": gold_answer,
        "raw_generated_response": raw_response,
        "extracted_prediction": prediction,
        "answer_extraction_status": extraction_status,
        "exact_match": em,
        "token_f1": f1,
    }


def make_combined_result_record(
    example: dict[str, Any],
    direct_generation: dict[str, str | int],
    tiser_generation: dict[str, str | int],
) -> dict[str, Any]:
    direct_raw_response = str(direct_generation["response"])
    direct_answer = extract_direct_answer(direct_raw_response)
    tiser_raw_response = str(tiser_generation["response"])
    tiser_answer, extraction_status = extract_prediction(
        tiser_raw_response,
        "tiser",
    )
    gold_answer = str(example["answer"])

    return {
        "question_id": example["question_id"],
        "dataset_name": example["dataset_name"],
        "question": str(example["question"]),
        "temporal_context": extract_temporal_context(str(example["prompt"])),
        "gold_answer": gold_answer,
        "direct_answer": direct_answer,
        "direct_em": exact_match(direct_answer, gold_answer),
        "direct_f1": token_f1(direct_answer, gold_answer),
        "direct_generated_tokens": int(direct_generation["generated_tokens"]),
        "tiser_raw_response": tiser_raw_response,
        "tiser_answer": tiser_answer,
        "tiser_answer_extraction_status": extraction_status,
        "tiser_em": exact_match(tiser_answer, gold_answer),
        "tiser_f1": token_f1(tiser_answer, gold_answer),
        "tiser_generated_tokens": int(tiser_generation["generated_tokens"]),
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    records = []
    repair_trailing_line = False
    last_line_ended_with_newline = True
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            last_line_ended_with_newline = line.endswith("\n")
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                if file.read().strip():
                    raise ValueError(
                        f"Invalid JSONL record at line {line_number} in {path}."
                    ) from error
                repair_trailing_line = True
                break
            if not isinstance(record, dict):
                raise TypeError(
                    f"Expected a JSON object at line {line_number} in {path}."
                )
            records.append(record)

    # A runtime interruption can leave a partial final line or omit its newline.
    # Rewrite only the completed records before appending resumed batches.
    if repair_trailing_line or not last_line_ended_with_newline:
        write_jsonl(path, records)

    return records


def to_percentage(score: float) -> float:
    return score * 100


def compute_macro_metrics(records: list[dict[str, Any]]) -> tuple[float, float]:
    # The paper reports a macro average across datasets, so each dataset gets
    # equal weight here even if it contributes a different number of examples.
    records_by_dataset: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        dataset_name = str(record["dataset_name"])
        records_by_dataset.setdefault(dataset_name, []).append(record)

    dataset_em_scores = []
    dataset_f1_scores = []
    for dataset_records in records_by_dataset.values():
        dataset_total = len(dataset_records)
        dataset_em_scores.append(
            sum(record["exact_match"] for record in dataset_records) / dataset_total
        )
        dataset_f1_scores.append(
            sum(record["token_f1"] for record in dataset_records) / dataset_total
        )

    macro_em = sum(dataset_em_scores) / len(dataset_em_scores)
    macro_token_f1 = sum(dataset_f1_scores) / len(dataset_f1_scores)
    return macro_em, macro_token_f1


def write_summary(path: Path, records: list[dict[str, Any]]) -> dict[str, float | int]:
    total_examples = len(records)
    if total_examples == 0:
        summary = {
            "total_examples": 0,
            "overall_em": 0.0,
            "overall_token_f1": 0.0,
            "macro_em": 0.0,
            "macro_token_f1": 0.0,
        }
    else:
        overall_em = sum(record["exact_match"] for record in records) / total_examples
        overall_token_f1 = sum(record["token_f1"] for record in records) / total_examples
        macro_em, macro_token_f1 = compute_macro_metrics(records)
        summary = {
            "total_examples": total_examples,
            "overall_em": to_percentage(overall_em),
            "overall_token_f1": to_percentage(overall_token_f1),
            "macro_em": to_percentage(macro_em),
            "macro_token_f1": to_percentage(macro_token_f1),
        }

    with path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
        file.write("\n")

    return summary


def summarize_combined_branch(
    records: list[dict[str, Any]],
    branch_name: str,
) -> dict[str, float | int]:
    total_examples = len(records)
    if total_examples == 0:
        return {
            f"{branch_name}_overall_em": 0.0,
            f"{branch_name}_overall_token_f1": 0.0,
            f"{branch_name}_macro_em": 0.0,
            f"{branch_name}_macro_token_f1": 0.0,
            f"{branch_name}_total_generated_tokens": 0,
            f"{branch_name}_average_generated_tokens": 0.0,
        }

    metric_records = [
        {
            "dataset_name": record["dataset_name"],
            "exact_match": record[f"{branch_name}_em"],
            "token_f1": record[f"{branch_name}_f1"],
        }
        for record in records
    ]
    overall_em = (
        sum(record["exact_match"] for record in metric_records) / total_examples
    )
    overall_token_f1 = (
        sum(record["token_f1"] for record in metric_records) / total_examples
    )
    macro_em, macro_token_f1 = compute_macro_metrics(metric_records)
    total_generated_tokens = sum(
        int(record[f"{branch_name}_generated_tokens"])
        for record in records
    )

    return {
        f"{branch_name}_overall_em": to_percentage(overall_em),
        f"{branch_name}_overall_token_f1": to_percentage(overall_token_f1),
        f"{branch_name}_macro_em": to_percentage(macro_em),
        f"{branch_name}_macro_token_f1": to_percentage(macro_token_f1),
        f"{branch_name}_total_generated_tokens": total_generated_tokens,
        f"{branch_name}_average_generated_tokens": (
            total_generated_tokens / total_examples
        ),
    }


def write_combined_summary(
    path: Path,
    records: list[dict[str, Any]],
) -> dict[str, float | int]:
    summary = {
        "total_examples": len(records),
        **summarize_combined_branch(records, "direct"),
        **summarize_combined_branch(records, "tiser"),
    }

    with path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
        file.write("\n")

    return summary


def run_combined_prompt_evaluation(
    args: argparse.Namespace,
    model: torch.nn.Module,
    tokenizer: Any,
    test_dataset: Any,
    results_path: Path,
) -> None:
    existing_records = read_jsonl(results_path)
    completed_question_ids = {
        str(record["question_id"])
        for record in existing_records
    }
    del existing_records

    pending_indices = []
    completed_examples = 0
    for index in range(len(test_dataset)):
        question_id = str(test_dataset[index]["question_id"])
        if question_id in completed_question_ids:
            completed_examples += 1
        else:
            pending_indices.append(index)

    with results_path.open("a", encoding="utf-8") as results_file:
        with tqdm(
            total=len(test_dataset),
            initial=completed_examples,
            desc="Generating direct and TISER",
        ) as progress_bar:
            for batch_start in range(0, len(pending_indices), args.batch_size):
                batch_indices = pending_indices[
                    batch_start : batch_start + args.batch_size
                ]
                batch_examples = [
                    test_dataset[index]
                    for index in batch_indices
                ]
                direct_prompts = [
                    get_prompt_text(example, "standard")
                    for example in batch_examples
                ]
                tiser_prompts = [
                    get_prompt_text(example, "tiser")
                    for example in batch_examples
                ]

                direct_generations = generate_responses_with_metadata(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_texts=direct_prompts,
                    prompt_type="standard",
                    max_new_tokens=args.direct_max_new_tokens,
                )
                tiser_generations = generate_responses_with_metadata(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_texts=tiser_prompts,
                    prompt_type="tiser",
                    max_new_tokens=args.tiser_max_new_tokens,
                )

                batch_records = []
                for example, direct_generation, tiser_generation in zip(
                    batch_examples,
                    direct_generations,
                    tiser_generations,
                ):
                    batch_records.append(
                        make_combined_result_record(
                            example=example,
                            direct_generation=direct_generation,
                            tiser_generation=tiser_generation,
                        )
                    )
                for record in batch_records:
                    results_file.write(
                        json.dumps(record, ensure_ascii=False, default=str) + "\n"
                    )
                results_file.flush()
                progress_bar.update(len(batch_records))


def run_evaluation(args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, model = load_evaluation_model(args)
    test_dataset = load_filtered_test_dataset()

    if args.prompt_type == "both":
        result_prefix = f"{args.model_type}_both"
        results_path = output_dir / f"{result_prefix}_results.jsonl"
        summary_path = output_dir / f"{result_prefix}_summary.json"
        run_combined_prompt_evaluation(
            args=args,
            model=model,
            tokenizer=tokenizer,
            test_dataset=test_dataset,
            results_path=results_path,
        )
        records = read_jsonl(results_path)
        summary = write_combined_summary(summary_path, records)

        print(f"Saved predictions to: {results_path}")
        print(f"Saved summary to: {summary_path}")
        print(f"Direct overall EM: {summary['direct_overall_em']:.2f}%")
        print(
            "Direct overall token-level F1: "
            f"{summary['direct_overall_token_f1']:.2f}%"
        )
        print(f"Direct macro EM: {summary['direct_macro_em']:.2f}%")
        print(
            "Direct macro token-level F1: "
            f"{summary['direct_macro_token_f1']:.2f}%"
        )
        print(f"TISER overall EM: {summary['tiser_overall_em']:.2f}%")
        print(
            "TISER overall token-level F1: "
            f"{summary['tiser_overall_token_f1']:.2f}%"
        )
        print(f"TISER macro EM: {summary['tiser_macro_em']:.2f}%")
        print(
            "TISER macro token-level F1: "
            f"{summary['tiser_macro_token_f1']:.2f}%"
        )
        return

    records = []
    with tqdm(total=len(test_dataset), desc="Generating") as progress_bar:
        for batch_start in range(0, len(test_dataset), args.batch_size):
            batch_end = min(batch_start + args.batch_size, len(test_dataset))
            batch_examples = [
                test_dataset[index]
                for index in range(batch_start, batch_end)
            ]
            prompt_texts = [
                get_prompt_text(example, args.prompt_type)
                for example in batch_examples
            ]
            raw_responses = generate_responses(
                model=model,
                tokenizer=tokenizer,
                prompt_texts=prompt_texts,
                prompt_type=args.prompt_type,
                max_new_tokens=(
                    args.direct_max_new_tokens
                    if args.prompt_type == "standard"
                    else args.tiser_max_new_tokens
                ),
            )
            for example, raw_response in zip(batch_examples, raw_responses):
                records.append(
                    make_result_record(
                        example=example,
                        prompt_type=args.prompt_type,
                        model_type=args.model_type,
                        raw_response=raw_response,
                    )
                )
            progress_bar.update(len(batch_examples))

    # Predictions stay in JSONL for per-example inspection, while the summary
    # stores the aggregate metrics used to compare experiment conditions.
    result_prefix = f"{args.model_type}_{args.prompt_type}"
    results_path = output_dir / f"{result_prefix}_results.jsonl"
    summary_path = output_dir / f"{result_prefix}_summary.json"

    write_jsonl(results_path, records)
    summary = write_summary(summary_path, records)

    print(f"Saved predictions to: {results_path}")
    print(f"Saved summary to: {summary_path}")
    print(f"Overall EM: {summary['overall_em']:.2f}%")
    print(f"Overall token-level F1: {summary['overall_token_f1']:.2f}%")
    print(f"Macro EM: {summary['macro_em']:.2f}%")
    print(f"Macro token-level F1: {summary['macro_token_f1']:.2f}%")


if __name__ == "__main__":
    run_evaluation(parse_args())
