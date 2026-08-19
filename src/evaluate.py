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

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.dataset import load_tiser_dataset
from src.model import MODEL_NAME, load_qwen_model


ANSWER_TAG_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Qwen on the TISER test split.")
    parser.add_argument("--model-type", choices=("base", "lora"), required=True)
    parser.add_argument("--prompt-type", choices=("standard", "tiser"), required=True)
    parser.add_argument("--lora-adapter-path", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
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
        # LoRA evaluation uses the same base model, with the trained adapter attached.
        model = PeftModel.from_pretrained(model, args.lora_adapter_path)

    model.eval()
    return tokenizer, model


def load_test_split() -> Any:
    dataset = load_tiser_dataset()
    if "test" not in dataset:
        raise KeyError("AmazonScience/TISER does not contain a test split.")

    test_dataset = dataset["test"]
    required_columns = {"question_id", "dataset_name", "question", "prompt", "answer"}
    missing_columns = required_columns.difference(test_dataset.column_names)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise KeyError(f"TISER test split is missing required columns: {missing}")

    return test_dataset


def get_prompt_text(example: dict[str, Any], prompt_type: str) -> str:
    # Standard prompting asks the raw question. TISER prompting uses the dataset's
    # provided prompt, which already contains the intended TISER formatting.
    if prompt_type == "standard":
        return str(example["question"])
    if prompt_type == "tiser":
        return str(example["prompt"])
    raise ValueError(f"Unknown prompt type: {prompt_type}")


def get_model_input_device(model: torch.nn.Module) -> torch.device:
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


def generate_response(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt_text: str,
    max_new_tokens: int,
) -> str:
    inputs = tokenizer(prompt_text, return_tensors="pt")
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

    with torch.inference_mode():
        generated_ids = model.generate(**generation_args)

    # The model output includes the prompt tokens first, so score only the continuation.
    new_token_ids = generated_ids[0][input_length:]
    return tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()


def extract_prediction(raw_response: str, prompt_type: str) -> tuple[str, str]:
    # TISER-formatted generations are expected to put the final answer inside tags.
    # Missing or broken tags are recorded so those cases can be inspected later.
    match = ANSWER_TAG_PATTERN.search(raw_response)
    has_open_tag = bool(re.search(r"<answer>", raw_response, re.IGNORECASE))
    has_close_tag = bool(re.search(r"</answer>", raw_response, re.IGNORECASE))

    if match:
        return match.group(1).strip(), "answer_tag_found"

    if prompt_type == "tiser":
        if has_open_tag or has_close_tag:
            return raw_response.strip(), "malformed_answer_tags"
        return raw_response.strip(), "missing_answer_tags"

    # Standard prompting has no tag contract, so plain generated text is acceptable.
    if has_open_tag or has_close_tag:
        return raw_response.strip(), "malformed_answer_tags"
    return raw_response.strip(), "plain_text"


def normalize_for_metrics(text: Any) -> str:
    # Keep normalization minimal: trim outside whitespace and collapse whitespace runs.
    return " ".join(str(text).strip().split())


def exact_match(prediction: str, gold_answer: str) -> bool:
    return normalize_for_metrics(prediction) == normalize_for_metrics(gold_answer)


def tokenize_for_f1(text: Any) -> list[str]:
    normalized = normalize_for_metrics(text)
    if not normalized:
        return []
    return normalized.split()


def token_f1(prediction: str, gold_answer: str) -> float:
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


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def write_summary(path: Path, records: list[dict[str, Any]]) -> dict[str, float | int]:
    total_examples = len(records)
    if total_examples == 0:
        summary = {
            "total_examples": 0,
            "overall_em": 0.0,
            "overall_token_f1": 0.0,
        }
    else:
        summary = {
            "total_examples": total_examples,
            "overall_em": sum(record["exact_match"] for record in records) / total_examples,
            "overall_token_f1": sum(record["token_f1"] for record in records) / total_examples,
        }

    with path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
        file.write("\n")

    return summary


def run_evaluation(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, model = load_evaluation_model(args)
    test_dataset = load_test_split()

    records = []
    # Generation is intentionally one example at a time to keep memory usage predictable.
    for example in tqdm(test_dataset, desc="Generating", total=len(test_dataset)):
        prompt_text = get_prompt_text(example, args.prompt_type)
        raw_response = generate_response(
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt_text,
            max_new_tokens=args.max_new_tokens,
        )
        records.append(
            make_result_record(
                example=example,
                prompt_type=args.prompt_type,
                model_type=args.model_type,
                raw_response=raw_response,
            )
        )

    # File names include the condition so the four experiment outputs can share a directory.
    result_prefix = f"{args.model_type}_{args.prompt_type}"
    results_path = output_dir / f"{result_prefix}_results.jsonl"
    summary_path = output_dir / f"{result_prefix}_summary.json"

    write_jsonl(results_path, records)
    summary = write_summary(summary_path, records)

    print(f"Saved predictions to: {results_path}")
    print(f"Saved summary to: {summary_path}")
    print(f"Overall EM: {summary['overall_em']:.4f}")
    print(f"Overall token-level F1: {summary['overall_token_f1']:.4f}")


if __name__ == "__main__":
    run_evaluation(parse_args())
