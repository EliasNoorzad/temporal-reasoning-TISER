"""Phase 4: tiny LoRA SFT run for TISER reproduction debugging.

This script is intentionally configured for a cheap end-to-end smoke test, not
the full TISER training experiment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel
from transformers import set_seed
from trl import SFTConfig, SFTTrainer

from src.dataset import load_tiser_dataset
from src.model import MODEL_NAME, load_qwen_model


DEFAULT_OUTPUT_DIR = "/content/drive/MyDrive/TISER/checkpoints/phase4_lora/"
DEFAULT_LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a tiny LoRA SFT smoke test on AmazonScience/TISER."
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--subset-size", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=5)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--optim", default="adamw_torch")
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--verify-samples", type=int, default=3)
    parser.add_argument("--generation-max-new-tokens", type=int, default=1024)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        nargs="+",
        default=list(DEFAULT_LORA_TARGET_MODULES),
    )
    parser.add_argument(
        "--no-gradient-checkpointing",
        action="store_false",
        dest="gradient_checkpointing",
    )
    parser.set_defaults(gradient_checkpointing=True)
    return parser.parse_args()


def prepare_tiser_train_subset(
    subset_size: int,
    seed: int,
    tokenizer: Any,
    max_seq_length: int,
) -> Any:
    dataset = load_tiser_dataset()
    if "train" not in dataset:
        raise KeyError("AmazonScience/TISER does not contain a train split.")

    train_dataset = dataset["train"]
    required_columns = {"prompt", "output"}
    missing_columns = required_columns.difference(train_dataset.column_names)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise KeyError(f"TISER train split is missing required columns: {missing}")

    if subset_size <= 0:
        raise ValueError("--subset-size must be greater than 0 for this smoke test.")

    subset_count = min(subset_size, len(train_dataset))
    train_dataset = train_dataset.shuffle(seed=seed).select(range(subset_count))

    def tokenize_prompt_completion(example: dict[str, Any]) -> dict[str, list[int]]:
        prompt = str(example["prompt"])
        completion = str(example["output"])
        eos_token = tokenizer.eos_token or ""
        if eos_token and not completion.endswith(eos_token):
            completion = f"{completion}{eos_token}"

        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]

        prompt_budget = max_seq_length - len(completion_ids)
        if prompt_budget < 0:
            prompt_ids = []
            completion_ids = completion_ids[:max_seq_length]
            if tokenizer.eos_token_id is not None and completion_ids:
                completion_ids[-1] = tokenizer.eos_token_id
        elif len(prompt_ids) > prompt_budget:
            prompt_ids = prompt_ids[-prompt_budget:] if prompt_budget > 0 else []

        input_ids = prompt_ids + completion_ids
        labels = [-100] * len(prompt_ids) + completion_ids
        attention_mask = [1] * len(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    return train_dataset.map(
        tokenize_prompt_completion,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing TISER prompt/output pairs for SFT",
    )


def build_lora_config(args: argparse.Namespace) -> LoraConfig:
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )


def print_parameter_summary(model: torch.nn.Module) -> None:
    total_params = 0
    trainable_params = 0
    for parameter in model.parameters():
        count = parameter.numel()
        total_params += count
        if parameter.requires_grad:
            trainable_params += count

    trainable_pct = 100 * trainable_params / total_params
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_pct:.4f}%)")
    if trainable_params == 0 or trainable_params == total_params:
        raise RuntimeError("LoRA setup failed: trainable parameter count is invalid.")
    if trainable_pct > 5:
        raise RuntimeError(
            "LoRA setup exposed more than 5% of parameters as trainable; "
            "refusing to continue because this may indicate full fine-tuning."
        )


def build_sft_config(args: argparse.Namespace, output_dir: Path) -> SFTConfig:
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    return SFTConfig(
        output_dir=str(output_dir),
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        optim=args.optim,
        report_to=args.report_to,
        max_length=args.max_seq_length,
        packing=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        save_safetensors=True,
    )


def build_trainer(
    model: torch.nn.Module,
    tokenizer: Any,
    train_dataset: Any,
    args: argparse.Namespace,
    output_dir: Path,
) -> SFTTrainer:
    lora_config = build_lora_config(args)
    sft_config = build_sft_config(args, output_dir)

    # Prompt-completion format trains loss on TISER output tokens only.
    return SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )


def save_final_adapter(trainer: SFTTrainer, tokenizer: Any, output_dir: Path) -> Path:
    final_adapter_dir = output_dir / "final_adapter"
    final_adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_adapter_dir))
    tokenizer.save_pretrained(str(final_adapter_dir))
    print(f"Final LoRA adapter saved to: {final_adapter_dir}")
    return final_adapter_dir


def generate_from_reloaded_adapter(
    adapter_dir: Path,
    prompts: list[str],
    device_map: str,
    max_new_tokens: int,
) -> list[dict[str, str]]:
    base_bundle = load_qwen_model(MODEL_NAME, device_map=device_map)
    model = PeftModel.from_pretrained(base_bundle.model, str(adapter_dir))
    model.eval()
    tokenizer = base_bundle.tokenizer

    generations = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        output_ids = generated_ids[0][inputs.input_ids.shape[-1] :]
        response = tokenizer.decode(output_ids, skip_special_tokens=True)
        generations.append({"prompt": prompt, "response": response})
    return generations


def write_reload_verification(
    generations: list[dict[str, str]],
    output_dir: Path,
) -> Path:
    output_path = output_dir / "reload_verification_generations.jsonl"
    with output_path.open("w", encoding="utf-8") as file:
        for generation in generations:
            file.write(json.dumps(generation, ensure_ascii=False) + "\n")

    print(f"Reload verification generations saved to: {output_path}")
    for index, generation in enumerate(generations, start=1):
        print(f"\nReload verification sample {index}")
        print(f"Prompt: {generation['prompt']}")
        print(f"Response: {generation['response']}")
    return output_path


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_bundle = load_qwen_model(MODEL_NAME, device_map=args.device_map)
    tokenizer = model_bundle.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = prepare_tiser_train_subset(
        subset_size=args.subset_size,
        seed=args.seed,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
    )
    verification_prompts = [
        tokenizer.decode(
            [
                token_id
                for token_id, label in zip(example["input_ids"], example["labels"])
                if label == -100
            ],
            skip_special_tokens=True,
        )
        for example in train_dataset.select(range(min(args.verify_samples, len(train_dataset))))
    ]

    model = model_bundle.model
    model.config.use_cache = False
    if args.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    trainer = build_trainer(model, tokenizer, train_dataset, args, output_dir)
    print_parameter_summary(trainer.model)

    print("Starting tiny Phase 4 LoRA SFT smoke test.")
    trainer.train()
    final_adapter_dir = save_final_adapter(trainer, tokenizer, output_dir)

    del trainer
    del model
    del model_bundle
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    generations = generate_from_reloaded_adapter(
        adapter_dir=final_adapter_dir,
        prompts=verification_prompts,
        device_map=args.device_map,
        max_new_tokens=args.generation_max_new_tokens,
    )
    write_reload_verification(generations, output_dir)


if __name__ == "__main__":
    main()
