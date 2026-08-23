"""LoRA supervised fine-tuning for TISER."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel
from tqdm.auto import tqdm
from transformers import TrainerCallback, TrainerControl, TrainerState, set_seed
from trl import SFTConfig, SFTTrainer

from src.dataset import load_tiser_dataset
from src.evaluate import (
    compute_macro_metrics,
    exact_match,
    extract_prediction,
    generate_responses,
    token_f1,
    to_percentage,
)
from src.model import MODEL_NAME, load_qwen_model


DEFAULT_OUTPUT_DIR = "/content/drive/MyDrive/TISER/checkpoints/train_lora/"
DEFAULT_LORA_TARGET_MODULES = (
    # LoRA is applied to Qwen's attention projection layers, so the base model
    # stays frozen while these small adapter matrices learn the task.
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LoRA supervised fine-tuning on AmazonScience/TISER."
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--validation-ratio", type=float, required=True)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--optim", default="adamw_torch")
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--verify-samples", type=int, default=3)
    parser.add_argument("--generation-max-new-tokens", type=int, default=2048)
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


def split_tiser_train_validation(
    validation_ratio: float,
    seed: int,
) -> tuple[Any, Any]:
    if not 0 < validation_ratio < 1:
        raise ValueError("--validation-ratio must be greater than 0 and less than 1.")

    dataset = load_tiser_dataset()
    if "train" not in dataset:
        raise KeyError("AmazonScience/TISER does not contain a train split.")

    train_dataset = dataset["train"]
    # In TISER, the prompt is the question/context given to the model, and the
    # output is the answer we want the model to learn to produce.
    required_columns = {"prompt", "output", "answer", "dataset_name"}
    missing_columns = required_columns.difference(train_dataset.column_names)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise KeyError(f"TISER train split is missing required columns: {missing}")

    # Split before tokenization so validation keeps the original fields needed
    # for generation and metric calculation.
    original_train_examples = len(train_dataset)
    split_dataset = train_dataset.train_test_split(
        test_size=validation_ratio,
        seed=seed,
        shuffle=True,
    )
    training_dataset = split_dataset["train"]
    validation_dataset = split_dataset["test"]

    print(f"Original training examples: {original_train_examples}")
    print(f"Training examples: {len(training_dataset)}")
    print(f"Validation examples: {len(validation_dataset)}")
    return training_dataset, validation_dataset


def prepare_tiser_train_dataset(
    train_dataset: Any,
    tokenizer: Any,
    max_seq_length: int,
) -> Any:
    def tokenize_prompt_completion(example: dict[str, Any]) -> dict[str, list[int]]:
        prompt = str(example["prompt"])
        completion = str(example["output"])
        eos_token = tokenizer.eos_token or ""
        if eos_token and not completion.endswith(eos_token):
            completion = f"{completion}{eos_token}"

        # The prompt and answer are tokenized separately so we know exactly
        # which tokens belong to the input and which tokens belong to the target.
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]

        # If the combined sequence is longer than the limit, preserve as much of
        # the answer as possible because only answer tokens contribute to loss.
        # The prompt is trimmed first; if the answer alone is too long, it is cut
        # to the maximum length and forced to end with EOS when available.
        prompt_budget = max_seq_length - len(completion_ids)
        if prompt_budget < 0:
            prompt_ids = []
            completion_ids = completion_ids[:max_seq_length]
            if tokenizer.eos_token_id is not None and completion_ids:
                completion_ids[-1] = tokenizer.eos_token_id
        elif len(prompt_ids) > prompt_budget:
            prompt_ids = prompt_ids[-prompt_budget:] if prompt_budget > 0 else []

        input_ids = prompt_ids + completion_ids
        # Labels use -100 for prompt tokens because the model should condition
        # on the prompt, not be trained to reproduce it. The answer keeps its
        # token IDs, so supervised fine-tuning optimizes only the answer text.
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
    # The LoRA settings are CLI arguments so experiments can adjust adapter
    # capacity without changing the training script.
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )


def print_parameter_summary(model: torch.nn.Module) -> None:
    # LoRA should leave almost all base-model weights frozen. Printing and
    # checking this count catches accidental full fine-tuning before training.
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


def format_epoch(epoch_value: float | None, fallback: int) -> int | float:
    if epoch_value is None:
        return fallback
    rounded_epoch = round(float(epoch_value), 6)
    if rounded_epoch.is_integer():
        return int(rounded_epoch)
    return rounded_epoch


def evaluate_validation_dataset(
    model: torch.nn.Module,
    tokenizer: Any,
    validation_dataset: Any,
    batch_size: int,
    max_new_tokens: int,
    epoch: int | float,
) -> dict[str, float | int]:
    was_training = model.training
    model.eval()

    records = []
    try:
        with tqdm(
            total=len(validation_dataset),
            desc=f"Validation epoch {epoch}",
        ) as progress_bar:
            for batch_start in range(0, len(validation_dataset), batch_size):
                batch_end = min(batch_start + batch_size, len(validation_dataset))
                batch_examples = [
                    validation_dataset[index]
                    for index in range(batch_start, batch_end)
                ]
                prompt_texts = [str(example["prompt"]) for example in batch_examples]
                raw_responses = generate_responses(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_texts=prompt_texts,
                    prompt_type="tiser",
                    max_new_tokens=max_new_tokens,
                )

                for example, raw_response in zip(batch_examples, raw_responses):
                    prediction, _ = extract_prediction(raw_response, "tiser")
                    gold_answer = str(example["answer"])
                    records.append(
                        {
                            "dataset_name": example["dataset_name"],
                            "exact_match": exact_match(prediction, gold_answer),
                            "token_f1": token_f1(prediction, gold_answer),
                        }
                    )
                progress_bar.update(len(batch_examples))
    finally:
        if was_training:
            model.train()

    if not records:
        return {
            "epoch": epoch,
            "overall_em": 0.0,
            "macro_token_f1": 0.0,
        }

    # Macro F1 is calculated by averaging F1 within each component dataset first,
    # then averaging those dataset-level scores so each dataset has equal weight.
    _, macro_token_f1 = compute_macro_metrics(records)
    overall_em = sum(record["exact_match"] for record in records) / len(records)
    return {
        "epoch": epoch,
        "overall_em": to_percentage(overall_em),
        "macro_token_f1": to_percentage(macro_token_f1),
    }


def write_validation_metrics(
    output_dir: Path,
    validation_metrics: list[dict[str, float | int]],
) -> Path:
    output_path = output_dir / "validation_metrics.json"
    temporary_path = output_dir / "validation_metrics.json.tmp"
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(validation_metrics, file, indent=2)
        file.write("\n")
    temporary_path.replace(output_path)
    return output_path


class EpochValidationCallback(TrainerCallback):
    def __init__(
        self,
        validation_dataset: Any,
        tokenizer: Any,
        output_dir: Path,
        batch_size: int,
        max_new_tokens: int,
    ) -> None:
        self.validation_dataset = validation_dataset
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.completed_epochs: set[int | float] = set()
        self.validation_metrics: list[dict[str, float | int]] = []

    def on_save(
        self,
        args: Any,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> TrainerControl:
        if not getattr(state, "is_world_process_zero", True):
            return control

        model = kwargs.get("model")
        if model is None:
            return control

        epoch = format_epoch(state.epoch, len(self.completed_epochs) + 1)
        if epoch in self.completed_epochs:
            return control
        self.completed_epochs.add(epoch)

        # Checkpoint saves happen once per epoch, so validation runs at the same
        # boundary instead of running expensive generation every few steps.
        print(f"\nEpoch {epoch} validation")
        metrics = evaluate_validation_dataset(
            model=model,
            tokenizer=self.tokenizer,
            validation_dataset=self.validation_dataset,
            batch_size=self.batch_size,
            max_new_tokens=self.max_new_tokens,
            epoch=epoch,
        )
        self.validation_metrics.append(metrics)
        metrics_path = write_validation_metrics(self.output_dir, self.validation_metrics)
        print(f"Overall EM: {metrics['overall_em']:.2f}%")
        print(f"Macro F1: {metrics['macro_token_f1']:.2f}%")
        print(f"Validation metrics saved to: {metrics_path}")
        return control


def build_sft_config(args: argparse.Namespace, output_dir: Path) -> SFTConfig:
    # On Colab GPUs, bf16 is preferred when the hardware supports it; otherwise
    # fp16 reduces memory use compared with full float32 training.
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    return SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        logging_first_step=True,
        save_strategy="epoch",
        save_total_limit=args.save_total_limit,
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        optim=args.optim,
        report_to=args.report_to,
        max_length=args.max_seq_length,
        packing=False,
        # The dataset is already tokenized and includes labels, so TRL should not
        # rebuild prompt/completion masks or alter the examples.
        dataset_kwargs={"skip_prepare_dataset": True}
    )


def build_trainer(
    model: torch.nn.Module,
    tokenizer: Any,
    train_dataset: Any,
    args: argparse.Namespace,
    output_dir: Path,
    callbacks: list[TrainerCallback] | None = None,
) -> SFTTrainer:
    lora_config = build_lora_config(args)
    sft_config = build_sft_config(args, output_dir)

    # SFTTrainer still handles the training loop, while our explicit labels
    # decide which tokens are trained on.
    return SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
        callbacks=callbacks,
    )


def save_final_adapter(trainer: SFTTrainer, tokenizer: Any, output_dir: Path) -> Path:
    # Epoch checkpoints keep each training state. final_adapter is only the
    # final training state, not a validation-selected best checkpoint.
    final_adapter_dir = output_dir / "final_adapter"
    final_adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_adapter_dir))
    tokenizer.save_pretrained(str(final_adapter_dir))
    print(f"Final training-state LoRA adapter saved to: {final_adapter_dir}")
    return final_adapter_dir


def generate_from_reloaded_adapter(
    adapter_dir: Path,
    prompts: list[str],
    device_map: str,
    max_new_tokens: int,
) -> list[dict[str, str]]:
    # Loading the adapter from disk checks the handoff that matters in practice:
    # a later notebook or runtime can attach these LoRA weights to the same base model.
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
    # The saved generations are a quick sanity check that the reloaded adapter
    # can run inference and produce text for real TISER prompts.
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
    if args.per_device_eval_batch_size < 1:
        raise ValueError("--per-device-eval-batch-size must be at least 1.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_bundle = load_qwen_model(MODEL_NAME, device_map=args.device_map)
    tokenizer = model_bundle.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    training_dataset, validation_dataset = split_tiser_train_validation(
        validation_ratio=args.validation_ratio,
        seed=args.seed,
    )

    # Tokenization happens before TRL so the prompt mask and answer labels stay
    # exactly as defined in this script.
    train_dataset = prepare_tiser_train_dataset(
        train_dataset=training_dataset,
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

    validation_callback = EpochValidationCallback(
        validation_dataset=validation_dataset,
        tokenizer=tokenizer,
        output_dir=output_dir,
        batch_size=args.per_device_eval_batch_size,
        max_new_tokens=args.generation_max_new_tokens,
    )
    trainer = build_trainer(
        model,
        tokenizer,
        train_dataset,
        args,
        output_dir,
        callbacks=[validation_callback],
    )
    print_parameter_summary(trainer.model)

    print("Starting LoRA supervised fine-tuning.")
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
