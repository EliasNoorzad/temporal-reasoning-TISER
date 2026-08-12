"""Model loading helpers for Qwen2.5-3B-Instruct."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"


@dataclass
class QwenModelBundle:
    """Container for the tokenizer and model."""

    tokenizer: AutoTokenizer
    model: AutoModelForCausalLM


def get_default_dtype() -> torch.dtype:
    """Use a GPU-friendly dtype when CUDA is available."""
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def load_qwen_model(
    model_name: str = MODEL_NAME,
    device_map: str = "auto",
    torch_dtype: torch.dtype | None = None,
) -> QwenModelBundle:
    """Load Qwen2.5-3B-Instruct from Hugging Face.

    This uses Hugging Face's normal cache mechanism and does not manually
    download model files.
    """
    dtype = torch_dtype or get_default_dtype()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()
    return QwenModelBundle(tokenizer=tokenizer, model=model)


if __name__ == "__main__":
    load_qwen_model()
