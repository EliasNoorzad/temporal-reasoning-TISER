"""Model loading helpers for Qwen2.5-3B-Instruct."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

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


@lru_cache(maxsize=1)
def load_qwen_tokenizer(model_name: str = MODEL_NAME) -> AutoTokenizer:
    """Load and reuse the Qwen tokenizer within one Python process."""
    return AutoTokenizer.from_pretrained(model_name)


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
    tokenizer = load_qwen_tokenizer(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        device_map=device_map
    )
    model.eval()
    return QwenModelBundle(tokenizer=tokenizer, model=model)


if __name__ == "__main__":
    load_qwen_model()
