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


def generate_response(
    prompt: str,
    bundle: QwenModelBundle,
    system_prompt: str = "You are a careful temporal reasoning assistant.",
    max_new_tokens: int = 256,
) -> str:
    """Generate a response from Qwen2.5-3B-Instruct."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    text = bundle.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = bundle.tokenizer([text], return_tensors="pt").to(bundle.model.device)

    with torch.no_grad():
        generated_ids = bundle.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
        )

    output_ids = generated_ids[0][inputs.input_ids.shape[-1] :]
    return bundle.tokenizer.decode(output_ids, skip_special_tokens=True)


if __name__ == "__main__":
    qwen = load_qwen_model()
    answer = generate_response("What is temporal reasoning?", qwen)
    print(answer)
