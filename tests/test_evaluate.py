"""Focused tests for TISER evaluation behavior."""

from __future__ import annotations

import sys
import unittest
from unittest.mock import patch

import torch

from src.evaluate import (
    count_generated_tokens,
    extract_direct_answer,
    extract_prediction,
    generate_responses_with_metadata,
    make_combined_result_record,
    parse_args,
)
from src.prompts import get_prompt_text


class FakeTokenizer:
    eos_token_id = 99
    pad_token_id = 99

    def __init__(self) -> None:
        self.padding_side = "right"

    def __call__(
        self,
        prompt_texts: list[str],
        return_tensors: str,
        padding: bool,
    ) -> dict[str, torch.Tensor]:
        self.padding_side_during_call = self.padding_side
        if prompt_texts != ["short", "long"]:
            raise AssertionError(f"Unexpected prompts: {prompt_texts}")
        return {
            "input_ids": torch.tensor([[99, 1], [2, 3]]),
            "attention_mask": torch.tensor([[0, 1], [1, 1]]),
        }

    def decode(
        self,
        token_ids: torch.Tensor,
        skip_special_tokens: bool,
    ) -> str:
        pieces = {
            10: "Alpha",
            20: "Beta",
            21: " answer",
            99: "",
        }
        return "".join(pieces[int(token_id)] for token_id in token_ids)


class FakeModel:
    device = torch.device("cpu")

    def generate(self, **generation_args: object) -> torch.Tensor:
        self.generation_args = generation_args
        input_ids = generation_args["input_ids"]
        continuation_ids = torch.tensor([[10, 99, 99], [20, 21, 99]])
        return torch.cat([input_ids, continuation_ids], dim=1)


class DirectAnswerExtractionTests(unittest.TestCase):
    def test_normal_short_answer(self) -> None:
        response = "Chris Evans was born in Bristol, Connecticut."
        self.assertEqual(extract_direct_answer(response), response)

    def test_explanation_after_answer(self) -> None:
        response = (
            "Chris Evans was born in Bristol, Connecticut.\n\n"
            "The temporal context gives his birthplace directly."
        )
        self.assertEqual(
            extract_direct_answer(response),
            "Chris Evans was born in Bristol, Connecticut.",
        )

    def test_runaway_dialogue_after_answer(self) -> None:
        response = (
            "Chris Evans was born in Bristol, Connecticut.\n"
            "Human: What year was he born?\n"
            "Assistant: He was born in 1981."
        )
        self.assertEqual(
            extract_direct_answer(response),
            "Chris Evans was born in Bristol, Connecticut.",
        )

    def test_valid_tiser_answer_tags(self) -> None:
        prediction, status = extract_prediction(
            "<reasoning>Check the context.</reasoning>\n"
            "<answer>Chris Evans was born in Bristol, Connecticut.</answer>",
            "tiser",
        )
        self.assertEqual(
            prediction,
            "Chris Evans was born in Bristol, Connecticut.",
        )
        self.assertEqual(status, "answer_tag_found")


class GenerationTests(unittest.TestCase):
    def test_batched_generation_and_per_example_token_counts(self) -> None:
        model = FakeModel()
        tokenizer = FakeTokenizer()

        generations = generate_responses_with_metadata(
            model=model,
            tokenizer=tokenizer,
            prompt_texts=["short", "long"],
            prompt_type="standard",
            max_new_tokens=32,
        )

        self.assertEqual(
            generations,
            [
                {"response": "Alpha", "generated_tokens": 1},
                {"response": "Beta answer", "generated_tokens": 2},
            ],
        )
        self.assertEqual(tokenizer.padding_side_during_call, "left")
        self.assertEqual(tokenizer.padding_side, "right")
        self.assertEqual(model.generation_args["max_new_tokens"], 32)
        self.assertFalse(model.generation_args["do_sample"])

    def test_generated_token_count_excludes_prompt_eos_and_padding(self) -> None:
        generated_row = torch.tensor([7, 8, 10, 11, 99, 99])
        self.assertEqual(
            count_generated_tokens(
                generated_row=generated_row,
                input_length=2,
                eos_token_id=99,
                pad_token_id=99,
            ),
            2,
        )


class EvaluationContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.example = {
            "question_id": "q1",
            "dataset_name": "example_dataset",
            "question": "Where was Chris Evans born?",
            "prompt": (
                "Temporal context:\n"
                "Chris Evans was born in Bristol, Connecticut.\n"
                "### Answer:"
            ),
            "answer": "Chris Evans was born in Bristol, Connecticut.",
        }

    def test_standard_prompt_is_unchanged(self) -> None:
        self.assertEqual(
            get_prompt_text(self.example, "standard"),
            "You are an AI assistant that has to respond to questions given a context.\n\n"
            "Question: Where was Chris Evans born?\n\n"
            "Temporal Context: Chris Evans was born in Bristol, Connecticut.",
        )

    def test_combined_result_schema_is_exact(self) -> None:
        record = make_combined_result_record(
            example=self.example,
            direct_generation={
                "response": (
                    "Chris Evans was born in Bristol, Connecticut.\n"
                    "Human: Ask another question."
                ),
                "generated_tokens": 12,
            },
            tiser_generation={
                "response": (
                    "<reasoning>Check the context.</reasoning>\n"
                    "<answer>Chris Evans was born in Bristol, Connecticut.</answer>"
                ),
                "generated_tokens": 24,
            },
        )

        self.assertEqual(
            set(record),
            {
                "question_id",
                "dataset_name",
                "question",
                "temporal_context",
                "gold_answer",
                "direct_answer",
                "direct_em",
                "direct_f1",
                "direct_generated_tokens",
                "tiser_raw_response",
                "tiser_answer",
                "tiser_answer_extraction_status",
                "tiser_em",
                "tiser_f1",
                "tiser_generated_tokens",
            },
        )
        self.assertEqual(record["direct_answer"], self.example["answer"])
        self.assertTrue(record["direct_em"])
        self.assertEqual(record["direct_f1"], 1.0)
        self.assertEqual(record["direct_generated_tokens"], 12)
        self.assertEqual(record["tiser_generated_tokens"], 24)

    def test_cli_uses_separate_generation_limits(self) -> None:
        argv = [
            "evaluate.py",
            "--model-type",
            "base",
            "--prompt-type",
            "both",
            "--output-dir",
            "results",
        ]
        with patch.object(sys, "argv", argv):
            args = parse_args()

        self.assertEqual(args.direct_max_new_tokens, 128)
        self.assertEqual(args.tiser_max_new_tokens, 2048)


if __name__ == "__main__":
    unittest.main()
