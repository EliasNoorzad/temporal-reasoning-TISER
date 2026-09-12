# Adaptive Routing for TISER-Based Temporal Question Answering

This repository reproduces the TISER temporal reasoning workflow using
`Qwen/Qwen2.5-3B-Instruct` as the base model and LoRA supervised fine-tuning on
the official TISER data. It evaluates two inference paths:

- **Direct:** short-answer generation from the question and temporal context.
- **TISER:** the dataset's structured prompt for reasoning, timeline
  construction, reflection, and a final answer.

The repository also includes an adaptive-routing extension. It uses TF-IDF
question-to-context-sentence similarity to estimate how broadly relevant
evidence is distributed, then selects Direct or TISER for each example.

## Resources

- [Original TISER paper](https://arxiv.org/abs/2504.05258)
- [Amazon Science TISER page](https://www.amazon.science/code-and-datasets/tiser)
- [Official AmazonScience/TISER dataset](https://huggingface.co/datasets/AmazonScience/TISER)
- [Qwen2.5-3B-Instruct base model](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
- [Trained LoRA checkpoint](https://huggingface.co/EliElias/TISER-Qwen2.5-3B-LoRA)
- [Evaluation results dataset](https://huggingface.co/datasets/EliElias/TISER-Evaluation-Results)

The Hugging Face evaluation-results dataset contains the saved outputs used for
reproducibility:

- `lora_both_results.jsonl` contains the original Direct and TISER predictions
  and their metrics from the initial evaluation run, before post-hoc
  normalization.
- `lora_both_results_rescored.jsonl` contains the same saved predictions with
  Direct and TISER EM/F1 recomputed by the final deterministic normalization in
  `src/evaluate.py`.

Rescoring does not rerun inference or modify any generated prediction.

## Reproducibility Workflow

The reproducible workflow does not require access to the author's private
Google Drive:

```text
GitHub repository
  -> Hugging Face LoRA checkpoint
  -> Hugging Face saved evaluation predictions
  -> deterministic rescoring with src/evaluate.py
  -> adaptive-routing analysis
  -> saved extension outputs
```

The notebooks can use temporary local Colab storage while running. The
persistent LoRA checkpoint and reusable evaluation outputs are hosted publicly
on Hugging Face, and every output path used by the CLI tools is configurable.

## Evaluation Setup

The official TISER test split is filtered by the tokenized TISER prompt length.
Prompts of at most 2,048 tokens are retained, producing 20,442 evaluation
examples. Predictions are generated and saved for this complete retained set,
including Test-of-Time semantic examples.

The main paper-style evaluation uses 19,102 examples from five in-domain
datasets:

- TGQA (`tgqa_test`)
- TempReason-L2 (`tempreason_l2_test`)
- TempReason-L3 (`tempreason_l3_test`)
- TimeQA-easy (`timeqa_easy_test`)
- TimeQA-hard (`timeqa_hard_test`)

EM and token-level F1 are calculated within each dataset and then averaged
equally across the five datasets. Test-of-Time semantic (`tot_semantic_test`) is
kept as a separate evaluation subset and is excluded from this macro average.

## Final Baseline Results

These are the final normalized scores. All values are percentages.

| Dataset | Direct EM | Direct F1 | TISER EM | TISER F1 |
| --- | ---: | ---: | ---: | ---: |
| TGQA | 47.20 | 64.41 | 68.79 | 80.95 |
| TempReason-L2 | 89.93 | 92.26 | 90.32 | 92.55 |
| TempReason-L3 | 78.18 | 82.09 | 88.21 | 90.58 |
| TimeQA-easy | 92.46 | 96.52 | 94.43 | 97.15 |
| TimeQA-hard | 87.59 | 92.44 | 88.95 | 92.40 |
| **Macro Avg.** | **79.07** | **85.54** | **86.14** | **90.73** |

Generation cost over the 19,102 in-domain examples:

| Path | Average generated tokens | Total generated tokens |
| --- | ---: | ---: |
| Direct | 5.93 | 113,257 |
| TISER | 272.00 | 5,195,747 |

## Raw Pre-Normalization Results

The following five-dataset results were calculated before the final answer
normalization pass. They should not be confused with the final results above.

| Dataset | Direct EM | Direct F1 | TISER EM | TISER F1 |
| --- | ---: | ---: | ---: | ---: |
| TGQA | 38.87 | 60.66 | 54.58 | 75.33 |
| TempReason-L2 | 89.93 | 92.26 | 90.32 | 92.55 |
| TempReason-L3 | 78.18 | 82.09 | 88.21 | 90.58 |
| TimeQA-easy | 92.46 | 96.52 | 94.43 | 97.15 |
| TimeQA-hard | 87.59 | 92.44 | 88.95 | 92.40 |
| **Macro Avg.** | **77.41** | **84.79** | **83.30** | **89.60** |

The original pooled raw metrics over all 20,442 retained examples were:

- Direct: EM 74.83%, token-level F1 81.27%.
- TISER: EM 80.42%, token-level F1 85.91%.

These pooled values are not directly comparable to the final five-dataset
macro values: they use a different aggregation and include every retained test
example, including the separate Test-of-Time semantic subset.

The model predictions were not regenerated or edited to obtain the final
scores. Evaluation applies deterministic post-hoc normalization and the correct
five-dataset macro protocol. The normalization handles specific formatting
differences in duration answers with `year`/`years`, event-boundary answers with
`starts`/`ends`, and insignificant punctuation spacing. This changes only how
saved predictions and gold answers are compared; it does not change prediction
text. The implementation is in `src/evaluate.py`.

## Adaptive Routing

The adaptive router uses the entropy-derived effective evidence count `N_eff`.
Its selected policy is:

```text
N_eff < 6  -> Direct
N_eff >= 6 -> TISER
```

The comparison baseline routes using tokenized temporal-context length with a
threshold of 163 tokens.

| Method | Macro EM | Macro F1 | Avg. generated tokens | Total generated tokens | Direct | TISER | Token saving vs. Always-TISER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Context length, threshold 163 | 83.31 | 88.79 | 141.56 | 2,703,997 | 53.28% | 46.72% | 47.96% |
| TF-IDF router, threshold 6 | 84.36 | 89.56 | 135.87 | 2,595,310 | 54.89% | 45.11% | 50.05% |

At a similar Direct/TISER routing ratio, the TF-IDF router has higher Macro EM
and Macro F1 while using fewer generated tokens than the context-length
baseline.

## Repository Structure

```text
src/
  dataset.py                 TISER loading and 2,048-token test filtering
  model.py                   Cached tokenizer and Qwen model loading
  prompts.py                 Direct and TISER prompt selection
  train_lora.py              LoRA supervised fine-tuning and validation
  evaluate.py                Inference, extraction, metrics, summaries, rescoring

notebooks/
  run_train_lora.ipynb       Colab LoRA training workflow
  run_test_lora.ipynb        Colab combined Direct/TISER evaluation workflow
  Extension_1.ipynb          Adaptive-routing analysis workflow

extensions/adaptive_routing/
  tfidf_analysis.py          TF-IDF evidence-distribution features
  tfidf_validation.py        Quartile and correlation analysis
  tfidf_router.py            Effective-evidence threshold sweep
  context_length_baseline.py Context-length threshold sweep
```

## Installation

```bash
git clone https://github.com/EliasNoorzad/temporal-reasoning-TISER.git
cd temporal-reasoning-TISER
python -m pip install -r requirements.txt
```

Model and dataset files use the standard Hugging Face cache; they are not
manually downloaded by the source scripts.

## Training and Evaluation

Inspect the official dataset:

```bash
python src/dataset.py --sample-rows 3
```

The Colab training notebook runs LoRA SFT with:

```bash
python -m src.train_lora \
  --validation-ratio 0.1 \
  --num-train-epochs 2
```

`--output-dir` makes checkpoint storage configurable. The published adapter
stores the selected weights in the `checkpoint-49040` subfolder, so it can be
loaded directly from its Hugging Face repository for combined Direct and TISER
evaluation:

```bash
python src/evaluate.py \
  --model-type lora \
  --prompt-type both \
  --lora-adapter-path EliElias/TISER-Qwen2.5-3B-LoRA \
  --output-dir evaluation_outputs \
  --batch-size 16
```

PEFT loads the published adapter from Hugging Face and attaches it to the same
`Qwen/Qwen2.5-3B-Instruct` base model. `notebooks/run_test_lora.ipynb` uses
temporary Colab storage with the public Hugging Face checkpoint and saved
evaluation outputs. `notebooks/run_train_lora.ipynb` records the original
training run, including its historical Google Drive output paths; current
training defaults to the local `checkpoints/train_lora` path. Private Drive
access is not required for reproduction.

## Rescoring Existing Predictions

Existing combined predictions can be rescored without loading a model or
running inference. The implemented interface is:

```bash
python src/evaluate.py \
  --rescore-existing <raw_results.jsonl> \
  --output-rescored-results <rescored_results.jsonl> \
  --output-summary <summary.json>
```

For the public result filenames, the corresponding command is:

```bash
python src/evaluate.py \
  --rescore-existing lora_both_results.jsonl \
  --output-rescored-results lora_both_results_rescored.jsonl \
  --output-summary lora_both_summary_rescored.json
```

This reads the original saved predictions, applies the final deterministic
normalization, recomputes Direct and TISER EM/F1, and writes a new JSONL plus
summary. It preserves the original generated answers and does not perform model
generation.

The rescored JSONL can be used directly by the adaptive-routing workflow:

```bash
python extensions/adaptive_routing/tfidf_analysis.py \
  --input lora_both_results_rescored.jsonl \
  --output tfidf_analysis.jsonl \
  --summary-output tfidf_analysis_summary.csv

python extensions/adaptive_routing/tfidf_validation.py \
  --input tfidf_analysis.jsonl \
  --output-dir tfidf_validation

python extensions/adaptive_routing/tfidf_router.py \
  --input tfidf_analysis.jsonl \
  --output tfidf_router_sweep.csv

python extensions/adaptive_routing/context_length_baseline.py \
  --input tfidf_analysis.jsonl \
  --output context_length_baseline.csv
```

See `notebooks/Extension_1.ipynb` and
`extensions/adaptive_routing/README.md` for the exploratory validation workflow.
