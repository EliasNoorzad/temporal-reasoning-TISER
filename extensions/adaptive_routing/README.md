# Adaptive Routing

This extension evaluates whether each temporal question should use the short
Direct generation path or the more expensive TISER reasoning path. The router
uses a lightweight structural signal to preserve most of TISER's answer quality
while reducing generated tokens.

## TF-IDF Signal

`tfidf_analysis.py` splits each temporal context into sentences and computes
TF-IDF similarity between the question and every context sentence. The
similarities are normalized into a distribution, and its entropy is converted
to an effective evidence count, `N_eff`.

A low `N_eff` means that relevance is concentrated in a small number of
sentences. A high value means that relevant evidence is distributed across more
of the context. The selected routing rule is:

```text
N_eff < tau  -> Direct
N_eff >= tau -> TISER
```

The TF-IDF operating point uses `tau = 6`. The comparison baseline routes by
tokenized temporal-context length and uses threshold `163`.

## Evaluation Protocol

Answer quality follows the TISER paper's five-dataset macro protocol. EM and F1
are calculated independently for each dataset and then averaged equally across:

- `tgqa_test`
- `tempreason_l2_test`
- `tempreason_l3_test`
- `timeqa_easy_test`
- `timeqa_hard_test`

`tot_semantic_test` may remain in the feature file, but it is excluded from the
in-domain routing evaluation. Token totals, routing percentages, coverage, and
rescue statistics are calculated over all 19,102 retained examples.

## Scripts

- `tfidf_analysis.py` computes TF-IDF evidence-dispersion features for each
  evaluation record. It intentionally remains independent of the evaluation
  dataset filtering.
- `tfidf_validation.py` examines the two TF-IDF signals with complexity
  quartiles and Spearman correlations on the five in-domain datasets.
- `tfidf_router.py` sweeps the fixed evidence-count thresholds and reports the
  selected `tau = 6` operating point.
- `context_length_baseline.py` sweeps context-token thresholds and reports the
  threshold `163` comparison point.

## Reproducible Workflow

The analysis starts from `lora_both_results_rescored.jsonl` in the public
[TISER evaluation-results dataset](https://huggingface.co/datasets/EliElias/TISER-Evaluation-Results).
This file contains the original saved Direct and TISER predictions with EM/F1
recomputed using the final deterministic normalization in `src/evaluate.py`.
No inference is rerun during rescoring.

Download the file with `huggingface_hub`:

```python
from huggingface_hub import hf_hub_download

input_path = hf_hub_download(
    repo_id="EliElias/TISER-Evaluation-Results",
    filename="lora_both_results_rescored.jsonl",
    repo_type="dataset",
)
print(input_path)
```

Pass the downloaded path to the analysis scripts. The following commands write
working outputs to `extension_1/`; each script creates its required output
directory.

```bash
python extensions/adaptive_routing/tfidf_analysis.py \
  --input <downloaded_file_path> \
  --output extension_1/tfidf_analysis.jsonl \
  --summary-output extension_1/tfidf_analysis_summary.csv

python extensions/adaptive_routing/tfidf_validation.py \
  --input extension_1/tfidf_analysis.jsonl \
  --output-dir extension_1

python extensions/adaptive_routing/tfidf_router.py \
  --input extension_1/tfidf_analysis.jsonl \
  --output extension_1/tfidf_router_sweep.csv

python extensions/adaptive_routing/context_length_baseline.py \
  --input extension_1/tfidf_analysis.jsonl \
  --output extension_1/context_length_baseline.csv
```

`notebooks/Extension_1.ipynb` follows this workflow using temporary
`/content/extension_1` storage in Colab. Its final upload step publishes the
generated files under `extension_1/` in the same Hugging Face
evaluation-results dataset.

## Final Comparison

| Method | Macro EM | Macro F1 | Avg. generated tokens | Total generated tokens | Direct | TISER | Token saving vs. Always TISER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Always Direct | 79.07 | 85.54 | 5.93 | 113,257 | 100.00% | 0.00% | - |
| Always TISER | 86.14 | 90.73 | 272.00 | 5,195,747 | 0.00% | 100.00% | 0.00% |
| Context length, threshold 163 | 83.31 | 88.79 | 141.56 | 2,703,997 | 53.28% | 46.72% | 47.96% |
| TF-IDF router, threshold 6 | 84.36 | 89.56 | 135.87 | 2,595,310 | 54.89% | 45.11% | 50.05% |

At a similar Direct/TISER routing ratio, the TF-IDF router provides a better
answer-quality and generated-token trade-off than the context-length baseline.
