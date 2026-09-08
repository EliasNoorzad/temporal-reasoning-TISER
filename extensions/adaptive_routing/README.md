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

## Usage

```bash
python extensions/adaptive_routing/tfidf_analysis.py \
  --input results/lora_both_results.jsonl \
  --output results/tfidf_analysis.jsonl \
  --summary-output results/tfidf_analysis_summary.csv

python extensions/adaptive_routing/tfidf_validation.py \
  --input results/tfidf_analysis.jsonl \
  --output-dir results/tfidf_validation

python extensions/adaptive_routing/tfidf_router.py \
  --input results/tfidf_analysis.jsonl \
  --output results/tfidf_router_thresholds.csv

python extensions/adaptive_routing/context_length_baseline.py \
  --input results/tfidf_analysis.jsonl \
  --output results/context_length_thresholds.csv
```

## Final Comparison

| Method | Macro EM | Macro F1 | Generated-token saving vs Always TISER |
| --- | ---: | ---: | ---: |
| Always Direct | 79.07 | 85.54 | - |
| Always TISER | 86.14 | 90.73 | 0.00% |
| Context length, threshold 163 | 83.31 | 88.79 | 47.96% |
| TF-IDF router, threshold 6 | 84.36 | 89.56 | 50.05% |

At a similar Direct/TISER routing ratio, the TF-IDF router provides a better
answer-quality and generated-token trade-off than the context-length baseline.
