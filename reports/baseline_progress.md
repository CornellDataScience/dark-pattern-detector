# Task 2 — text baseline

Completed 2026-11-14. [`baseline.py`](../baseline.py) runs on Owner 1's eight real
ContextDP transcriptions; the data-to-baseline handoff is complete.

```sh
uv sync --locked
uv run python baseline.py --data examples/baseline_sample.jsonl
uv run python -m unittest discover -s tests -v
```

Dependencies are in the shared `pyproject.toml` and `uv.lock`. The baseline uses
scikit-learn's `TfidfVectorizer(ngram_range=(1, 2))` and
`LogisticRegression(max_iter=1000, random_state=42)` in a pipeline. A stratified
75/25 split with seed 42 happens before fitting. Only the six training texts
contribute to the vocabulary and inverse document frequencies.

## Observed run

Training IDs, in order: `web:3`, `web:12`, `web:4`, `web:7`, `web:11`, `web:14`.
Training has three examples of each label; held-out data has one of each.

| Held-out ID | True label | Predicted label |
| --- | --- | --- |
| web:13 | 0 | 1 |
| web:6 | 1 | 1 |

The CLI prints the complete held-out text beside each true and predicted label,
plus sample provenance, split IDs, counts, and seed. It does not report a benchmark
score. These two predictions only establish that the pipeline executes.

Three automated tests passed: real sample and stratification checks; vocabulary
leakage and repeatability checks; and invalid-input cases including duplicate IDs,
missing fields, blank text, noninteger labels, unknown provenance, malformed JSON,
and insufficient examples per class.

Next: use consistently extracted text and a larger reviewed, source-separated
dataset before interpreting detection performance.

## Full-corpus evaluation — 2026-11-14

Merged `main` and `data-real-examples`, keeping the eight ContextDP records at their
existing path and the incoming 12 Yada et al. records at
`examples/ec_darkpattern_sample.jsonl`. The loader accepts the latter's dataset URL
as provenance. Both sample commands run; the experiment below uses the full TSV.

### Reproduce

```sh
uv sync --locked
uv run python scripts/prepare_real_dataset_sample.py
uv run python baseline.py --ec-darkpattern dataset/ec-darkpattern/dataset.tsv \
  --output dataset/ec-darkpattern/baseline_result.json
uv run python -m unittest discover -s tests -v
```

The download is pinned to `yamanalab/ec-darkpattern` commit
`85ea9df31928d5ad6ea0b106316b441c201adbbe`. Dataset SHA-256:
`bd7bc7d93f5bf08591b10886d7aa3a7ba8ac266f5ce8206c00cd4daf7868508a`.
Observed runtime: Python 3.11.15 and scikit-learn 1.9.1, on local CPU.

### Data and fixed experiment

- Source: 2,356 labeled rows, 1,178 per class, 1,248 page IDs. All text lengths and
  categories are eligible. Labels and label/category consistency are validated.
- Normalize whitespace; identify duplicates by case-folded text. Retain the first
  occurrence of each text. Remove 98 duplicate rows, retaining 2,258 unique texts:
  1,167 negative and 1,091 positive. No conflicting duplicate labels were found.
- Before deduplication, connect pages sharing identical normalized text, including
  transitive connections. Keep each of the resulting 1,155 groups in one partition.
- Use the first fold of `StratifiedGroupKFold(n_splits=4, shuffle=True,
  random_state=42)`. This is one fixed approximately 75/25 holdout, not four-fold
  cross-validation. No parameter search or threshold tuning was performed.
- Train: 1,693 texts (875 negative, 818 positive), covering 931 original page IDs.
  Holdout: 565 texts (292 negative, 273 positive), covering 317 original page IDs.
  Verified overlap: **zero page IDs and zero normalized texts**.
- TF-IDF word unigrams/bigrams, default lowercase/tokenization and L2 normalization;
  fit only on training text. Vocabulary: 11,040 features. Logistic regression:
  `C=1.0`, `max_iter=1000`, `random_state=42`, default L2 regularization and
  no class weighting. Binary decision threshold: 0.5.

### Held-out results

| Metric | TF-IDF + logistic regression | Majority-class baseline |
| --- | ---: | ---: |
| Accuracy | 0.9416 | 0.5168 |
| Macro-F1 | 0.9414 | 0.3407 |
| Positive-class F1 | 0.9376 | 0.0000 |
| Positive-class precision | 0.9688 | 0.0000 |
| Positive-class recall | 0.9084 | 0.0000 |
| AUROC | 0.9832 | 0.5000 |

Confusion matrix (rows = true label, columns = predicted label):

| | Predicted 0 | Predicted 1 |
| --- | ---: | ---: |
| True 0 | 284 | 8 |
| True 1 | 25 | 248 |

The 25 missed positives include 14 Misdirection examples, four Social Proof, four
Urgency, two Scarcity, and one Sneaking. Recall by positive category:

| Category | Detected / held-out positives | Recall |
| --- | ---: | ---: |
| Scarcity | 95 / 97 | 0.9794 |
| Social Proof | 64 / 68 | 0.9412 |
| Urgency | 50 / 54 | 0.9259 |
| Misdirection | 35 / 49 | 0.7143 |
| Obstruction | 4 / 4 | 1.0000 |
| Sneaking | 0 / 1 | 0.0000 |
| Forced Action | 0 / 0 | Not evaluated |

The complete local JSON includes split IDs, page IDs, dataset audit, versions,
model settings, metrics, texts, predictions, and probabilities. Raw data and these
full artifacts are ignored by Git. All 18 automated tests pass, including the
merged sample, transitive duplicate-page grouping, training-only vocabulary,
deterministic splitting, conflicting duplicate labels, and metric calculations.

### Interpretation

This establishes a text-only baseline on the source annotations. The split groups
the supplied `page_id` values; the TSV has no domain column, so domain separation
is not established. Near-duplicate templates can remain across pages. Results are
from one holdout, and rare categories have little or no coverage. Misdirection is
the largest observed source of false negatives. Screenshot-based multimodal
performance still requires a separate training and evaluation run on ContextDP.
