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
