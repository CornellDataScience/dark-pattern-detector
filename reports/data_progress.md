# Data progress — Task 1

## Source
Yuki Yada et al., *Dark patterns in e-commerce: a dataset and its baseline evaluations* (IEEE BigData 2022). Official dataset: https://github.com/yamanalab/ec-darkpattern/blob/master/dataset/dataset.tsv. The repository indicates an Apache-2.0 license; preserve credit and check project redistribution rules before publishing copies. The corpus pairs positive dark-pattern text from Mathur et al. (2019) with non-dark-pattern website text collected by Yada et al.

## What was implemented
- `scripts/prepare_real_dataset_sample.py` generates `examples/baseline_sample.jsonl` from the upstream **real labeled** TSV.
- Output schema: JSON Lines; each record has `id`, `text`, `label`, `source`.
- Labels are preserved from the upstream TSV: `1` = dark pattern, `0` = not dark pattern.
- It selects 6 positives and 6 negatives, deduplicates exact normalized text and favors multiple positive pattern categories.
- The file contains source-labeled examples, **not independent verification** that the original annotations are correct. Some labels depend on context, which a text-only prototype may not provide.
- The full downloaded TSV goes to `dataset/ec-darkpattern/dataset.tsv` and should remain local. If the team wants even the 12-row sample kept local, do not commit `examples/baseline_sample.jsonl` either.

## Run from repository root
```
python scripts/prepare_real_dataset_sample.py
python -c "import json; p='examples/baseline_sample.jsonl'; a=[json.loads(s) for s in open(p,encoding='utf-8')]; print(len(a),{i:sum(r['label']==i for r in a) for i in (0,1)})"
```

## Limitations and next steps
Twelve records are for a **smoke test only**, not an accurate model benchmark. For full evaluation use grouped train/test splits to avoid duplicate or page-level leakage, review ambiguous labels, and separately collect screenshots and bounding boxes for the multimodal tasks. There are no screenshots in this particular text-only dataset.

## External references
- Official dataset/code: https://github.com/yamanalab/ec-darkpattern
- Original positive corpus: https://github.com/aruneshmathur/dark-patterns
- Yada et al. paper: https://arxiv.org/abs/2211.06543
