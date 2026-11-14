# Task 1 — sample data and label inspection

Completed 2026-11-14. [`examples/baseline_sample.jsonl`](../examples/baseline_sample.jsonl)
contains eight real, manually transcribed examples: four positive and four negative.
All eight screenshots were opened and visually checked. Line breaks were replaced
with spaces; text is a short excerpt, not a full screenshot transcription. The fire
emoji following “Limited stock!” was omitted. No synthetic examples were needed.

## Provenance and labels

Source: ContextDP's [web annotations](https://github.com/SageSELab/AidUI/blob/bb97865abf6321ff2564f064da8b48ac0a82f284/evaluation/evaluation_dataset/web/result.json)
at AidUI commit `bb97865abf6321ff2564f064da8b48ac0a82f284`.
The existing local `dataset/web/bounding-boxes.json` is JSON-equivalent to upstream
`result.json`; its SHA-256 is
`94e0e1f2a45f88a5ddc31f254af642bf13b45c9ad18c6823a3853f3d2aa8618d`.
Screenshots reside under `dataset/web/images/` and are excluded from Git.

Category IDs come from this annotation file, not a guessed ordering:

| ID | Category |
| --- | --- |
| 0 | ACTIVITY MESSAGE |
| 1 | COUNTDOWN TIMER |
| 2 | LIMITED TIME MESSAGE |
| 3 | LOW STOCK MESSAGE |
| 4 | NO DP |
| 5 | HIGH DEMAND MESSAGE |
| 6 | DEFAULT CHOICE |
| 7 | ATTENTION DISTRACTION |

`web:N` preserves the annotation image ID. Label `1` means the source screenshot
has positive annotations and no `NO DP` annotation; `0` requires a nonempty set of
annotations containing only `NO DP`. These are inherited dataset labels, not newly
verified claims about the sites' behavior. Positive excerpts include the annotated
cue; negative excerpts contain visible product, store, or navigation text.

| Sample | Screenshot filename | Category IDs | Label |
| --- | --- | --- | --- |
| web:3 | beemission.com.png | 1, 2 | 1 |
| web:4 | justhype.com.png | 1, 2 | 1 |
| web:6 | knockaround.com.png | 0, 3 | 1 |
| web:7 | shapermint.com.png | 1, 2 | 1 |
| web:11 | groupon.com.png | 4 | 0 |
| web:12 | 2ndsworld.com.au.png | 4 | 0 |
| web:13 | 4wdsupacentre.com.au.png | 4 | 0 |
| web:14 | aptdeco.com.png | 4 | 0 |

To retrieve an original screenshot, substitute its filename in
`https://raw.githubusercontent.com/SageSELab/AidUI/bb97865abf6321ff2564f064da8b48ac0a82f284/evaluation/evaluation_dataset/web/images/FILENAME`.
All eight files were already accessible locally. Inspection of the 162 web image
records and 185 annotations found no unannotated images or mixed positive/`NO DP`
image IDs. No candidate was skipped for missing or conflicting labels; the other
154 images were outside this deliberately small selection. Mobile data was not
audited for this milestone.

## Validation and handoff

JSON parsing, exact four-field records, nonempty text, unique IDs, binary integer
labels, and class counts were checked. Each selected ID was joined back to the
annotation table and its file was confirmed to exist. Owner 2's input is
`examples/baseline_sample.jsonl`; provenance is `manual_transcription` for all rows.
The baseline's reproducible command is
`uv run python baseline.py --data examples/baseline_sample.jsonl`.

This hand-selected sample checks data loading and training. Its positive excerpts
focus on cues while negatives cover broader interface text, so it cannot support
performance claims. The next data step is consistent text extraction and
source-grouped splits across a larger reviewed sample.

## Merged e-commerce text dataset — 2026-11-14

The `data-real-examples` branch contributes Yuki Yada et al.'s *Dark patterns in
e-commerce: a dataset and its baseline evaluations* (IEEE BigData 2022).
The [official corpus](https://github.com/yamanalab/ec-darkpattern) combines positive
text from Mathur et al. (2019) with collected non-dark-pattern website text. Its
repository identifies an Apache-2.0 license. These are inherited source labels;
the text corpus does not supply screenshots or bounding boxes.

Both branches' samples are preserved separately:

- `examples/baseline_sample.jsonl`: the eight ContextDP transcriptions above.
- `examples/ec_darkpattern_sample.jsonl`: the incoming 12 text records, six per label.
  Their `source` is the official dataset URL. The sample favors positive category
  diversity and restricts text length to 12–180 characters, for a pipeline check.

`uv run python scripts/prepare_real_dataset_sample.py` downloads the full TSV to
`dataset/ec-darkpattern/dataset.tsv` if absent and regenerates the e-commerce sample
with seed 2026. The download is pinned to upstream commit
`85ea9df31928d5ad6ea0b106316b441c201adbbe`. Regeneration exactly matches the merged
12-row sample. Raw data and full result artifacts remain under ignored `dataset/`.

The full TSV contains 2,356 rows (1,178 per label) from 1,248 `page_id` values. Its
SHA-256 is `bd7bc7d93f5bf08591b10886d7aa3a7ba8ac266f5ce8206c00cd4daf7868508a`.
The full baseline applies no sample-length or category-selection filter. Whitespace
normalization and case folding identify 98 duplicate rows, leaving 2,258 texts
(1,091 positive, 1,167 negative); there are no blank texts or conflicting duplicate
labels. Pages connected by duplicate text form 1,155 groups kept intact across the
evaluation split. See [the full results](baseline_progress.md#full-corpus-evaluation--2026-11-14).
