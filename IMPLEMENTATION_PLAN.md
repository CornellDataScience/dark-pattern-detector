# Four-owner, one-hour work plan

## Target for the next update

Each owner spends **one hour**, working in parallel. Deliver a small text baseline and a minimal multimodal computation graph. This is an implementation milestone; meaningful model evaluation comes later.

Assumption: the visual branch processes static screenshots, not videos. Use binary classification for this prototype: `1` means a dark-pattern example and `0` means a negative example. Multi-label outputs are deferred.

## Shared contracts — use these without waiting for another owner

### Baseline sample

One JSON object per line in `examples/baseline_sample.jsonl`:

```json
{"id":"synthetic:1","text":"Only two left — buy now!","label":1,"source":"synthetic"}
```

Use `source: "manual_transcription"` and the original screenshot ID for real examples. Synthetic labels are test fixtures, not validated claims about an actual interface.

### Model interfaces

- `UIEncoder.forward(text_features, boxes)` accepts `[B,N,384]` text features and `[B,N,4]` normalized `[x1,y1,x2,y2]` boxes; returns `[B,N,256]` UI tokens.
- `FusionClassifier.forward(ui_tokens, image_features)` accepts `[B,N,256]` UI tokens and `[B,512]` full-screenshot features; returns binary logits `[B,1]`.
- Use `B=2`, `N=4` for demonstrations. All elements are valid; variable-length padding is deferred.
- Random features belong only in demonstration code, never inside a model's `forward()` method.

## Owner 1 — sample data and label inspection

**Owns:** `examples/baseline_sample.jsonl`, `reports/data_progress.md`.

### Hour allocation

- **0–10 min:** Inspect available ContextDP files or its [web annotations](https://github.com/SageSELab/AidUI/blob/master/evaluation/evaluation_dataset/web/result.json). Map category IDs through the category-name table.
- **10–40 min:** Prepare 8–12 short text examples, with at least four examples of each binary label. Prefer transcribing visible text from accessible screenshots. Use positive annotations for positive examples and verified `NO DP`-only screenshots for negatives; skip missing or conflicting labels. Preserve screenshot IDs. If data access consumes more than ten minutes, prepare clearly marked synthetic fixtures instead and record the access blocker.
- **40–60 min:** Check JSON validity, IDs, label values, and class counts. Send Owner 2 the file location as soon as it is ready. Write a short note identifying provenance, counts, and any skipped examples.

### Finished when

- [ ] JSONL loads successfully and contains 8–12 examples, at least four per label.
- [ ] Each record has the four fields in the shared contract and a unique ID.
- [ ] Real examples are traceable to screenshots; synthetic examples are clearly marked.
- [ ] Owner 2 has the file location and the provenance/count note.

**Update artifact:** the sample and a brief explanation of its labels. If synthetic, state that real-data preparation remains pending.

## Owner 2 — baseline code

**Owns:** `baseline.py`. Do not edit Owner 1's sample file.

### Hour allocation

- **0–10 min:** Implement JSONL loading against the contract. Use a temporary in-memory synthetic fixture while Owner 1 prepares data.
- **10–40 min:** Implement scikit-learn TF-IDF word unigrams/bigrams and logistic regression (`max_iter=1000`). Use a stratified 75/25 development split with seed 42. Fit the vectorizer on training text only. Print held-out texts, true labels, and predicted labels.
- **40–60 min:** Run against Owner 1's sample if ready; otherwise demonstrate the temporary fixture and report the pending handoff. Add dependencies and the run command in the module docstring.

### Finished when

- [ ] `python baseline.py --data examples/baseline_sample.jsonl` loads the agreed format and runs training/prediction when that file is available.
- [ ] A demonstrated run completes on a labeled sample with both labels in training and held-out partitions.
- [ ] TF-IDF is fitted only on training text.
- [ ] The docstring gives dependencies, command, and whether the demonstrated run used real or synthetic data.

**Update artifact:** executable script and held-out predictions. These tiny-sample predictions are a pipeline check, not a performance benchmark. Report integration with Owner 1 separately if it remains pending.

## Owner 3 — UI encoder module

**Owns:** `ui_encoder.py`.

### Hour allocation

- **0–10 min:** Define `UIEncoder` with the shared forward signature. Intended future inputs are MiniLM text features and OmniParser boxes; no parser or pretrained weights are needed today.
- **10–35 min:** Concatenate text features and boxes, then apply Linear `388 → 256` and GELU. Return one token per UI element.
- **35–60 min:** Add a runnable demonstration using random text features and valid normalized boxes with `B=2`, `N=4`. Assert output shape `[2,4,256]` and finite values; backpropagate a scalar loss and confirm finite, nonzero projection gradients. Document dependencies and the interface in a docstring.

### Finished when

- [ ] `python ui_encoder.py` passes the shape, finite-value, and gradient checks.
- [ ] The forward method uses supplied tensors and preserves batch/element dimensions.
- [ ] Owner 4 has the module name and import: `from ui_encoder import UIEncoder`.
- [ ] The docstring identifies OmniParser/MiniLM integration as future work.

**Update artifact:** UI encoder code, output shape, and a short explanation of how text and geometry combine.

## Owner 4 — visual projection and fusion classifier

**Owns:** `fusion_model.py`.

### Hour allocation

- **0–10 min:** Define `FusionClassifier` with the shared signature. The `[B,512]` image feature is intended to come from frozen CLIP later.
- **10–40 min:** Project the image feature with Linear `512 → 256`; prepend it to UI tokens. Apply one transformer encoder layer with hidden size 256, eight attention heads, feed-forward size 512, and `batch_first=True`. Apply Linear `256 → 1` to the output image token.
- **40–60 min:** Demonstrate forward and backward passes with random UI/image features. If Owner 3's module is ready, also pass its output into the classifier; otherwise retain the exact shared interface and report integration as pending. Document dependencies and the command in a docstring.

### Finished when

- [ ] `python fusion_model.py` returns finite logits of shape `[2,1]`.
- [ ] `BCEWithLogitsLoss` on two binary targets is finite and backward produces finite, nonzero classifier gradients.
- [ ] The model accepts UI tokens from Owner 3 without reshaping or changing their contract.
- [ ] The demonstration clearly identifies synthetic features and whether the UI module was connected.

**Update artifact:** visual/fusion architecture and runnable computation graph. This is not yet a trained detector.

## Update and handoff

Each owner shows their file, the check they ran, and one next step. Owner 1 hands data to Owner 2; Owner 3 hands UI tokens to Owner 4. Owners 2 and 4 can develop with fixtures immediately. Complete either integration within the hour if both sides are ready; otherwise list it explicitly as the next task rather than claiming an end-to-end result.

## Deferred

Full dataset download/audit, OCR and OmniParser setup, real MiniLM/CLIP extraction, element crops, padding masks, multi-label classification, full model training, checkpoints, meaningful evaluation, localization, and a frontend.
