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

- Owner 3 uses OmniParser directly and exports UI elements as JSON: `{"sample_id":"web:0","elements":[{"text":"Example","bbox":[0.1,0.2,0.4,0.3]}]}`. Boxes use normalized `[x1,y1,x2,y2]` coordinates. Document whether text is OCR output or a generated description.
- OmniParser output is structured text/boxes, not the dense UI tokens required by fusion. Converting these records into embeddings is a later integration task. Owner 4 uses synthetic UI tokens for this hour.
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

## Owner 3 — UI parsing with OmniParser

**Owns:** `parse_ui.py`, `reports/omniparser_progress.md`. Keep screenshots, weights, and generated outputs under the ignored `dataset/` directory.

Use [Microsoft OmniParser](https://github.com/microsoft/OmniParser) as the UI component. No custom UI encoder or MiniLM integration is required for this hour.

### Hour allocation

- **0–25 min:** Follow the upstream setup instructions and obtain the required weights. Record the code version, dependencies, hardware, and setup command in the progress note. Use a separate environment if upstream requirements conflict with the shared prototype environment.
- **25–45 min:** Run the upstream example on one real ContextDP screenshot. Save its native output and annotated image locally. Reuse the upstream inference code through a small `parse_ui.py` wrapper where practical.
- **45–60 min:** Export text/content and normalized boxes using the shared JSON format. Compare several boxes and text entries against the screenshot, and document the exact command and output location.

### Finished when

- [ ] OmniParser successfully processes one real screenshot.
- [ ] A saved JSON record contains element text/content and normalized bounding boxes, with the screenshot ID and text provenance documented.
- [ ] Several extracted elements have been visually checked against the screenshot.
- [ ] The progress note contains reproducible commands, versions, and local output locations for Owner 4.

**Time limit:** Stop after one hour. If installation or model downloads block inference, record the exact blocker and next command to try. Report setup progress rather than claiming successful parsing. A custom encoder is not a substitute for this task.

**Update artifact:** parsed screenshot and element output, or a concrete setup-status report if blocked.

## Owner 4 — CLIP screenshot encoder and fusion classifier

**Owns:** `fusion_model.py`, `reports/visual_encoder_progress.md`.

### Encoder choice

Use **OpenAI CLIP ViT-B/32**, checkpoint **`openai/clip-vit-base-patch32`**, through Hugging Face Transformers. This branch encodes the entire screenshot.

- Load `CLIPImageProcessor` and `CLIPVisionModelWithProjection` from that checkpoint. Use the checkpoint's image processor for RGB conversion/input preparation, resizing, cropping, and normalization; do not implement custom preprocessing.
- Pass the processed screenshot into the vision model and use its `image_embeds` output: **512 values per screenshot**, shape `[B,512]`. Do not use the 768-dimensional vision hidden states as the fusion input.
- Freeze all CLIP parameters, keep the encoder in evaluation mode, and extract features under `torch.no_grad()`. Only the projection, transformer, and classifier are trainable.
- Preserve the screenshot's original file. Standard CLIP preprocessing can crop parts of a wide or tall UI; note this limitation in the progress report. Alternative resizing policies are deferred.
- Add `transformers` and `pillow` to the shared environment with `uv add transformers pillow`, updating `pyproject.toml` and `uv.lock`. Owner 4 owns these dependency changes for this task. Record the resolved checkpoint revision and package versions.

### Hour allocation

- **0–15 min:** Install dependencies and load the processor and frozen CLIP vision encoder. Cache downloaded weights outside Git.
- **15–30 min:** Load one real screenshot with Pillow, convert it to RGB, process it, and obtain a finite `[1,512]` feature vector. Keep feature extraction in a helper separate from `FusionClassifier.forward()`.
- **30–50 min:** Implement `FusionClassifier`: project the 512-dimensional image feature to 256 dimensions, prepend it to UI tokens, then apply one transformer encoder layer with hidden size 256, eight attention heads, feed-forward size 512, and `batch_first=True`. Apply Linear `256 → 1` to the output image token. Use synthetic `[1,4,256]` UI tokens while the OmniParser-to-embedding adapter is pending.
- **50–60 min:** Run the real image feature through fusion with synthetic UI tokens. Compute `BCEWithLogitsLoss` using a demonstration target and call backward. Record the command, shapes, and results in the progress note. The target exercises the computation graph; it is not a meaningful training experiment.

### Finished when

- [x] `uv run python fusion_model.py --image PATH` extracts real CLIP features from a screenshot and returns finite logits of shape `[1,1]`.
- [x] The CLIP feature shape is `[1,512]`, and the checkpoint processor is used.
- [x] Backward produces finite, nonzero classifier gradients; CLIP parameters remain frozen with no gradients.
- [x] The classifier accepts the documented `[B,N,256]` UI-token interface. Synthetic UI tokens and the pending OmniParser embedding step are explicitly identified.
- [x] The progress note records dependencies, checkpoint revision, command, and observed tensor shapes.

Completed 2026-11-13; see [observed results and reproduction commands](reports/visual_encoder_progress.md).

**Time limit:** If weight downloads or setup block progress, finish the fusion module with synthetic image/UI features and record the blocker. Report fusion-skeleton completion separately from successful real-image encoding.

**Update artifact:** the named CLIP checkpoint, a real screenshot's feature shape, and the fusion forward/backward output, or the specific setup blocker.

## Update and handoff

Each owner shows their file, the check they ran, and one next step. Owner 1 hands data to Owner 2; Owner 3 hands parsed JSON and its schema to Owner 4; embedding and fusion integration follow later. Owners 2 and 4 can develop with fixtures immediately. Complete the data-to-baseline integration within the hour if both sides are ready; otherwise list it explicitly as the next task rather than claiming an end-to-end result.

## Deferred

Full dataset download/audit, dataset-wide OmniParser processing, conversion of parsed elements into UI embeddings, dataset-wide CLIP feature extraction, element crops, padding masks, multi-label classification, full model training, checkpoints, meaningful evaluation, localization, and a frontend.
