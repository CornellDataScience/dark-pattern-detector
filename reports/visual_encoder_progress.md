# Task 4 — CLIP screenshot encoder and fusion classifier

Completed 2026-11-13. A real ContextDP screenshot passed CLIP extraction and fusion forward/backward checks on CPU. Implementation: [`fusion_model.py`](../fusion_model.py).

## Checkpoint and environment

- Checkpoint: [OpenAI CLIP ViT-B/32](https://huggingface.co/openai/clip-vit-base-patch32).
- Resolved revision: `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`, pinned for both `CLIPImageProcessor` and `CLIPVisionModelWithProjection`.
- Hardware/runtime: macOS 27.2, Apple Silicon (`arm64`), CPU, Python 3.11.15.
- Weights cached outside Git at `/Users/spencerliang/.cache/huggingface/hub/models--openai--clip-vit-base-patch32/`.

| Package | Resolved version |
| --- | --- |
| torch | 2.15.0 |
| transformers | 5.23.0 |
| pillow | 13.0.0 |
| huggingface-hub | 2.4.0 |
| socksio | 1.0.0 |

Dependencies were added with `uv add transformers pillow` and `uv add socksio`; `pyproject.toml` and `uv.lock` are updated. The Hugging Face HTTP client initially failed because this machine sets a SOCKS `ALL_PROXY` and lacked `socksio`. Installing that dependency fixed the standard command below.

Transformers uses its Pillow-backed checkpoint image processor when torchvision is absent. Loading the vision-only model from the full CLIP checkpoint reports unused text-model weights; no vision weights were reported missing.

## Reproduce

From the repository root:

```sh
uv sync --locked
uv run python fusion_model.py --image dataset/web/images/knockaround.com.png
```

The screenshot was already available locally under the ignored `dataset/` directory. Its metadata in `dataset/web/bounding-boxes.json` identifies it as image ID `6` (`web:6`), with dimensions 1700 × 919 and filename `images/knockaround.com.png`. Another local screenshot can be supplied through `--image`.

The source image remains unchanged. SHA-256: `13f06cc86e3bba311e6db73d1788e90a474b7eb581a6615b2d50261b9923bbe2`.

## Implementation and observed results

`load_clip_encoder()` freezes every CLIP parameter and sets evaluation mode. `extract_image_features()` reads the screenshot with Pillow, converts it to RGB, uses checkpoint preprocessing, and returns `image_embeds` under `torch.no_grad()`. The helper is separate from fusion.

`FusionClassifier.forward(ui_tokens, image_features)` projects image features from 512 to 256 dimensions and prepends that image token to the UI tokens. One transformer encoder layer uses hidden size 256, eight attention heads, feed-forward size 512, and `batch_first=True`. A linear `256 → 1` head reads the output image token. Only these fusion components are trainable.

The demonstration seeds PyTorch with 42 after CLIP loading and uses four synthetic UI tokens. The target `1.0` exercises `BCEWithLogitsLoss` and backward; it is a demonstration value, not a label inferred from the screenshot.

| Check | Observed result |
| --- | --- |
| Real CLIP image features | `[1,512]`, finite, no gradient tracking |
| Synthetic UI tokens | `[1,4,256]` |
| Fusion logits | `[1,1]`, finite; `-0.7684860229` |
| BCE loss | `1.1494634151` |
| Classifier weight gradient L2 norm | `10.9310417175`, finite and nonzero |
| All fusion parameter gradients | Present and finite |
| CLIP after backward | Evaluation mode; all parameters frozen; all gradients `None` |

The CLI checks these conditions before printing `checks_passed: true`. A separate synthetic `B=2, N=4` check also passed: output shape `[2,1]`, finite nonzero gradients for both input tensors and every fusion parameter, deterministic repeated evaluation, equivalent batched and individual inference, and changed logits when either input branch was zeroed. `uv lock --check` and `git diff --check` passed.

## Preprocessing limitation and handoff

The checkpoint processor resizes the short edge to 224 pixels, center-crops to 224 × 224, and applies its saved normalization. Wide screenshots lose left/right content and tall screenshots lose top/bottom content. The original screenshot is preserved; alternative resizing is deferred. See the [Transformers CLIP documentation](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPImageProcessor).

The next integration step is to convert Owner 3's OmniParser text/box records into real `[B,N,256]` UI embeddings. UI tokens remain synthetic in this demonstration. Dataset-wide training and meaningful detection evaluation remain deferred.
