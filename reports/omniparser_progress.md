# Task 3 — real screenshot parsing with OmniParser

Completed 2026-11-14. [`parse_ui.py`](../parse_ui.py) successfully ran Microsoft's
upstream `Omniparser.parse()` on ContextDP screenshot `web:6` and saved 59 elements,
their native records, and an annotated image. Several text and icon regions were
visually checked against the original screenshot and the saved overlay.

## Environment and versions

- Upstream: [Microsoft OmniParser](https://github.com/microsoft/OmniParser/tree/354021201345a96178360b28733573e27269f2de),
  commit `354021201345a96178360b28733573e27269f2de`, unmodified local checkout at
  `dataset/OmniParser/`.
- Hardware: Apple Silicon (`arm64`), macOS 27.2.1. The detector and Florence caption
  model ran on CPU; upstream EasyOCR selected available MPS acceleration.
- Isolated Python 3.12.13 environment: `dataset/.venv-omniparser/`. The shared
  Python 3.11 / Transformers 5 environment for CLIP is unchanged.
- Core versions: torch 2.6.0, torchvision 0.21.0, transformers 4.49.0,
  huggingface-hub 0.36.0, numpy 1.26.4, pillow 13.0.0, easyocr 1.7.2,
  paddleocr 2.10.0, paddlepaddle 2.6.2, supervision 0.18.0.
- Full installed package snapshot: `dataset/omniparser-packages.txt`.
  `uv pip check --python dataset/.venv-omniparser/bin/python` passed.

The upstream requirements also include unrelated interactive-agent applications.
The following installs the inference dependencies imported by `util/omniparser.py`
and `util/utils.py`. Transformers 4 and PaddleOCR 2 match the APIs used there;
installing these into the shared environment would conflict with its Transformers 5
requirement. PaddleOCR is initialized by upstream even though parsing uses EasyOCR.

## Reproduce from the repository root

For a fresh setup:

```sh
git clone --depth 1 https://github.com/microsoft/OmniParser.git dataset/OmniParser
git -C dataset/OmniParser fetch --depth 1 origin 354021201345a96178360b28733573e27269f2de
git -C dataset/OmniParser checkout --detach 354021201345a96178360b28733573e27269f2de
uv venv --python 3.12 dataset/.venv-omniparser
uv pip install --python dataset/.venv-omniparser/bin/python \
  torch==2.6.0 torchvision==0.21.0 transformers==4.49.0 \
  huggingface-hub==0.36.0 numpy==1.26.4 pillow==13.0.0 \
  supervision==0.18.0 easyocr==1.7.2 paddleocr==2.10.0 paddlepaddle==2.6.2 \
  openai==1.3.5 einops==0.8.0 timm==1.0.31 accelerate==1.15.0 \
  socksio==1.0.0 'httpx[socks]==0.27.2'
```

Download the pinned YOLOv9-E detector and Florence caption weights with the shared
environment's Hugging Face client (the detector comes from upstream's PR #37):

```sh
uv run python - <<'PY'
from pathlib import Path
from huggingface_hub import snapshot_download

weights = Path('dataset/OmniParser/weights')
snapshot_download(
    'microsoft/OmniParser-v2.0',
    revision='d10c4687d4af8245d67780ae59900f09660aad3a',
    allow_patterns=['icon_detect_v3/model.pt'], local_dir=weights,
)
snapshot_download(
    'microsoft/OmniParser-v2.0',
    revision='f55d0750e5b94db2125ef0b45b0fa4a85ddc59b4',
    allow_patterns=['icon_caption/*'], local_dir=weights,
)
(weights / 'icon_caption').rename(weights / 'icon_caption_florence')
PY
```

Upstream downloads the Florence processor and Python model code on first use.
Observed code revisions were `5ca5edf5bd017b9919c05d08aebef5e4c7ac3bac` for
`microsoft/Florence-2-base` and `f6c1a25888ffc1d945ee8a1a77ac833c7303d46e` for
`microsoft/Florence-2-base-ft`. The upstream helper resolves these from `main`, so
future upstream code changes may require rechecking this environment. Detector and
caption weights are pinned by the commands above. The wrapper stores EasyOCR and
PaddleOCR weights under `dataset/OmniParser/weights/` by default; those weights
download automatically on first use. Florence code/tokenizer files use the normal
Hugging Face cache outside Git.

Obtain the screenshot if it is not already present:

```sh
mkdir -p dataset/web/images
curl -fL https://raw.githubusercontent.com/SageSELab/AidUI/bb97865abf6321ff2564f064da8b48ac0a82f284/evaluation/evaluation_dataset/web/images/knockaround.com.png \
  -o dataset/web/images/knockaround.com.png
```

Run inference and save outputs:

```sh
dataset/.venv-omniparser/bin/python -u parse_ui.py \
  --image dataset/web/images/knockaround.com.png --sample-id web:6 \
  --output-dir dataset/parsed/web-6
```

The original 1700 × 919 screenshot was unchanged after parsing; SHA-256:
`13f06cc86e3bba311e6db73d1788e90a474b7eb581a6615b2d50261b9923bbe2`.
The observed run is logged at `dataset/omniparser-run.log`. Caption generation took
approximately 30 seconds after loading models and detecting regions. Setup and
inference completed within the task's one-hour limit; no blocker remains.

## Outputs and schema

All generated files are ignored by Git:

| Local artifact | Contents |
| --- | --- |
| `dataset/parsed/web-6/elements.json` | Shared `{sample_id, elements: [{text, bbox}]}` record |
| `dataset/parsed/web-6/native.json` | Screenshot path, code revision, and all native element fields |
| `dataset/parsed/web-6/annotated.png` | Upstream labeled screenshot, with zero-based element indices |

Boxes in the shared record are normalized `[x1,y1,x2,y2]`, taken directly from
native element `bbox` fields. The wrapper rejects nonfinite, out-of-range, empty,
or reversed boxes. Overlay coordinates from other upstream helpers can be `xywh`
and are not used for export. All 59 exported boxes passed these checks.

Element order is identical in both JSON files. Read `native.json`'s `source` at
the corresponding index to determine text provenance:

| Native source | Count | Text provenance |
| --- | --- | --- |
| `box_ocr_content_ocr` | 12 | EasyOCR text in an OCR box |
| `box_yolo_content_ocr` | 26 | EasyOCR text associated with a detector box |
| `box_yolo_content_yolo` | 21 | Florence-generated icon description in a detector box |

Generated descriptions are model outputs, not transcriptions. Native content is
preserved without correcting OCR mistakes or generated captions.

## Visual checks and observed errors

Coordinates below are rounded for readability; JSON retains upstream precision.

| Element | Normalized box | Visual comparison |
| --- | --- | --- |
| 0 | `[0.5429, 0.3123, 0.5782, 0.3297]` | Box covers the price; OCR reads `S28.00` instead of `$28.00`. |
| 4 | `[0.5418, 0.6583, 0.5900, 0.6736]` | Box covers the scarcity cue; OCR reads `Limited stockl` instead of `Limited stock!`. |
| 12 | `[0.6580, 0.4732, 0.8060, 0.5488]` | Box covers the blue purchase button; OCR reads `ADD T CART`, missing the O. |
| 40 | `[0.2050, 0.7074, 0.2339, 0.7618]` | Box covers a sunglasses thumbnail; generated description identifies sunglasses. |
| 50 | `[0.7781, 0.0804, 0.8024, 0.1145]` | Box covers the cart icon; generated description incorrectly calls it a login function. |

The activity message was fragmented across several OCR boxes and some numbers
were missed. There are also spurious border detections and inaccurate captions.
These findings matter for later embedding and label-quality work; successful
parsing alone does not establish detection accuracy.

## Handoff

Owner 4 can consume `dataset/parsed/web-6/elements.json` using the shared schema.
It contains structured text and geometry, not dense UI embeddings. Converting
these records to `[B,N,256]` tokens remains the explicitly deferred integration
step; the fusion demonstration still uses synthetic UI tokens.

Export contract tests run in the shared environment without model downloads:

```sh
uv run python -m unittest discover -s tests -p 'test_parse_ui.py' -v
```
