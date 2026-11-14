"""Task 3: run upstream OmniParser and export normalized UI text/box records.

Use the separate environment and pinned setup in reports/omniparser_progress.md:
dataset/.venv-omniparser/bin/python parse_ui.py \
    --image dataset/web/images/knockaround.com.png --sample-id web:6
Outputs under dataset/parsed/: elements.json, native.json, and annotated.png.
Native element sources distinguish OCR text from generated icon descriptions.
"""

import argparse
import base64
import json
import math
import os
from pathlib import Path
import subprocess
import sys


def export_elements(sample_id: str, native_elements: list[dict]) -> dict:
    """Preserve upstream order and normalized xyxy boxes (not overlay xywh boxes)."""
    elements = []
    for index, element in enumerate(native_elements):
        text = element["content"]
        bbox = element["bbox"]
        if not isinstance(text, str):
            raise ValueError(f"Element {index} has no text/content")
        if len(bbox) != 4 or not all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and 0 <= value <= 1 for value in bbox
        ):
            raise ValueError(f"Element {index} needs four finite normalized coordinates")
        x1, y1, x2, y2 = bbox
        if x1 >= x2 or y1 >= y2:
            raise ValueError(f"Element {index} has an empty or reversed xyxy box")
        elements.append({"text": text, "bbox": list(bbox)})
    return {"sample_id": sample_id, "elements": elements}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="Original screenshot")
    parser.add_argument("--sample-id", required=True, help="Original dataset ID, e.g. web:6")
    parser.add_argument("--omniparser-dir", type=Path, default=Path("dataset/OmniParser"))
    parser.add_argument("--output-dir", type=Path, default=Path("dataset/parsed"))
    args = parser.parse_args()
    root = args.omniparser_dir.resolve()
    detector = root / "weights/icon_detect_v3/model.pt"
    caption = root / "weights/icon_caption_florence"
    for path in (args.image, root / "util/omniparser.py", detector, caption / "model.safetensors"):
        if not path.is_file():
            parser.error(f"Missing {path}; see reports/omniparser_progress.md for setup")
    if not args.sample_id.strip():
        parser.error("--sample-id must be nonempty")
    output = args.output_dir
    if args.image.resolve() in {(output / name).resolve() for name in
                                 ("elements.json", "native.json", "annotated.png")}:
        parser.error("Choose an output directory that will not overwrite the screenshot")
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()

    # Keep the baseline/CLIP environment independent of upstream dependencies.
    os.environ.setdefault("EASYOCR_MODULE_PATH", str(root / "weights/easyocr"))
    os.environ.setdefault("PADDLE_OCR_BASE_DIR", str(root / "weights/paddleocr"))
    sys.path.insert(0, str(root))
    from util.omniparser import Omniparser

    model = Omniparser({
        "som_model_path": str(detector),
        "caption_model_name": "florence2",
        "caption_model_path": str(caption),
        "BOX_TRESHOLD": 0.05,
    })
    annotated, native_elements = model.parse(
        base64.b64encode(args.image.read_bytes()).decode("ascii")
    )
    record = export_elements(args.sample_id, native_elements)
    output.mkdir(parents=True, exist_ok=True)
    (output / "elements.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output / "native.json").write_text(json.dumps({
        "sample_id": args.sample_id,
        "image": str(args.image),
        "omniparser_revision": revision,
        "elements": native_elements,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "annotated.png").write_bytes(base64.b64decode(annotated))
    print(json.dumps({
        "sample_id": args.sample_id,
        "element_count": len(record["elements"]),
        "output_dir": str(output),
        "text_provenance": "native.json elements[*].source; same order as elements.json",
        "omniparser_revision": revision,
    }, indent=2))


if __name__ == "__main__":
    main()
