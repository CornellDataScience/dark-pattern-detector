"""Task 2: TF-IDF word unigrams/bigrams and binary logistic regression.

Dependencies: scikit-learn (install with ``uv sync --locked``).
Run: ``uv run python baseline.py --data examples/baseline_sample.jsonl``.
The demonstrated sample contains real ContextDP manual transcriptions. Held-out
predictions on this tiny sample are a pipeline check, not a performance benchmark.
"""

import argparse
from collections import Counter
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def load_samples(path: str | Path) -> list[dict]:
    """Read the four-field JSONL contract and reject ambiguous labels or IDs."""
    samples = []
    seen = set()
    for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
        location = f"{path}:{line_number}"
        try:
            sample = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{location}: invalid JSON: {error.msg}") from error
        if not isinstance(sample, dict) or set(sample) != {"id", "text", "label", "source"}:
            raise ValueError(f"{location}: expected id, text, label, and source fields")
        for field in ("id", "text", "source"):
            if not isinstance(sample[field], str) or not sample[field].strip():
                raise ValueError(f"{location}: {field} must be a nonempty string")
        if type(sample["label"]) is not int or sample["label"] not in (0, 1):
            raise ValueError(f"{location}: label must be integer 0 or 1")
        if sample["source"] not in ("manual_transcription", "synthetic"):
            raise ValueError(f"{location}: unknown source {sample['source']!r}")
        if sample["id"] in seen:
            raise ValueError(f"{location}: duplicate ID {sample['id']!r}")
        seen.add(sample["id"])
        samples.append(sample)
    counts = Counter(sample["label"] for sample in samples)
    if counts[0] < 4 or counts[1] < 4:
        raise ValueError("Provide at least four examples per label for the 75/25 split")
    return samples


def train_baseline(samples: list[dict]) -> tuple[Pipeline, list[dict], list[dict]]:
    """Split before fitting so held-out words never enter the TF-IDF vocabulary."""
    train, held_out = train_test_split(
        samples, test_size=0.25, random_state=42,
        stratify=[sample["label"] for sample in samples],
    )
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    model.fit([sample["text"] for sample in train], [sample["label"] for sample in train])
    return model, train, held_out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Labeled JSONL sample")
    args = parser.parse_args()
    try:
        samples = load_samples(args.data)
        model, train, held_out = train_baseline(samples)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    predictions = model.predict([sample["text"] for sample in held_out])
    print(json.dumps({
        "data": str(args.data),
        "sources": dict(Counter(sample["source"] for sample in samples)),
        "seed": 42,
        "train_ids": [sample["id"] for sample in train],
        "train_label_counts": dict(Counter(sample["label"] for sample in train)),
        "held_out_label_counts": dict(Counter(sample["label"] for sample in held_out)),
        "held_out_predictions": [
            {"id": sample["id"], "text": sample["text"],
             "true_label": sample["label"], "predicted_label": int(prediction)}
            for sample, prediction in zip(held_out, predictions)
        ],
        "purpose": "Tiny-sample pipeline check; not a performance benchmark",
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
