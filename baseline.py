"""TF-IDF word unigrams/bigrams and binary logistic regression.

Dependencies: scikit-learn (install with ``uv sync --locked``).
Run: ``uv run python baseline.py --data examples/baseline_sample.jsonl``.
Full corpus: ``uv run python baseline.py --ec-darkpattern dataset/ec-darkpattern/dataset.tsv``.
JSONL samples are pipeline checks; the full TSV uses a grouped holdout evaluation.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import platform

import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline

from scripts.prepare_real_dataset_sample import SOURCE_PAGE, clean


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
        if sample["source"] not in ("manual_transcription", "synthetic", SOURCE_PAGE):
            raise ValueError(f"{location}: unknown source {sample['source']!r}")
        if sample["id"] in seen:
            raise ValueError(f"{location}: duplicate ID {sample['id']!r}")
        seen.add(sample["id"])
        samples.append(sample)
    counts = Counter(sample["label"] for sample in samples)
    if counts[0] < 4 or counts[1] < 4:
        raise ValueError("Provide at least four examples per label for the 75/25 split")
    return samples


def load_ec_darkpattern(path: str | Path) -> tuple[list[dict], dict]:
    """Deduplicate text and connect pages sharing text before splitting."""
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"page_id", "text", "label", "Pattern Category"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: expected TSV columns {sorted(required)}")
        rows = list(reader)
    parent = {}
    by_text = defaultdict(list)

    def find(page):
        while parent[page] != page:
            parent[page] = parent[parent[page]]
            page = parent[page]
        return page

    for line, row in enumerate(rows, 2):
        page, text = clean(row["page_id"]), clean(row["text"])
        category, label = clean(row["Pattern Category"]), clean(row["label"])
        if not page or not text or not category or label not in ("0", "1"):
            raise ValueError(f"{path}:row{line}: missing content or invalid label")
        if (label == "0") != (category.casefold() == "not dark pattern"):
            raise ValueError(f"{path}:row{line}: label/category mismatch")
        parent.setdefault(page, page)
        duplicates = by_text[text.casefold()]
        if duplicates:
            if duplicates[0]["label"] != int(label):
                raise ValueError(f"{path}:row{line}: conflicting labels for duplicate text")
            parent[find(page)] = find(duplicates[0]["page_id"])
        duplicates.append({
            "id": f"yada2022:{page}:row{line}", "text": text,
            "label": int(label), "source": SOURCE_PAGE, "page_id": page,
            "category": category,
        })
    samples = []
    for duplicates in by_text.values():
        sample = duplicates[0]
        sample["group"] = find(sample["page_id"])
        sample["page_ids"] = sorted({row["page_id"] for row in duplicates})
        samples.append(sample)
    return samples, {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "raw_rows": len(rows), "retained_rows": len(samples),
        "duplicate_rows_removed": len(rows) - len(samples),
        "page_count": len(parent),
        "group_count": len({sample["group"] for sample in samples}),
        "label_counts": dict(Counter(sample["label"] for sample in samples)),
        "category_counts": dict(Counter(sample["category"] for sample in samples)),
    }


def train_baseline(
    samples: list[dict], *, grouped: bool = False,
) -> tuple[Pipeline, list[dict], list[dict]]:
    """Split before fitting so held-out words never enter the TF-IDF vocabulary."""
    labels = [sample["label"] for sample in samples]
    if grouped:
        splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
        train_indices, test_indices = next(splitter.split(
            samples, labels, groups=[sample["group"] for sample in samples],
        ))
        train = [samples[i] for i in train_indices]
        held_out = [samples[i] for i in test_indices]
    else:
        train, held_out = train_test_split(
            samples, test_size=0.25, random_state=42, stratify=labels,
        )
    if {s["label"] for s in train} != {0, 1} or {s["label"] for s in held_out} != {0, 1}:
        raise ValueError("Both training and held-out partitions must contain both labels")
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    model.fit([sample["text"] for sample in train], [sample["label"] for sample in train])
    return model, train, held_out


def classification_metrics(labels, predictions, probabilities) -> dict:
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "positive_f1": f1_score(labels, predictions, zero_division=0),
        "positive_precision": precision_score(labels, predictions, zero_division=0),
        "positive_recall": recall_score(labels, predictions, zero_division=0),
        "roc_auc": roc_auc_score(labels, probabilities),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--data", type=Path, help="Labeled JSONL sample (pipeline check)")
    inputs.add_argument("--ec-darkpattern", type=Path, help="Full labeled dataset.tsv")
    parser.add_argument("--output", type=Path, help="Save complete results as JSON")
    args = parser.parse_args()
    try:
        if args.ec_darkpattern:
            samples, audit = load_ec_darkpattern(args.ec_darkpattern)
        else:
            samples, audit = load_samples(args.data), None
        model, train, held_out = train_baseline(samples, grouped=bool(args.ec_darkpattern))
    except (OSError, ValueError) as error:
        parser.error(str(error))
    predictions = model.predict([sample["text"] for sample in held_out])
    probabilities = model.predict_proba([sample["text"] for sample in held_out])[:, 1]
    result = {
        "data": str(args.ec_darkpattern or args.data),
        "sources": dict(Counter(sample["source"] for sample in samples)),
        "seed": 42,
        "train_ids": [sample["id"] for sample in train],
        "train_label_counts": dict(Counter(sample["label"] for sample in train)),
        "held_out_label_counts": dict(Counter(sample["label"] for sample in held_out)),
        "held_out_predictions": [
            {"id": sample["id"], "text": sample["text"],
             "true_label": sample["label"], "predicted_label": int(prediction),
             "positive_probability": float(probability)}
            for sample, prediction, probability in zip(held_out, predictions, probabilities)
        ],
        "purpose": "Tiny-sample pipeline check; not a performance benchmark",
    }
    if audit is not None:
        labels = [sample["label"] for sample in held_out]
        train_texts = {clean(sample["text"]).casefold() for sample in train}
        held_out_texts = {clean(sample["text"]).casefold() for sample in held_out}
        train_pages = {page for sample in train for page in sample["page_ids"]}
        held_out_pages = {page for sample in held_out for page in sample["page_ids"]}
        dummy = DummyClassifier(strategy="most_frequent").fit(
            [[0]] * len(train), [sample["label"] for sample in train],
        )
        result.update({
            "purpose": "Full-corpus grouped holdout evaluation; no hyperparameter tuning",
            "dataset_audit": audit,
            "split": "First fold of StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42); pages connected by identical normalized text are grouped",
            "versions": {"python": platform.python_version(), "scikit_learn": sklearn.__version__},
            "model": {"ngram_range": [1, 2], "C": 1.0, "max_iter": 1000,
                      "class_weight": None, "threshold": 0.5,
                      "vocabulary_size": len(model.named_steps["tfidf"].vocabulary_)},
            "train_page_ids": sorted(train_pages),
            "held_out_page_ids": sorted(held_out_pages),
            "leakage_checks": {"shared_pages": len(train_pages & held_out_pages),
                               "shared_texts": len(train_texts & held_out_texts)},
            "metrics": classification_metrics(labels, predictions, probabilities),
            "majority_dummy_metrics": classification_metrics(
                labels, dummy.predict([[0]] * len(held_out)),
                dummy.predict_proba([[0]] * len(held_out))[:, 1],
            ),
        })
    output = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    print(output, end="")


if __name__ == "__main__":
    main()
