#!/usr/bin/env python3
"""Create a 12-record prototype JSONL from Yada et al.'s labeled text dataset.

The original TSV is downloaded to dataset/ec-darkpattern/ (normally ignored by git).
Run from the root of the dark-pattern-detector repository:
    python scripts/prepare_real_dataset_sample.py
Or provide a locally downloaded dataset file:
    python scripts/prepare_real_dataset_sample.py --source-file dataset/ec-darkpattern/dataset.tsv
"""

import argparse
import csv
import json
import random
import re
import urllib.request
from pathlib import Path

SOURCE_URL = "https://raw.githubusercontent.com/yamanalab/ec-darkpattern/master/dataset/dataset.tsv"
SOURCE_PAGE = "https://github.com/yamanalab/ec-darkpattern/blob/master/dataset/dataset.tsv"
DEFAULT_TSV = Path("dataset/ec-darkpattern/dataset.tsv")
DEFAULT_OUTPUT = Path("examples/baseline_sample.jsonl")


def clean(text):
    return re.sub(r"\s+", " ", (text or "").strip())


def load_rows(source_file):
    with source_file.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"page_id", "text", "label", "Pattern Category"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Unexpected TSV columns: {reader.fieldnames}; expected {sorted(required)}")
        seen = set()
        rows = []
        for i, row in enumerate(reader, start=2):
            label_str = clean(row.get("label", ""))
            text = clean(row.get("text", ""))
            category = clean(row.get("Pattern Category", ""))
            if label_str not in ("0", "1") or not 12 <= len(text) <= 180:
                continue
            normalized = text.casefold()
            if normalized in seen:
                continue
            seen.add(normalized)
            label = int(label_str)
            if label == 0 and category.casefold() != "not dark pattern":
                # Protect against unanticipated label/category mapping.
                continue
            if label == 1 and category.casefold() == "not dark pattern":
                continue
            page_id = clean(row.get("page_id", "")) or "unknown"
            rows.append({"id": f"yada2022:{page_id}:row{i}", "text": text,
                         "label": label, "source": SOURCE_PAGE, "_category": category})
    return rows


def select_balanced(rows, seed):
    rng = random.Random(seed)
    positives_by_category = {}
    negatives = []
    for row in rows:
        if row["label"] == 1:
            positives_by_category.setdefault(row["_category"], []).append(row)
        else:
            negatives.append(row)
    rng.shuffle(negatives)
    for group in positives_by_category.values():
        rng.shuffle(group)

    # Prefer examples from different categories. This is a *small demo fixture*,
    # not an unbiased or representative evaluation set.
    categories = [c for c in ("Urgency", "Scarcity", "Misdirection", "Social Proof")
                  if positives_by_category.get(c)]
    categories += [c for c in sorted(positives_by_category)
                   if c not in categories]
    positives = [positives_by_category[c].pop() for c in categories[:4]]
    remaining = [r for group in positives_by_category.values() for r in group]
    rng.shuffle(remaining)
    positives += remaining[:6 - len(positives)]
    if len(positives) != 6 or len(negatives) < 6:
        raise ValueError("Need at least 6 distinct examples from each class")
    chosen = positives + negatives[:6]
    rng.shuffle(chosen)
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-file", type=Path, default=DEFAULT_TSV,
                        help="Local path of official dataset.tsv")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if not args.source_file.is_file():
        args.source_file.parent.mkdir(parents=True, exist_ok=True)
        print("Downloading dataset from:", SOURCE_URL)
        try:
            urllib.request.urlretrieve(SOURCE_URL, args.source_file)
        except Exception as exc:
            raise SystemExit(f"Download failed: {exc}\n"
                  "Download dataset.tsv manually from https://github.com/yamanalab/ec-darkpattern/"
                  "blob/master/dataset/dataset.tsv and use --source-file.")
    records = select_balanced(load_rows(args.source_file), args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            public_record = {k: record[k] for k in ("id", "text", "label", "source")}
            handle.write(json.dumps(public_record, ensure_ascii=False) + "\n")
    print("Wrote", len(records), "real source-labeled examples to", args.output)
    print("Class counts: 6 dark pattern (1), 6 not dark pattern (0)")
    print("Selected categories:", ", ".join(sorted(set(r["_category"] for r in records if r["label"] == 1))))
    print("These are original dataset labels, not newly verified judgments.")
    print("Raw downloaded TSV:", args.source_file, "(do NOT git add)")

if __name__ == "__main__":
    main()
