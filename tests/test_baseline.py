import csv
import json
from pathlib import Path
import tempfile
import unittest

from baseline import classification_metrics, load_ec_darkpattern, load_samples, train_baseline
from scripts.prepare_real_dataset_sample import load_rows, select_balanced


SAMPLE = Path(__file__).resolve().parents[1] / "examples/baseline_sample.jsonl"


class BaselineTests(unittest.TestCase):
    def test_merged_ec_sample_runs_without_replacing_contextdp(self):
        samples = load_samples(SAMPLE.with_name("ec_darkpattern_sample.jsonl"))
        self.assertEqual(len(samples), 12)
        self.assertEqual(sum(s["label"] for s in samples), 6)
        model, train, held_out = train_baseline(samples)
        self.assertEqual((len(train), len(held_out)), (9, 3))
        self.assertEqual(len(model.predict([s["text"] for s in held_out])), 3)

    def test_full_corpus_deduplication_and_grouped_split_prevent_leakage(self):
        rows = [
            [str(page), f"{'offer' if label else 'catalog'} uniqueterm{page}label{label}",
             str(label), "Urgency" if label else "Not Dark Pattern"]
            for page in range(24) for label in (0, 1)
        ]
        # Link pages 0 -> 1 -> 2 through different duplicated texts.
        rows += [["1", "  OFFER   UNIQUETERM0LABEL1 ", "1", "Urgency"],
                 ["2", "offer uniqueterm1label1", "1", "Urgency"]]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.tsv"
            with path.open("w", newline="") as handle:
                writer = csv.writer(handle, delimiter="\t")
                writer.writerow(["page_id", "text", "label", "Pattern Category"])
                writer.writerows(rows)
            samples, audit = load_ec_darkpattern(path)
            self.assertEqual((audit["raw_rows"], audit["retained_rows"]), (50, 48))
            self.assertEqual(audit["group_count"], 22)
            self.assertEqual(len({s["group"] for s in samples if s["page_id"] in {"0", "1", "2"}}), 1)
            model, train, held_out = train_baseline(samples, grouped=True)
            self.assertFalse({p for s in train for p in s["page_ids"]} &
                             {p for s in held_out for p in s["page_ids"]})
            self.assertFalse({s["text"].casefold() for s in train} &
                             {s["text"].casefold() for s in held_out})
            for sample in held_out:
                self.assertNotIn(sample["text"].split()[-1], model["tfidf"].vocabulary_)
            _, other_train, other_held_out = train_baseline(samples, grouped=True)
            self.assertEqual((train, held_out), (other_train, other_held_out))
            # The merged sample generator remains usable with the same TSV.
            selected = select_balanced(load_rows(path), seed=2026)
            self.assertEqual((len(selected), sum(s["label"] for s in selected)), (12, 6))
            with path.open("a") as handle:
                handle.write("0\toffer uniqueterm0label1\t0\tNot Dark Pattern\n")
            with self.assertRaisesRegex(ValueError, "conflicting labels"):
                load_ec_darkpattern(path)

    def test_metrics_use_probability_auc_and_true_predicted_matrix_order(self):
        metrics = classification_metrics(
            [0, 0, 0, 1, 1], [0, 1, 1, 0, 1], [0.1, 0.6, 0.8, 0.4, 0.9],
        )
        self.assertEqual(metrics["confusion_matrix"], [[1, 2], [1, 1]])
        self.assertEqual(metrics["macro_f1"], 0.4)
        self.assertEqual(metrics["positive_precision"], 1 / 3)
        self.assertEqual(metrics["positive_recall"], 0.5)
        self.assertAlmostEqual(metrics["roc_auc"], 2 / 3)

    def test_real_sample_contract_and_stratified_split(self):
        samples = load_samples(SAMPLE)
        self.assertEqual(len(samples), 8)
        self.assertEqual({s["source"] for s in samples}, {"manual_transcription"})
        model, train, held_out = train_baseline(samples)
        self.assertEqual((len(train), len(held_out)), (6, 2))
        self.assertEqual({s["label"] for s in train}, {0, 1})
        self.assertEqual({s["label"] for s in held_out}, {0, 1})
        self.assertFalse({s["id"] for s in train} & {s["id"] for s in held_out})
        self.assertEqual(set(model.classes_), {0, 1})

    def test_held_out_vocabulary_is_excluded_and_split_is_reproducible(self):
        samples = [
            {"id": f"synthetic:{i}", "text": f"shared word uniqueterm{i}",
             "label": i % 2, "source": "synthetic"}
            for i in range(8)
        ]
        model, train, held_out = train_baseline(samples)
        vocabulary = model.named_steps["tfidf"].vocabulary_
        for sample in held_out:
            self.assertNotIn(sample["text"].split()[-1], vocabulary)
        for sample in train:
            self.assertIn(sample["text"].split()[-1], vocabulary)
        self.assertIn("shared word", vocabulary)
        other_model, other_train, other_held_out = train_baseline(samples)
        self.assertEqual((train, held_out), (other_train, other_held_out))
        texts = [sample["text"] for sample in held_out]
        self.assertEqual(model.predict(texts).tolist(), other_model.predict(texts).tolist())

    def test_invalid_records_have_actionable_errors(self):
        samples = load_samples(SAMPLE)
        invalid = [
            ("not json", "invalid JSON"),
            ("\n", "invalid JSON"),
            (json.dumps([]), "expected id"),
            (json.dumps({"id": "missing"}), "expected id"),
            (json.dumps({**samples[0], "label": True}), "integer 0 or 1"),
            (json.dumps({**samples[0], "label": 2}), "integer 0 or 1"),
            (json.dumps({**samples[0], "text": " "}), "nonempty string"),
            (json.dumps({**samples[0], "source": "unknown"}), "unknown source"),
            ("\n".join(json.dumps(samples[0]) for _ in range(8)), "duplicate ID"),
            ("", "at least four"),
            ("\n".join(json.dumps(s) for s in samples[:4]), "at least four"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.jsonl"
            for content, message in invalid:
                with self.subTest(message=message, content=content):
                    path.write_text(content)
                    with self.assertRaisesRegex(ValueError, message):
                        load_samples(path)


if __name__ == "__main__":
    unittest.main()
