import json
from pathlib import Path
import tempfile
import unittest

from baseline import load_samples, train_baseline


SAMPLE = Path(__file__).resolve().parents[1] / "examples/baseline_sample.jsonl"


class BaselineTests(unittest.TestCase):
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
