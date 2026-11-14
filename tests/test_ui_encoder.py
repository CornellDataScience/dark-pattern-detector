import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch
from torch import nn
from torch.nn import functional as F

from fusion_model import FusionClassifier
from parse_ui import export_elements
from ui_encoder import UIEmbeddingAdapter, load_ui_record


def tokenize(texts, *, padding, truncation, max_length, return_tensors):
    """Small deterministic tokenizer so unit tests require no downloaded weights."""
    sequences = [[ord(char) % 31 + 1 for char in text[:max_length]] or [1]
                 for text in texts]
    width = max(map(len, sequences))
    return {
        "input_ids": torch.tensor([row + [0] * (width - len(row)) for row in sequences]),
        "attention_mask": torch.tensor([[1] * len(row) + [0] * (width - len(row))
                                        for row in sequences]),
    }


class TinyTextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(32, 384)
        self.dropout = nn.Dropout(0.5)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.dropout(self.embedding(input_ids)))


class UIEncoderTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.tokenizer = Mock(side_effect=tokenize)
        self.adapter = UIEmbeddingAdapter(self.tokenizer, TinyTextEncoder())
        self.records = [
            export_elements("web:1", [
                {"content": "AB", "bbox": [0.1, 0.2, 0.4, 0.3]},
                {"content": "CDE", "bbox": [0.5, 0.6, 0.9, 1.0]},
            ]),
            export_elements("web:2", [{"content": "", "bbox": [0, 0, 1, 1]}]),
            export_elements("web:3", []),
        ]

    def test_loads_parser_export_and_rejects_invalid_records(self):
        invalid = [[], {}, {"sample_id": " ", "elements": []},
                   {"sample_id": "web:1", "elements": None}]
        for element in (None, {"text": None, "bbox": [0, 0, 1, 1]},
                        {"text": "text"}):
            invalid.append({"sample_id": "web:1", "elements": [element]})
        for box in ([0, 0, 2, 1], [0, 0, float("nan"), 1], [0, 0, float("inf"), 1],
                    [0.5, 0, 0.2, 1], [0, 0, 0, 1], [0, 0, True, 1], [0, 0, 1]):
            invalid.append({"sample_id": "web:1", "elements": [{"text": "", "bbox": box}]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "elements.json"
            for record in self.records:
                path.write_text(json.dumps(record))
                self.assertEqual(load_ui_record(path), record)
            for record in invalid:
                with self.subTest(record=record), self.assertRaises(ValueError):
                    path.write_text(json.dumps(record))
                    load_ui_record(path)
            path.write_text("{")
            with self.assertRaises(ValueError):
                load_ui_record(path)

    def test_pools_only_text_tokens_and_preserves_geometry_order_and_padding(self):
        # Expose text coordinates and geometry directly to check the adapter inputs.
        with torch.no_grad():
            self.adapter.projection.weight.zero_()
            self.adapter.projection.bias.zero_()
            self.adapter.projection.weight[:252, :252] = torch.eye(252)
            self.adapter.projection.weight[252:, 384:] = torch.eye(4)
        tokens, mask = self.adapter(self.records)
        self.assertEqual(tokens.shape, (3, 2, 256))
        self.assertEqual(mask.dtype, torch.bool)
        self.assertEqual(mask.tolist(), [[False, False], [False, True], [True, True]])
        torch.testing.assert_close(tokens[mask], torch.zeros_like(tokens[mask]))
        weights = self.adapter.text_encoder.embedding.weight
        expected_text = F.normalize(torch.stack([
            weights[[4, 5]].mean(0), weights[[6, 7, 8]].mean(0), weights[1],
        ]), dim=1)
        torch.testing.assert_close(tokens[~mask, :252], expected_text[:, :252])
        torch.testing.assert_close(tokens[~mask, 252:], torch.tensor([
            [0.1, 0.2, 0.4, 0.3], [0.5, 0.6, 0.9, 1.0], [0, 0, 1, 1],
        ]))
        self.tokenizer.assert_called_once_with(
            ["AB", "CDE", ""], padding=True, truncation=True,
            max_length=256, return_tensors="pt",
        )

    def test_empty_records_skip_text_encoding_and_use_image_only(self):
        tokens, mask = self.adapter([self.records[2], self.records[2]])
        self.assertEqual(tokens.shape, (2, 0, 256))
        self.assertEqual(mask.shape, (2, 0))
        self.tokenizer.assert_not_called()
        logits = FusionClassifier()(tokens, torch.randn(2, 512), mask)
        self.assertEqual(logits.shape, (2, 1))
        self.assertTrue(torch.isfinite(logits).all())
        logits.sum().backward()
        with self.assertRaisesRegex(ValueError, "at least one"):
            self.adapter([])

    def test_fusion_backpropagates_to_text_and_box_projection_but_freezes_encoder(self):
        self.adapter.train()
        self.assertTrue(self.adapter.projection.training)
        self.assertFalse(self.adapter.text_encoder.training)
        tokens, mask = self.adapter(self.records)
        repeated, repeated_mask = self.adapter(self.records)
        torch.testing.assert_close(tokens, repeated)
        self.assertTrue(torch.equal(mask, repeated_mask))
        model = FusionClassifier()
        image = torch.randn(3, 512, requires_grad=True)
        loss = nn.BCEWithLogitsLoss()(
            model(tokens, image, mask), torch.tensor([[1.0], [0.0], [1.0]])
        )
        loss.backward()
        for parameter in self.adapter.text_encoder.parameters():
            self.assertFalse(parameter.requires_grad)
            self.assertIsNone(parameter.grad)
        for name, parameter in [*self.adapter.projection.named_parameters(),
                                *model.named_parameters(), ("image", image)]:
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(parameter.grad.norm().item(), 0)
        self.assertGreater(self.adapter.projection.weight.grad[:, :384].norm().item(), 0)
        self.assertGreater(self.adapter.projection.weight.grad[:, 384:].norm().item(), 0)

    def test_exported_record_batches_match_individual_predictions_and_use_text_and_boxes(self):
        self.adapter.eval()
        model = FusionClassifier().eval()
        images = torch.randn(3, 512)
        with torch.no_grad():
            tokens, mask = self.adapter(self.records)
            batched = model(tokens, images, mask)
            separate = []
            for index, record in enumerate(self.records):
                one, one_mask = self.adapter([record])
                separate.append(model(one, images[index:index + 1], one_mask))
            torch.testing.assert_close(batched, torch.cat(separate), atol=1e-6, rtol=1e-5)
            for field, value in (("text", "Different wording"), ("bbox", [0, 0, 1, 1])):
                changed = copy.deepcopy(self.records)
                changed[0]["elements"][0][field] = value
                other_tokens, other_mask = self.adapter(changed)
                other = model(other_tokens, images, other_mask)
                with self.subTest(field=field):
                    self.assertFalse(torch.allclose(batched[0], other[0]))
                    torch.testing.assert_close(batched[1:], other[1:])


if __name__ == "__main__":
    unittest.main()
