import unittest

import torch
from torch import nn

from fusion_model import FusionClassifier


class FusionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = FusionClassifier()
        self.ui = torch.randn(2, 4, 256, requires_grad=True)
        self.image = torch.randn(2, 512, requires_grad=True)

    def test_binary_loss_backpropagates_through_both_branches(self):
        logits = self.model(self.ui, self.image)
        self.assertEqual(logits.shape, (2, 1))
        self.assertTrue(torch.isfinite(logits).all())
        loss = nn.BCEWithLogitsLoss()(logits, torch.tensor([[0.0], [1.0]]))
        loss.backward()
        for name, tensor in [("ui", self.ui), ("image", self.image),
                             *self.model.named_parameters()]:
            with self.subTest(tensor=name):
                self.assertIsNotNone(tensor.grad)
                self.assertTrue(torch.isfinite(tensor.grad).all())
                self.assertGreater(tensor.grad.norm().item(), 0)

    def test_eval_is_repeatable_independent_across_samples_and_uses_both_inputs(self):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.ui, self.image)
            torch.testing.assert_close(logits, self.model(self.ui, self.image))
            separate = torch.cat([
                self.model(self.ui[i:i+1], self.image[i:i+1]) for i in range(2)
            ])
            torch.testing.assert_close(logits, separate, atol=1e-6, rtol=1e-5)
            self.assertFalse(torch.allclose(logits, self.model(torch.zeros_like(self.ui), self.image)))
            self.assertFalse(torch.allclose(logits, self.model(self.ui, torch.zeros_like(self.image))))


if __name__ == "__main__":
    unittest.main()
