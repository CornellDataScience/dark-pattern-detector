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

    def test_padding_values_and_extra_padding_do_not_change_predictions(self):
        self.model.eval()
        mask = torch.tensor([[False, False, True, True], [True, True, True, True]])
        with torch.no_grad():
            expected = self.model(self.ui, self.image, mask)
            changed = self.ui.clone()
            changed[mask] = torch.randn_like(changed[mask]) * 1000
            torch.testing.assert_close(expected, self.model(changed, self.image, mask))
            extra = torch.cat((self.ui, torch.randn(2, 3, 256)), dim=1)
            extra_mask = torch.cat((mask, torch.ones(2, 3, dtype=torch.bool)), dim=1)
            torch.testing.assert_close(expected, self.model(extra, self.image, extra_mask),
                                       atol=1e-6, rtol=1e-5)
            unpadded = torch.cat([
                self.model(self.ui[0:1, :2], self.image[0:1]),
                self.model(self.ui[1:2, :0], self.image[1:2]),
            ])
            torch.testing.assert_close(expected, unpadded, atol=1e-6, rtol=1e-5)
            all_valid = torch.zeros(2, 4, dtype=torch.bool)
            torch.testing.assert_close(self.model(self.ui, self.image),
                                       self.model(self.ui, self.image, all_valid))

    def test_padding_has_zero_gradient(self):
        mask = torch.tensor([[False, False, True, True], [True, True, True, True]])
        logits = self.model(self.ui, self.image, mask)
        nn.BCEWithLogitsLoss()(logits, torch.tensor([[0.0], [1.0]])).backward()
        self.assertTrue(torch.isfinite(logits).all())
        self.assertEqual(torch.count_nonzero(self.ui.grad[mask]).item(), 0)
        self.assertGreater(self.ui.grad[~mask].norm().item(), 0)
        self.assertTrue(torch.isfinite(self.image.grad).all())
        self.assertGreater(self.image.grad[1].norm().item(), 0)

    def test_rejects_wrong_mask_shape_or_dtype(self):
        for mask in (torch.zeros(2, 4), torch.zeros(2, 5, dtype=torch.bool)):
            with self.subTest(shape=mask.shape, dtype=mask.dtype):
                with self.assertRaisesRegex(ValueError, "boolean with shape"):
                    self.model(self.ui, self.image, mask)


if __name__ == "__main__":
    unittest.main()
