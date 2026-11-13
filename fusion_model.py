"""Task 4: frozen CLIP screenshot features and a trainable fusion classifier.

Dependencies: torch, transformers, pillow (install with ``uv sync --locked``).
Run: ``uv run python fusion_model.py --image dataset/web/images/knockaround.com.png``.
The demo uses a real screenshot, synthetic UI tokens, and a demonstration target.
OmniParser output still needs an embedding adapter before it can supply UI tokens.
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection


CLIP_CHECKPOINT = "openai/clip-vit-base-patch32"
CLIP_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"


def load_clip_encoder() -> tuple[CLIPImageProcessor, CLIPVisionModelWithProjection]:
    """Load matching checkpoint preprocessing and frozen vision weights on CPU."""
    processor = CLIPImageProcessor.from_pretrained(
        CLIP_CHECKPOINT, revision=CLIP_REVISION
    )
    encoder = CLIPVisionModelWithProjection.from_pretrained(
        CLIP_CHECKPOINT, revision=CLIP_REVISION
    )
    encoder.requires_grad_(False)
    encoder.eval()
    return processor, encoder


@torch.no_grad()
def extract_image_features(
    image_path: str | Path,
    processor: CLIPImageProcessor,
    encoder: CLIPVisionModelWithProjection,
) -> torch.Tensor:
    """Return projected CLIP features [1,512], preserving the source image file."""
    encoder.eval()
    with Image.open(image_path) as image:
        inputs = processor(images=image.convert("RGB"), return_tensors="pt")
    return encoder(**inputs).image_embeds


class FusionClassifier(nn.Module):
    """Fuse UI tokens [B,N,256] and screenshot features [B,512] into logits [B,1].

    All UI tokens must be valid; variable-length padding is deferred.
    """

    def __init__(self) -> None:
        super().__init__()
        self.image_projection = nn.Linear(512, 256)
        self.transformer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=8,
            dim_feedforward=512,
            batch_first=True,
        )
        self.classifier = nn.Linear(256, 1)

    def forward(
        self, ui_tokens: torch.Tensor, image_features: torch.Tensor
    ) -> torch.Tensor:
        image_token = self.image_projection(image_features).unsqueeze(1)
        tokens = torch.cat((image_token, ui_tokens), dim=1)
        encoded = self.transformer(tokens)
        return self.classifier(encoded[:, 0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="Screenshot path")
    args = parser.parse_args()

    processor, encoder = load_clip_encoder()
    image_features = extract_image_features(args.image, processor, encoder)
    assert image_features.shape == (1, 512)
    assert torch.isfinite(image_features).all()
    assert not image_features.requires_grad

    torch.manual_seed(42)
    model = FusionClassifier()
    ui_tokens = torch.randn(1, 4, 256)
    logits = model(ui_tokens, image_features)
    assert logits.shape == (1, 1) and torch.isfinite(logits).all()

    # This target checks the computation graph; it is not a screenshot annotation.
    target = torch.ones(1, 1)
    loss = nn.BCEWithLogitsLoss()(logits, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    classifier_grad_norm = model.classifier.weight.grad.norm().item()
    assert classifier_grad_norm > 0
    assert not encoder.training
    assert all(not parameter.requires_grad for parameter in encoder.parameters())
    assert all(parameter.grad is None for parameter in encoder.parameters())

    print(json.dumps({
        "checkpoint": CLIP_CHECKPOINT,
        "revision": CLIP_REVISION,
        "image": str(args.image),
        "image_features_shape": list(image_features.shape),
        "ui_tokens_shape": list(ui_tokens.shape),
        "ui_tokens_source": "synthetic; OmniParser embedding adapter pending",
        "logits_shape": list(logits.shape),
        "logits": logits.detach().tolist(),
        "demonstration_target": target.tolist(),
        "loss": loss.item(),
        "classifier_gradient_norm": classifier_grad_norm,
        "clip_frozen_without_gradients": True,
        "checks_passed": True,
    }, indent=2))


if __name__ == "__main__":
    main()
