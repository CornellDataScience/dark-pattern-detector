"""Embed OmniParser text/box records as padded, trainable 256-dimensional tokens."""

import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer


MINILM_CHECKPOINT = "sentence-transformers/all-MiniLM-L6-v2"
MINILM_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def load_ui_record(path: str | Path) -> dict:
    """Read an elements.json export, preserving element order and text verbatim."""
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if (not isinstance(record, dict)
            or not isinstance(record.get("sample_id"), str)
            or not record["sample_id"].strip()
            or not isinstance(record.get("elements"), list)):
        raise ValueError("Expected a nonempty sample_id and an elements list")
    for index, element in enumerate(record["elements"]):
        if not isinstance(element, dict) or not isinstance(element.get("text"), str):
            raise ValueError(f"Element {index} needs string text")
        bbox = element.get("bbox")
        if (not isinstance(bbox, list) or len(bbox) != 4
                or not all(type(value) in (int, float) and math.isfinite(value)
                           and 0 <= value <= 1 for value in bbox)):
            raise ValueError(f"Element {index} needs four finite normalized coordinates")
        x1, y1, x2, y2 = bbox
        if x1 >= x2 or y1 >= y2:
            raise ValueError(f"Element {index} has an empty or reversed xyxy box")
    return record


def load_ui_adapter() -> "UIEmbeddingAdapter":
    """Load the pinned MiniLM tokenizer and encoder using the Hugging Face cache."""
    tokenizer = AutoTokenizer.from_pretrained(
        MINILM_CHECKPOINT, revision=MINILM_REVISION
    )
    text_encoder = AutoModel.from_pretrained(
        MINILM_CHECKPOINT, revision=MINILM_REVISION
    )
    return UIEmbeddingAdapter(tokenizer, text_encoder)


class UIEmbeddingAdapter(nn.Module):
    """Map exported records to (tokens [B,N,256], boolean padding mask [B,N]).

    MiniLM stays frozen in evaluation mode. Its normalized, attention-masked mean
    embedding (384 values) is concatenated with normalized xyxy geometry (4 values)
    and projected by a trainable linear layer. N is the largest element count in
    the batch; True marks padding. Empty records use only the fusion image token.
    """

    def __init__(self, tokenizer, text_encoder: nn.Module) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder.requires_grad_(False)
        self.text_encoder.eval()
        self.projection = nn.Linear(384 + 4, 256)

    def train(self, mode: bool = True):
        super().train(mode)
        self.text_encoder.eval()
        return self

    def forward(self, records: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode records from load_ui_record or parse_ui.export_elements."""
        if not records:
            raise ValueError("Provide at least one UI record")
        counts = [len(record["elements"]) for record in records]
        features = self.projection.weight.new_zeros(len(records), max(counts), 388)
        padding_mask = torch.ones(features.shape[:2], dtype=torch.bool,
                                  device=features.device)
        texts = [element["text"] for record in records for element in record["elements"]]
        if texts:
            inputs = self.tokenizer(
                texts, padding=True, truncation=True, max_length=256,
                return_tensors="pt",
            )
            inputs = {key: value.to(features.device) for key, value in inputs.items()}
            with torch.no_grad():
                hidden = self.text_encoder(**inputs).last_hidden_state
                attention = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * attention).sum(1) / attention.sum(1).clamp_min(1)
                embeddings = F.normalize(pooled, p=2, dim=1)
            offset = 0
            for row, (record, count) in enumerate(zip(records, counts)):
                if count:
                    features[row, :count, :384] = embeddings[offset:offset + count]
                    features[row, :count, 384:] = features.new_tensor(
                        [element["bbox"] for element in record["elements"]]
                    )
                    padding_mask[row, :count] = False
                    offset += count
        tokens = self.projection(features).masked_fill(padding_mask.unsqueeze(-1), 0)
        return tokens, padding_mask
