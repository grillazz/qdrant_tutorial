"""Sparse encoders that produce Qdrant sparse vectors locally.

Two implementations behind one protocol:

- ``Bm25SparseEncoder``  token counts, hashed to uint32. Qdrant applies IDF.
- ``SpladeSparseEncoder`` neural term expansion. Weights are already final.

Each encoder reports the collection config it needs via ``vector_params()``, so
the caller cannot pair an encoder with the wrong ``Modifier``.
"""

from __future__ import annotations

import re
from binascii import crc32
from collections import Counter
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from qdrant_client import models

from hybrid_search.encoders.device import describe_runtime, resolve_device, resolve_dtype

__all__ = ["SparseEncoder", "Bm25SparseEncoder", "SpladeSparseEncoder"]

POLISH_SPLADE = "sdadas/polish-splade"


@runtime_checkable
class SparseEncoder(Protocol):
    """Turns text into Qdrant sparse vectors."""

    def vector_params(self) -> models.SparseVectorParams:
        """Collection config this encoder requires."""
        ...

    def encode_passages(self, texts: Sequence[str]) -> list[models.SparseVector]: ...

    def encode_query(self, text: str) -> models.SparseVector: ...


def _sparse_vector(pairs: Sequence[tuple[int, float]]) -> models.SparseVector:
    """Build a SparseVector with sorted, unique indices."""
    merged: dict[int, float] = {}
    for index, value in pairs:
        merged[index] = merged.get(index, 0.0) + value
    ordered = sorted(merged.items())
    return models.SparseVector(
        indices=[i for i, _ in ordered],
        values=[v for _, v in ordered],
    )


class Bm25SparseEncoder:
    """Term frequencies with hashed token indices.

    Qdrant computes corpus IDF at search time, so this only emits raw counts.
    Kept for comparison against SPLADE; it has no Polish lemmatisation, which is
    exactly the weakness SPLADE addresses.
    """

    def __init__(self, pattern: str = r"(?u)\b\w\w+\b") -> None:
        self._tokenizer = re.compile(pattern)

    def vector_params(self) -> models.SparseVectorParams:
        return models.SparseVectorParams(modifier=models.Modifier.IDF)

    def encode_query(self, text: str) -> models.SparseVector:
        tokens = self._tokenizer.findall(text.lower())
        if not tokens:
            return models.SparseVector(indices=[], values=[])
        counts = Counter(tokens)
        return _sparse_vector(
            [(crc32(t.encode()), float(n)) for t, n in counts.items()]
        )

    def encode_passages(self, texts: Sequence[str]) -> list[models.SparseVector]:
        return [self.encode_query(t) for t in texts]


class SpladeSparseEncoder:
    """SPLADE++ term weights from a masked language model head.

    Pooling follows the model card exactly::

        max over sequence of log(1 + relu(logits)) * attention_mask

    The same encoding is used for passages and queries, which is how SPLADE++
    EnsembleDistil was trained.

    Memory note: the intermediate logits tensor is
    ``batch_size * max_length * vocab_size * 4`` bytes. With the defaults below
    and a 50k vocabulary that is roughly 800 MB, so raise ``batch_size`` only if
    you have room for it.

    Args:
        top_k: keep only the highest-weight terms per vector. SPLADE emits a few
            hundred non-zeros per passage; pruning shrinks the inverted index at
            a small recall cost. ``None`` keeps everything.
    """

    def __init__(
        self,
        model_name: str = POLISH_SPLADE,
        *,
        device: str | None = None,
        half: bool | None = None,
        max_length: int = 512,
        batch_size: int = 8,
        top_k: int | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.top_k = top_k

        self.device = resolve_device(device)
        self.dtype = resolve_dtype(self.device, half)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForMaskedLM.from_pretrained(model_name)
        # Cast after loading rather than via a from_pretrained kwarg, whose name
        # changed between transformers major versions.
        self.model = model.to(device=self.device, dtype=self.dtype).eval()
        self._torch = torch

    def __repr__(self) -> str:
        return (
            f"SpladeSparseEncoder({self.model_name!r}, "
            f"{describe_runtime(self.device, self.dtype)}, top_k={self.top_k})"
        )

    def vector_params(self) -> models.SparseVectorParams:
        # No IDF: SPLADE weights already encode term importance.
        return models.SparseVectorParams()

    def encode_passages(self, texts: Sequence[str]) -> list[models.SparseVector]:
        out: list[models.SparseVector] = []
        for start in range(0, len(texts), self.batch_size):
            chunk = list(texts[start : start + self.batch_size])
            for row in self._pool(chunk):
                out.append(self._to_sparse(row))
        return out

    def encode_query(self, text: str) -> models.SparseVector:
        return self._to_sparse(self._pool([text])[0])

    def decode(
        self, vector: models.SparseVector, limit: int = 10
    ) -> list[tuple[str, float]]:
        """Map indices back to tokens, heaviest first. For inspecting expansion."""
        terms = zip(vector.indices, vector.values, strict=True)
        pairs = sorted(terms, key=lambda p: p[1], reverse=True)[:limit]
        tokens = self.tokenizer.convert_ids_to_tokens([i for i, _ in pairs])
        return list(zip(tokens, (round(v, 3) for _, v in pairs), strict=True))

    def _pool(self, texts: list[str]):
        torch = self._torch
        with torch.inference_mode():
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**encoded).logits
            mask = encoded["attention_mask"].unsqueeze(-1)
            weighted = torch.log1p(torch.relu(logits)) * mask
            return weighted.max(dim=1).values.float()

    def _to_sparse(self, row) -> models.SparseVector:
        torch = self._torch
        indices = torch.nonzero(row, as_tuple=False).squeeze(-1)
        if self.top_k is not None and indices.numel() > self.top_k:
            keep = torch.topk(row[indices], self.top_k).indices
            indices = torch.sort(indices[keep]).values
        return models.SparseVector(
            indices=indices.cpu().tolist(),
            values=row[indices].cpu().tolist(),
        )
