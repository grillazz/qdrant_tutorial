"""Dense encoders for Polish retrieval.

Each model in this family needs its own query/passage prefix. Getting a prefix
wrong costs several points of NDCG silently, so the prefixes live in a registry
next to the model id instead of at the call site.

Prefixes taken from the model cards:
    sdadas/mmlw-retrieval-roberta-base   query "zapytanie: "
    ipipan/silver-retriever-base-v1.1    query "Pytanie: ", passage "</s>"
    sdadas/stella-pl-retrieval-mini-8k   query "Instruct: ...\\nQuery: "
    sdadas/stella-pl-retrieval-8k        query "Instruct: ...\\nQuery: "
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from sentence_transformers import SentenceTransformer

from hybrid_search.encoders.device import describe_runtime, resolve_device, resolve_dtype

__all__ = ["DenseModel", "DenseEncoder", "POLISH_DENSE_MODELS"]

STELLA_QUERY_PREFIX = (
    "Instruct: Given a web search query, retrieve relevant passages that "
    "answer the query.\nQuery: "
)


@dataclass(frozen=True, slots=True)
class DenseModel:
    """Everything needed to use a dense retriever correctly."""

    slug: str
    name: str
    dim: int
    query_prefix: str = ""
    passage_prefix: str = ""
    trust_remote_code: bool = False
    max_seq_length: int | None = None
    notes: str = ""


POLISH_DENSE_MODELS: dict[str, DenseModel] = {
    m.slug: m
    for m in (
        DenseModel(
            slug="mmlw-base",
            name="sdadas/mmlw-retrieval-roberta-base",
            dim=768,
            query_prefix="zapytanie: ",
            max_seq_length=512,
            notes="Polish RoBERTa, apache-2.0. Fast, good default.",
        ),
        DenseModel(
            slug="silver",
            name="ipipan/silver-retriever-base-v1.1",
            dim=768,
            query_prefix="Pytanie: ",
            passage_prefix="</s>",
            max_seq_length=512,
            notes="HerBERT-base, cc-by-sa-4.0. Tuned on PolQA/MAUPQA.",
        ),
        DenseModel(
            slug="stella-mini",
            name="sdadas/stella-pl-retrieval-mini-8k",
            dim=1024,
            query_prefix=STELLA_QUERY_PREFIX,
            trust_remote_code=True,
            notes="stella_en_400M_v5 base, 8k context. Needs remote code.",
        ),
        DenseModel(
            slug="stella",
            name="sdadas/stella-pl-retrieval-8k",
            dim=1024,
            query_prefix=STELLA_QUERY_PREFIX,
            trust_remote_code=True,
            notes="stella_en_1.5B_v5 base, ~6 GB download. Slow on CPU.",
        ),
    )
}


class DenseEncoder:
    """SentenceTransformer wrapper that applies the model's own prefixes.

    Embeddings are L2-normalised, so the collection should use cosine distance.
    """

    def __init__(
        self,
        model: DenseModel | str,
        *,
        device: str | None = None,
        half: bool | None = None,
        batch_size: int = 16,
    ) -> None:
        self.spec = POLISH_DENSE_MODELS[model] if isinstance(model, str) else model
        self.batch_size = batch_size
        self.device = resolve_device(device)
        self.dtype = resolve_dtype(self.device, half)

        self.model = SentenceTransformer(
            self.spec.name,
            device=str(self.device),
            trust_remote_code=self.spec.trust_remote_code,
        )
        if self.spec.max_seq_length:
            self.model.max_seq_length = self.spec.max_seq_length
        if self.dtype is not None and str(self.dtype) != "torch.float32":
            self.model = self.model.to(self.dtype)

        self.dim = self._resolve_dim()
        if self.dim != self.spec.dim:
            raise ValueError(
                f"{self.spec.name} reports {self.dim} dims, registry says "
                f"{self.spec.dim}. Update POLISH_DENSE_MODELS."
            )

    def __repr__(self) -> str:
        return (
            f"DenseEncoder({self.spec.slug}, {self.dim}d, "
            f"{describe_runtime(self.device, self.dtype)})"
        )

    def encode_passages(self, texts: Sequence[str]) -> list[list[float]]:
        prefixed = [self.spec.passage_prefix + t for t in texts]
        return self._encode(prefixed)

    def encode_query(self, text: str) -> list[float]:
        return self._encode([self.spec.query_prefix + text])[0]

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.astype("float32").tolist()

    def _resolve_dim(self) -> int:
        getter = getattr(self.model, "get_sentence_embedding_dimension", None)
        if callable(getter) and (dim := getter()) is not None:
            return int(dim)
        return len(self.encode_query("test"))
