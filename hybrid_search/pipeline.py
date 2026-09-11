"""Collection setup, batched upsert and RRF search.

Encoder-agnostic: anything satisfying the DenseEncoder / SparseEncoder shape
works, so swapping SPLADE for BM25 or one Polish model for another needs no
change here.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Iterator, Sequence
from typing import Any

from qdrant_client import QdrantClient, models

__all__ = [
    "DENSE_VECTOR",
    "SPARSE_VECTOR",
    "ensure_collection",
    "index_documents",
    "hybrid_search",
]

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"


def _chunks(items: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def ensure_collection(
    client: QdrantClient,
    collection_name: str,
    *,
    dense_encoder,
    sparse_encoder,
    recreate: bool = False,
) -> bool:
    """Create the collection if needed. Returns True if it was created."""
    if client.collection_exists(collection_name):
        if not recreate:
            return False
        client.delete_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            DENSE_VECTOR: models.VectorParams(
                size=dense_encoder.dim,
                distance=models.Distance.COSINE,
            ),
        },
        sparse_vectors_config={
            # The encoder decides whether IDF applies.
            SPARSE_VECTOR: sparse_encoder.vector_params(),
        },
    )
    return True


def index_documents(
    client: QdrantClient,
    collection_name: str,
    documents: Sequence[dict[str, Any]],
    *,
    dense_encoder,
    sparse_encoder=None,
    sparse_vectors: Sequence[models.SparseVector] | None = None,
    batch_size: int = 32,
) -> int:
    """Encode and upsert documents.

    Each document needs a ``text`` key; ``id`` is optional. Everything else
    becomes payload.

    Pass ``sparse_vectors`` to reuse vectors encoded earlier, which avoids
    re-running SPLADE when indexing the same corpus against several dense
    models. Otherwise pass ``sparse_encoder``.
    """
    if sparse_vectors is None:
        if sparse_encoder is None:
            raise ValueError("Provide sparse_encoder or sparse_vectors.")
        sparse_vectors = sparse_encoder.encode_passages(
            [doc["text"] for doc in documents]
        )
    if len(sparse_vectors) != len(documents):
        raise ValueError("sparse_vectors length does not match documents.")

    total = 0
    for offset in range(0, len(documents), batch_size):
        batch = documents[offset : offset + batch_size]
        sparse_batch = sparse_vectors[offset : offset + batch_size]
        dense_batch = dense_encoder.encode_passages([doc["text"] for doc in batch])

        points = [
            models.PointStruct(
                id=doc.get("id") or str(uuid.uuid4()),
                vector={DENSE_VECTOR: dense, SPARSE_VECTOR: sparse},
                payload={k: v for k, v in doc.items() if k != "id"},
            )
            for doc, dense, sparse in zip(batch, dense_batch, sparse_batch, strict=True)
        ]
        client.upsert(collection_name=collection_name, points=points, wait=True)
        total += len(points)
    return total


def hybrid_search(
    client: QdrantClient,
    collection_name: str,
    query: str,
    *,
    dense_encoder,
    sparse_encoder,
    limit: int = 5,
    candidates: int | None = None,
    rrf_k: int | None = None,
    weights: Iterable[float] | None = None,
) -> list[models.ScoredPoint]:
    """Search both vectors and fuse with RRF.

    Args:
        candidates: per-branch prefetch depth. Must be >= limit or fusion can
            return fewer results than asked for. Defaults to ``limit * 4``.
        weights: relative weight per branch, dense first. Leave unset unless you
            have measured better values on your own eval set.
    """
    depth = candidates or limit * 4
    weight_list = list(weights) if weights is not None else None
    if weight_list is not None and len(weight_list) != 2:
        raise ValueError("weights must contain one value per branch (dense, sparse).")

    response = client.query_points(
        collection_name=collection_name,
        prefetch=[
            models.Prefetch(
                query=dense_encoder.encode_query(query),
                using=DENSE_VECTOR,
                limit=depth,
            ),
            models.Prefetch(
                query=sparse_encoder.encode_query(query),
                using=SPARSE_VECTOR,
                limit=depth,
            ),
        ],
        query=models.RrfQuery(rrf=models.Rrf(k=rrf_k, weights=weight_list)),
        limit=limit,
        with_payload=True,
    )
    return response.points


def single_vector_search(
    client: QdrantClient,
    collection_name: str,
    query: str,
    *,
    encoder,
    using: str,
    limit: int = 5,
) -> list[models.ScoredPoint]:
    """Search one vector only. Useful for showing what fusion adds."""
    response = client.query_points(
        collection_name=collection_name,
        query=encoder.encode_query(query),
        using=using,
        limit=limit,
        with_payload=True,
    )
    return response.points
