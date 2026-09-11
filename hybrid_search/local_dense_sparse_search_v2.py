"""Local dense + sparse hybrid search, English corpus.

Replaces `models.Document(text=..., model=...)` with encoders that run in this
process, so any HuggingFace model works, not only the ones FastEmbed ships.

Encoders live in `encoders/`, Qdrant plumbing in `pipeline.py`. For Polish see
`polish_hybrid_search.py`.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient

from hybrid_search.encoders import Bm25SparseEncoder, DenseEncoder, DenseModel
from hybrid_search.pipeline import ensure_collection, hybrid_search, index_documents

MINILM = DenseModel(
    slug="minilm-l6",
    name="sentence-transformers/all-MiniLM-L6-v2",
    dim=384,
    max_seq_length=256,
    notes="English only. Small and fast.",
)

_CORPUS: list[tuple[int, str]] = [
    (1, "Aged Gouda develops a crystalline texture and nutty flavor after 18 months."),
    (2, "Mature Gouda becomes grainy and develops a rich, buttery taste with age."),
    (3, "Brie has a soft, creamy interior surrounded by an edible white rind."),
    (4, "This French cheese has a flowing, buttery center in a bloomy white crust."),
    (5, "Fresh mozzarella pairs well with ripe tomatoes and basil leaves."),
    (6, "Margherita pizza is topped with tomato sauce, mozzarella and fresh basil."),
    (7, "Parmesan needs at least 12 months of cave aging for its sharp taste."),
    (8, "Parmigiano-Reggiano gets its piquant flavor from long maturation."),
    (9, "Grilled cheese sandwiches are American comfort food for cold days."),
    (10, "Croque Monsieur combines ham and Gruyere in a French toasted sandwich."),
]

DOCUMENTS = [{"id": doc_id, "text": text} for doc_id, text in _CORPUS]

COLLECTION = "local_hybrid_collection"


def main() -> None:
    load_dotenv()

    dense_encoder = DenseEncoder(MINILM)
    sparse_encoder = Bm25SparseEncoder()

    client = QdrantClient(
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        api_key=os.getenv("QDRANT_API_KEY") or None,
        timeout=300,
    )

    if ensure_collection(
        client,
        COLLECTION,
        dense_encoder=dense_encoder,
        sparse_encoder=sparse_encoder,
    ):
        count = index_documents(
            client,
            COLLECTION,
            DOCUMENTS,
            dense_encoder=dense_encoder,
            sparse_encoder=sparse_encoder,
        )
        print(f"indexed {count} documents")
    else:
        print(f"reusing {COLLECTION}")

    query = "crunchy aged cheese with crystals"
    print(f"\nquery: {query}")
    for rank, point in enumerate(
        hybrid_search(
            client,
            COLLECTION,
            query,
            dense_encoder=dense_encoder,
            sparse_encoder=sparse_encoder,
            limit=3,
        ),
        1,
    ):
        print(f"{rank}. {point.score:.4f}  {point.payload['text']}")


if __name__ == "__main__":
    main()
