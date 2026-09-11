"""
Local Dense and Sparse Embedding Pipeline for Qdrant Hybrid Search.

Demonstrates how to replace high-level `models.Document(text=..., model="...")`
with locally hosted dense and sparse models:
1. Local Dense Encoder (e.g., SentenceTransformer) -> list[float]
2. Local Sparse Encoder (SPLADE neural sparse or BM25/TF-IDF token hashing) -> models.SparseVector(indices=..., values=...)
3. Batch Point Construction & Upsert into Qdrant
4. Hybrid Querying using RRF (Reciprocal Rank Fusion)
"""

import os
import sys
import re
import uuid
import binascii
from collections import Counter
from typing import Any, Dict, List, Tuple
from dotenv import load_dotenv

import numpy as np
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()


# ============================================================================
# 1. Local Dense Encoder
# ============================================================================
class LocalDenseEncoder:
    """Encodes texts into dense float vectors using a locally loaded model."""

    def __init__(self, model_name_or_path: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"):
        self.model = SentenceTransformer(model_name_or_path, device=device)
        self.dimension = self.model.get_embedding_dimension()

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """Encodes a batch of texts to a list of float vectors."""
        embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings.tolist()

    def encode_query(self, query: str) -> List[float]:
        """Encodes a single query string to a dense vector."""
        return self.model.encode(query, convert_to_numpy=True, normalize_embeddings=True).tolist()


# ============================================================================
# 2. Local Sparse Encoders
# ============================================================================

class LocalBM25SparseEncoder:
    """
    Lightweight local lexical encoder that produces token-frequency sparse vectors.
    
    Pairs with Qdrant collection configured with:
        sparse_vectors_config={"sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)}
    
    Qdrant computes corpus-level IDF dynamically at search time, so the local
    encoder only needs to emit token frequencies as values and token hashes as indices.
    """

    def __init__(self):
        # Regex word tokenizer
        self.tokenizer = re.compile(r"(?u)\b\w\w+\b")

    def _hash_token(self, token: str) -> int:
        """Hashes a string token into an unsigned 32-bit integer (0 to 4,294,967,295)."""
        # Python's built-in hash modulo 2^32 or standard crc32/murmur
        import binascii
        return binascii.crc32(token.lower().encode("utf-8"))

    def encode_text(self, text: str) -> models.SparseVector:
        """Encodes a single text into a Qdrant SparseVector."""
        tokens = self.tokenizer.findall(text)
        if not tokens:
            return models.SparseVector(indices=[], values=[])

        counts = Counter(tokens)
        token_hashes = [self._hash_token(token) for token in counts.keys()]
        term_frequencies = [float(freq) for freq in counts.values()]

        # Qdrant requires sorted or unique indices
        sorted_pairs = sorted(zip(token_hashes, term_frequencies), key=lambda x: x[0])
        indices = [pair[0] for pair in sorted_pairs]
        values = [pair[1] for pair in sorted_pairs]

        return models.SparseVector(indices=indices, values=values)

    def encode_batch(self, texts: List[str]) -> List[models.SparseVector]:
        """Encodes a batch of texts into sparse vectors."""
        return [self.encode_text(text) for text in texts]


class LocalSpladeSparseEncoder:
    """
    Neural sparse encoder using SPLADE or BGE-M3 sparse architecture locally via PyTorch / Transformers.
    
    Pairs with Qdrant collection configured with:
        sparse_vectors_config={"sparse": models.SparseVectorParams()} (No modifier needed)
    
    Vocabulary IDs become indices, and log(1 + ReLU(logits)) become values.
    """

    def __init__(self, model_name_or_path: str = "naver/splade-cocondenser-ensembledistil", device: str = "cpu"):
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name_or_path).to(self.device)
        self.model.eval()

    def encode_batch(self, texts: List[str]) -> List[models.SparseVector]:
        """Encodes a batch of texts into SPLADE SparseVectors."""
        import torch

        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  # shape: [batch_size, seq_len, vocab_size]
            
            # SPLADE pooling: max over sequence length of log(1 + relu(logits))
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            relu_log = torch.log(1 + torch.relu(logits)) * attention_mask
            splade_vectors = torch.max(relu_log, dim=1).values  # shape: [batch_size, vocab_size]

        sparse_vectors = []
        for vec in splade_vectors:
            non_zero_indices = torch.nonzero(vec).squeeze(-1)
            non_zero_values = vec[non_zero_indices]
            sparse_vectors.append(
                models.SparseVector(
                    indices=non_zero_indices.cpu().tolist(),
                    values=non_zero_values.cpu().tolist()
                )
            )
        return sparse_vectors


# ============================================================================
# 3. Batch Point Construction & Upsert Pipeline
# ============================================================================

def build_points_in_batches(
    documents: List[Dict[str, Any]],
    dense_encoder: LocalDenseEncoder,
    sparse_encoder: LocalBM25SparseEncoder,
    batch_size: int = 64,
) -> List[models.PointStruct]:
    """
    Constructs Qdrant points in batches by computing dense and sparse vectors locally.
    
    Each document dict should have at least a "text" key and an optional "id" / metadata.
    """
    all_points: List[models.PointStruct] = []

    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        batch_texts = [doc["text"] for doc in batch]

        # 1. Local dense embeddings
        dense_vectors = dense_encoder.encode_batch(batch_texts)

        # 2. Local sparse embeddings
        sparse_vectors = sparse_encoder.encode_batch(batch_texts)

        # 3. Assemble PointStruct instances
        for doc, dense_vec, sparse_vec in zip(batch, dense_vectors, sparse_vectors):
            point_id = doc.get("id") or str(uuid.uuid4())
            payload = {k: v for k, v in doc.items() if k != "id"}

            point = models.PointStruct(
                id=point_id,
                vector={
                    "dense": dense_vec,
                    "sparse": sparse_vec,  # models.SparseVector(indices=..., values=...)
                },
                payload=payload,
            )
            all_points.append(point)

    return all_points


def upsert_batched(
    client: QdrantClient,
    collection_name: str,
    points: List[models.PointStruct],
    batch_size: int = 128,
):
    """Upserts points into Qdrant in batches to optimize network and memory usage."""
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        client.upsert(
            collection_name=collection_name,
            points=batch,
            wait=True,
        )
        print(f"Upserted batch {i // batch_size + 1} ({len(batch)} points)")


# ============================================================================
# 4. Hybrid Search with Local Query Encoding and RRF Fusion
# ============================================================================

def hybrid_search(
    client: QdrantClient,
    collection_name: str,
    query_text: str,
    dense_encoder: LocalDenseEncoder,
    sparse_encoder: LocalBM25SparseEncoder,
    limit: int = 5,
):
    """
    Performs hybrid search by locally encoding the query for both dense and sparse,
    then combining the candidate sets in Qdrant using Reciprocal Rank Fusion (RRF).
    """
    # 1. Encode query with local models
    dense_query = dense_encoder.encode_query(query_text)
    sparse_query = sparse_encoder.encode_text(query_text)

    # 2. Query Qdrant with prefetch for both modalities and RRF fusion
    response = client.query_points(
        collection_name=collection_name,
        prefetch=[
            models.Prefetch(
                query=dense_query,
                using="dense",
                limit=limit * 3,
            ),
            models.Prefetch(
                query=sparse_query,
                using="sparse",
                limit=limit * 3,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
    )
    return response


# ============================================================================
# 5. Example Run / Demonstration
# ============================================================================

if __name__ == "__main__":
    # Sample documents (similar to cheese demo in dense_sparse_search.py)
    docs = [
        {"id": 1, "text": "Aged Gouda develops a crystalline texture and nutty flavor profile after 18 months of maturation."},
        {"id": 2, "text": "Mature Gouda cheese becomes grainy and develops a rich, buttery taste with extended aging."},
        {"id": 3, "text": "Brie cheese features a soft, creamy interior surrounded by an edible white rind."},
        {"id": 4, "text": "This French cheese has a flowing, buttery center encased in a bloomy white crust."},
        {"id": 5, "text": "Fresh mozzarella pairs beautifully with ripe tomatoes and basil leaves."},
        {"id": 6, "text": "Classic Margherita pizza topped with tomato sauce, mozzarella, and fresh basil."},
        {"id": 7, "text": "Parmesan requires at least 12 months of cave aging to develop its signature sharp taste."},
        {"id": 8, "text": "Parmigiano-Reggiano's distinctive piquant flavor comes from extended maturation in controlled environments."},
        {"id": 9, "text": "Grilled cheese sandwiches are the ultimate American comfort food for cold winter days."},
        {"id": 10, "text": "Croque Monsieur combines ham and Gruyère in France's answer to the toasted cheese sandwich."},
    ]

    print("Initializing local dense and sparse encoders...")
    dense_encoder = LocalDenseEncoder(model_name_or_path="sentence-transformers/all-MiniLM-L6-v2")
    sparse_encoder = LocalBM25SparseEncoder()

    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        timeout=300,
    )
    collection_name = "local_hybrid_collection_11092026_1"

    try:
        if client.collection_exists(collection_name=collection_name):
            client.delete_collection(collection_name=collection_name)

        print(f"Creating collection '{collection_name}' with dense and sparse vector configurations...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": models.VectorParams(
                    size=dense_encoder.dimension,
                    distance=models.Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    modifier=models.Modifier.IDF,  # Dynamically computed by Qdrant
                ),
            },
        )

        print("Building points in batches locally...")
        points = build_points_in_batches(docs, dense_encoder, sparse_encoder, batch_size=4)

        print("Upserting points to Qdrant...")
        upsert_batched(client, collection_name, points, batch_size=5)

        # Perform hybrid search
        query = "crunchy aged cheese with crystals"
        print(f"\nExecuting Hybrid Search for: '{query}'")
        search_results = hybrid_search(client, collection_name, query, dense_encoder, sparse_encoder, limit=3)

        print("\nTop Results:")
        for rank, point in enumerate(search_results.points, 1):
            print(f"{rank}. [Score: {point.score:.4f}] Doc ID {point.id}: {point.payload.get('text')}")
    finally:
        client.close()

    # Clean exit
    sys.stdout.flush()
    sys.stderr.flush()
    # os._exit(0)


# This error occurs because `models.Document(...)` relies on **FastEmbed** (or Qdrant Cloud), which only supports pre-built ONNX models (like `Qdrant/bm25` or `prithivida/Splade_PP_en_v1`). `sdadas/polish-splade` is not supported by FastEmbed.
#
# ### Fix: Encode with `transformers` and use `models.SparseVector`
#
# Compute the sparse vectors using PyTorch/HuggingFace and pass them directly:
#
# ```python
# import numpy as np
# import torch
# from qdrant_client import models
# from transformers import AutoModelForMaskedLM, AutoTokenizer
#
# tokenizer = AutoTokenizer.from_pretrained("sdadas/polish-splade")
# model = AutoModelForMaskedLM.from_pretrained("sdadas/polish-splade")
#
#
# def encode_splade(text: str) -> models.SparseVector:
#   inputs = tokenizer(
#       [text], padding=True, truncation=True, return_tensors="pt"
#   )
#   outputs = model(**inputs)
#   mask = inputs.attention_mask.unsqueeze(-1)
#   vec = (
#       torch.max(torch.log(torch.relu(outputs.logits) + 1) * mask, dim=1)[0]
#       .squeeze()
#       .detach()
#       .cpu()
#       .numpy()
#   )
#   idx = np.nonzero(vec)[0]
#   return models.SparseVector(indices=idx.tolist(), values=vec[idx].tolist())
# ```
#
# In your upsert:
#
# ```python
# vector = {
#     "dense": models.Document(
#         text=doc, model="sentence-transformers/all-MiniLM-L6-v2"
#     ),
#     "sparse": encode_splade(doc),
# }
# ```
#
# *(Alternatively, use `"Qdrant/bm25"` in `models.Document` if you want automatic FastEmbed inference).*