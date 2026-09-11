"""Polish hybrid search: SPLADE sparse vectors + a Polish dense retriever.

Qdrant/bm25 has no Polish lemmatisation, so inflected queries miss. This uses
sdadas/polish-splade for the lexical side, encoded locally, since FastEmbed has
no ONNX build for it.

Run a local Qdrant first:

    docker compose up -d

Then:

    python hybrid_search/polish_hybrid_search.py
    python hybrid_search/polish_hybrid_search.py --models mmlw-base silver
    python hybrid_search/polish_hybrid_search.py --models stella --device cuda
    python hybrid_search/polish_hybrid_search.py --sparse bm25   # for contrast
"""

from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient

from hybrid_search.encoders import (
    POLISH_DENSE_MODELS,
    Bm25SparseEncoder,
    DenseEncoder,
    SpladeSparseEncoder,
    describe_runtime,
)
from hybrid_search.pipeline import (
    DENSE_VECTOR,
    SPARSE_VECTOR,
    ensure_collection,
    hybrid_search,
    index_documents,
    single_vector_search,
)

_CORPUS: list[tuple[int, str, str]] = [
    (1, "ubezpieczenia",
     "Polisa OC obejmuje szkody wyrządzone osobom trzecim przez kierującego pojazdem."),
    (2, "ubezpieczenia",
     "Autocasco chroni pojazd właściciela przed kradzieżą oraz zniszczeniem."),
    (3, "gwarancje",
     "Gwarancja producenta na sprzęt elektroniczny wynosi dwa lata."),
    (4, "gwarancje",
     "Reklamację wadliwego urządzenia można złożyć w ciągu dwóch lat od zakupu."),
    (5, "odpady",
     "Opłata za gospodarowanie odpadami komunalnymi zależy od liczby mieszkańców."),
    (6, "odpady",
     "Segregowane śmieci należy wystawiać przed posesję w dniu wywozu."),
    (7, "podatki",
     "Deklarację podatkową PIT trzeba złożyć do trzydziestego kwietnia."),
    (8, "podatki",
     "Rozliczenie roczne można wysłać elektronicznie przez usługę Twój e-PIT."),
    (9, "dokumenty",
     "Wniosek o wydanie paszportu składa się osobiście w punkcie paszportowym."),
    (10, "dokumenty",
     "Dowód osobisty dla dziecka wydawany jest na wniosek jednego z rodziców."),
    (11, "pojazdy",
     "Rejestracja samochodu sprowadzonego z zagranicy wymaga tłumaczenia dokumentów."),
    (12, "pojazdy",
     "Przegląd techniczny pojazdu osobowego wykonuje się raz w roku."),
]

DOCUMENTS = [
    {"id": doc_id, "topic": topic, "text": text} for doc_id, topic, text in _CORPUS
]

# Each query uses word forms that do not appear literally in the corpus.
QUERIES = [
    "jak ubezpieczyć samochód",
    "kiedy złożyć zeznanie podatkowe",
    "ile kosztuje wywóz odpadów",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["mmlw-base"],
        choices=sorted(POLISH_DENSE_MODELS),
        help="dense models to try (default: mmlw-base)",
    )
    parser.add_argument(
        "--sparse",
        default="splade",
        choices=["splade", "bm25"],
        help="sparse encoder (default: splade)",
    )
    parser.add_argument("--device", help="cuda, mps or cpu (default: autodetect)")
    parser.add_argument(
        "--precision",
        default="auto",
        choices=["auto", "half", "full"],
        help="auto uses half precision on CUDA only",
    )
    parser.add_argument("--sparse-top-k", type=int, default=256,
                        help="keep N heaviest SPLADE terms per vector, 0 for all")
    parser.add_argument("--sparse-batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--recreate", action="store_true",
                        help="drop and rebuild collections")
    parser.add_argument("--list-models", action="store_true")
    return parser.parse_args()


def show_models() -> None:
    for slug, spec in POLISH_DENSE_MODELS.items():
        print(f"{slug:12} {spec.dim:5}d  {spec.name}")
        print(f"{'':12} {spec.notes}")


def show_results(label: str, points) -> None:
    print(f"  {label}")
    if not points:
        print("    no matches")
        return
    for rank, point in enumerate(points, 1):
        text = point.payload["text"]
        print(f"    {rank}. {point.score:.4f}  [{point.payload['topic']}] {text}")


def main() -> None:
    args = parse_args()
    if args.list_models:
        show_models()
        return

    load_dotenv()
    half = {"auto": None, "half": True, "full": False}[args.precision]

    if args.sparse == "splade":
        sparse_encoder = SpladeSparseEncoder(
            device=args.device,
            half=half,
            batch_size=args.sparse_batch_size,
            top_k=args.sparse_top_k or None,
        )
        print(f"sparse: {sparse_encoder}")
    else:
        sparse_encoder = Bm25SparseEncoder()
        print("sparse: Bm25SparseEncoder (hashed tokens, Qdrant applies IDF)")

    # Encode the corpus once and reuse it for every dense model.
    texts = [doc["text"] for doc in DOCUMENTS]
    sparse_vectors = sparse_encoder.encode_passages(texts)
    average_terms = sum(len(v.indices) for v in sparse_vectors) / len(sparse_vectors)
    print(f"sparse terms per document: {average_terms:.0f} average")

    if isinstance(sparse_encoder, SpladeSparseEncoder):
        expansion = sparse_encoder.decode(sparse_encoder.encode_query(QUERIES[0]), 8)
        terms = ", ".join(f"{t}:{w}" for t, w in expansion)
        print(f"query expansion for {QUERIES[0]!r}\n  {terms}")

    client = QdrantClient(
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        api_key=os.getenv("QDRANT_API_KEY") or None,
        timeout=300,
    )

    for slug in args.models:
        spec = POLISH_DENSE_MODELS[slug]
        print(f"\n=== {slug} ({spec.name})")
        dense_encoder = DenseEncoder(spec, device=args.device, half=half)
        print(f"dense: {dense_encoder.dim}d on "
              f"{describe_runtime(dense_encoder.device, dense_encoder.dtype)}")

        collection = f"pl_{args.sparse}_{slug.replace('-', '_')}"
        created = ensure_collection(
            client,
            collection,
            dense_encoder=dense_encoder,
            sparse_encoder=sparse_encoder,
            recreate=args.recreate,
        )
        if created:
            count = index_documents(
                client,
                collection,
                DOCUMENTS,
                dense_encoder=dense_encoder,
                sparse_vectors=sparse_vectors,
            )
            print(f"indexed {count} documents into {collection}")
        else:
            print(f"reusing {collection} (--recreate to rebuild)")

        for query in QUERIES:
            print(f"\n  query: {query}")
            show_results(
                "dense only",
                single_vector_search(
                    client, collection, query,
                    encoder=dense_encoder, using=DENSE_VECTOR, limit=args.limit,
                ),
            )
            show_results(
                "sparse only",
                single_vector_search(
                    client, collection, query,
                    encoder=sparse_encoder, using=SPARSE_VECTOR, limit=args.limit,
                ),
            )
            show_results(
                "hybrid (RRF)",
                hybrid_search(
                    client, collection, query,
                    dense_encoder=dense_encoder,
                    sparse_encoder=sparse_encoder,
                    limit=args.limit,
                ),
            )


if __name__ == "__main__":
    main()
