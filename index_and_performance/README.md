# Qdrant HNSW Indexing & Performance Guide

Reference notes for tuning Qdrant collections, based on
[Qdrant Essentials — Day 2](https://qdrant.tech/course/essentials/day-2/),
the [Optimization guide](https://qdrant.tech/documentation/ops-optimization/optimize/),
and the [HNSW paper (Malkov & Yashunin, 2016)](https://arxiv.org/pdf/1603.09320).

## HNSW in one paragraph

HNSW (Hierarchical Navigable Small World) is a multi-layer proximity graph.
Search starts at a sparse top layer, greedily descends to the dense bottom
layer, and expands a candidate beam there. Two knobs dominate:

| Parameter | Meaning | Effect |
|---|---|---|
| `m` | Max links per node per layer | ↑ m = better recall, more RAM (~`m × 2 × 4 bytes/vector` for layer 0), slower build |
| `ef_construct` | Beam width during **index build** | ↑ = better graph quality, slower ingest |
| `hnsw_ef` (query-time) | Beam width during **search** | ↑ = better recall, higher latency |
| `full_scan_threshold` | Below this many points (in KB of vectors), use exact scan | Avoids graph overhead on tiny segments |

Special case: **`m=0` disables HNSW entirely** — no graph is built, searches
fall back to full scan.

## The three profiles

All three scripts ingest the same 100K-point dbpedia dataset
(OpenAI `text-embedding-3-large`, 1536 dims, cosine) and run the same
benchmark: warm-up → baseline search → filtered search without payload
index → create text index → filtered search with index.

### 1. `fast_initial_upload.py` — bulk ingest pattern

```python
{"m": 0, "ef_construct": 100}
```

- `m=0` skips HNSW link creation during upload → **5–10× faster ingest**,
  because graph insertion (the expensive part of HNSW writes) is deferred.
- **This is a two-phase pattern**: after ingest completes, the script
  updates the collection with a real `m` to trigger index build in the
  background, then polls until the collection status is `GREEN`:

  ```python
  client.update_collection(
      collection_name=collection,
      hnsw_config=models.HnswConfigDiff(m=16),
  )
  ```

- Until phase 2 runs, every query is a full scan — fine for validation,
  terrible for production latency.
- Use for: initial data migrations, nightly full reloads, any write-heavy
  bootstrap phase.

### 2. `balanced.py` — general-purpose default

```python
{"m": 8, "ef_construct": 100}
```

- Moderate connectivity: decent recall, moderate RAM, reasonable build time.
- Qdrant's own default is `m=16`; `m=8` trades a little recall for ~half
  the graph memory. Good starting point when you don't yet know your
  recall/latency targets.
- Tune search-time recall with `hnsw_ef` per query
  (`SearchParams(hnsw_ef=100)`) rather than rebuilding the index.
- Use for: typical production workloads, mixed read/write.

### 3. `memory_optimized.py` — low-RAM pattern

```python
{"m": 8, "ef_construct": 100}  # plus on_disk + int8 quantization
```

Applies the memory optimizations from the Qdrant optimization guide:

```python
vectors_config=models.VectorParams(
    size=1536,
    distance=models.Distance.COSINE,
    on_disk=True,                      # original vectors on disk (memmap)
),
hnsw_config=models.HnswConfigDiff(
    m=8,                               # smaller graph in RAM
    ef_construct=100,
    on_disk=True,                      # HNSW graph itself on disk
),
quantization_config=models.ScalarQuantization(
    scalar=models.ScalarQuantizationConfig(
        type=models.ScalarType.INT8,   # 4× smaller vectors in RAM
        always_ram=True,               # keep quantized copy in RAM for speed
    ),
),
```

- Trade-off: on-disk vectors/graph add disk-I/O latency (mitigate with fast
  NVMe + quantized vectors in RAM); scalar int8 quantization costs a small
  amount of accuracy, recoverable with rescoring — the script's queries use
  `QuantizationSearchParams(rescore=True, oversampling=2.0)`.
- Use for: large collections that don't fit in RAM, cost-sensitive
  deployments.

## Payload indexing (second half of each script)

- Filtering **without** a payload index forces Qdrant to check the filter
  against raw payloads during graph traversal → large overhead
  (requires `unindexed_filtering_retrieve=True` / strict mode off).
- `create_payload_index(..., TextIndexParams(tokenizer="word"))` builds a
  full-text index so filter candidates are resolved cheaply; Qdrant's
  filterable HNSW also uses payload indexes to keep graph search efficient
  under filters.
- Rule of thumb: **index every field you filter on**, and create indexes
  *before* bulk upload when possible.

## Quick decision table

| Scenario | Profile |
|---|---|
| One-off bulk load, then read-heavy | `fast_initial_upload` (m=0 → then set m=16) |
| Standard app, unknown workload | `balanced` (m=8–16, ef_construct=100) |
| Dataset >> RAM, cost matters | `memory_optimized` (on_disk + int8 quantization) |
| Max recall, latency-tolerant | m=32–64, ef_construct=256+, high `hnsw_ef` |

> Note: `create_collection` is skipped when a collection already exists, so
> config changes require deleting the collection first
> (`client.delete_collection(...)`) or applying `update_collection`.
