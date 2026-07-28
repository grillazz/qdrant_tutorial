import numpy as np
import requests
from datasets import load_dataset
from qdrant_client import QdrantClient, models
from tqdm import tqdm
import openai
import time
import os

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=300,
)
ds = load_dataset("Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-100K")

config = {"name": "memory_optimized", "m": 8, "ef_construct": 100}  # m=0 = ingest-only

collection = str(config["name"])

if not client.collection_exists(collection_name=collection):
    client.create_collection(
        collection_name=collection,
        vectors_config=models.VectorParams(
            size=1536,
            distance=models.Distance.COSINE
        ),
        hnsw_config=models.HnswConfigDiff(
            m=int(config["m"]),
            ef_construct=int(config["ef_construct"]),
            full_scan_threshold=10000
        ),
        strict_mode_config=models.StrictModeConfig(
            enabled=False,
            unindexed_filtering_retrieve=True  # Allow filtering without indexes
        )
    )
collection_info = client.get_collection(collection_name=collection)
print(collection_info)

batch_size = 10000
total_points = len(ds['train'])

print(f"Uploading {total_points} points in batches of {batch_size}")

def upload_batch_without_indexes(start_idx, end_idx):
    points = []
    for i in range(start_idx, min(end_idx, total_points)):
        example = ds['train'][i]

        # Get the embedding
        embedding = example['text-embedding-3-large-1536-embedding']

        # Create payload
        payload = {
            'text': example['text'],
            'title': example['title'],
            '_id': example['_id'],
            'length': len(example['text']),
            'has_numbers': any(char.isdigit() for char in example['text'])
        }

        points.append(models.PointStruct(
            id=i,
            vector=embedding,
            payload=payload
        ))

    if points:
        client.upload_points(collection_name=collection, points=points)
        return len(points)
    return 0

# Upload all batches
# total_uploaded = 0
# for i in tqdm(range(0, total_points, batch_size), desc="Uploading points"):
#     uploaded = upload_batch_without_indexes(i, i + batch_size)
#     total_uploaded += uploaded
#
# print(f"\nUpload completed! Total points uploaded: {total_uploaded}")


url = "https://storage.googleapis.com/qdrant-examples/query_embedding_day_2.json"
resp = requests.get(url)
query_embedding = resp.json()["query_vector"]
# Warm up the RAM index/vectors cache with a test query
print("Warming up caches...")
client.query_points(collection_name=collection, query=query_embedding, limit=1)


# Measure vector search performance
search_times = []
for _ in range(3):  # Multiple runs for a stable average
    start_time = time.time()
    response = client.query_points(
        collection_name=collection,
        query=query_embedding,
        limit=10
    )
    search_time = (time.time() - start_time) * 1000
    search_times.append(search_time)

baseline_time = sum(search_times) / len(search_times)

print(f"Average search time: {baseline_time:.2f}ms")
print(f"Search times: {[f'{t:.2f}ms' for t in search_times]}")
print(f"Found {len(response.points)} results")
print(f"Top result: '{response.points[0].payload['title']}' (score: {response.points[0].score:.4f})")


print("Testing filtering without payload indexes")

# Create a text-based filter
text_filter = models.Filter(
    must=[
        models.FieldCondition(
            key="text",
            match=models.MatchText(text="metadata")
        )
    ]
)

# Run multiple times for more reliable measurement
unindexed_times = []
for i in range(3):
    start_time = time.time()
    response = client.query_points(
        collection_name=collection,
        query=query_embedding,
        limit=10,
        search_params=models.SearchParams(hnsw_ef=100),
        query_filter=text_filter
    )
    unindexed_times.append((time.time() - start_time) * 1000)

unindexed_filter_time = sum(unindexed_times) / len(unindexed_times)

print(f"Filtered search (WITHOUT index): {unindexed_filter_time:.2f}ms")
print(f"Individual times: {[f'{t:.2f}ms' for t in unindexed_times]}")
print(f"Overhead vs baseline: {unindexed_filter_time - baseline_time:.2f}ms")
print(f"Found {len(response.points)} matching results")
if response.points:
    print(f"Top result: '{response.points[0].payload['text']}'\nScore: {response.points[0].score:.4f}")
else:
    print("No results found - try a different filter term")


client.create_payload_index(
    collection_name=collection,
    field_name="text",
    wait=True,
    field_schema=models.TextIndexParams(
        type="text",
        tokenizer="word",
        phrase_matching=False
        )
    )

print("Payload index created for 'text' field")

print("Testing filtering WITH payload indexes...")

# Run multiple times for more reliable measurement
indexed_times = []
for i in range(3):
    start_time = time.time()
    response = client.query_points(
        collection_name=collection,
        query=query_embedding,
        limit=10,
        search_params=models.SearchParams(hnsw_ef=100),
        query_filter=text_filter
    )
    indexed_times.append((time.time() - start_time) * 1000)

indexed_filter_time = sum(indexed_times) / len(indexed_times)

print(f"Filtered search (WITH index): {indexed_filter_time:.2f}ms")
print(f"Individual times: {[f'{t:.2f}ms' for t in indexed_times]}")
print(f"Overhead vs baseline: {indexed_filter_time - baseline_time:.2f}ms")
print(f"Found {len(response.points)} matching results")
if response.points:
    print(f"Top result: '{response.points[0].payload['text']}'\nScore: {response.points[0].score:.4f}")
else:
    print("No results found - try a different filter term")

for point in response.points:
    print(f"{point.payload['text']}")


client.close()