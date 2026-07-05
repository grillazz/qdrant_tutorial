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

collection = "dbpedia_100K_1"
if not client.collection_exists(collection_name=collection):
    client.create_collection(
        collection_name=collection,
        vectors_config=models.VectorParams(
            size=1536,
            distance=models.Distance.COSINE
        ),
        hnsw_config=models.HnswConfigDiff(
            m=0,
            ef_construct=100,
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
total_uploaded = 0
for i in tqdm(range(0, total_points, batch_size), desc="Uploading points"):
    uploaded = upload_batch_without_indexes(i, i + batch_size)
    total_uploaded += uploaded

print(f"\nUpload completed! Total points uploaded: {total_uploaded}")


client.close()