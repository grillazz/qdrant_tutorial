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

# ds = load_dataset("Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-100K")

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

client.close()