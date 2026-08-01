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

# client.create_collection(
#     collection_name="bm25_vectors_collection",
#     sparse_vectors_config={
#         "bm25_sparse_vector": models.SparseVectorParams(
#             modifier=models.Modifier.IDF #Inverse Document Frequency
#         ),
#     },
# )


grocery_items_descriptions = [
    "Grated hard cheese",
    "White crusty bread roll",
    "Mac and cheese"
]

#Estimating the average length of documents in the corpus
avg_document_length = sum(len(description.split()) for description in grocery_items_descriptions) / len(grocery_items_descriptions)

print(f"Average document length: {avg_document_length}")

client.upsert(
    collection_name="bm25_vectors_collection",
    points=[
        models.PointStruct(
            id=i,
            payload={"text": description}, #meta data, descriptions text in human-readable format
            vector={
                "bm25_sparse_vector": models.Document( #to run FastEmbed under the hood
                    text=description,
                    model="Qdrant/bm25",
                    options={"avg_len": avg_document_length} #To pass BM25 parameters, here we're using default k & b for the BM25 formula
                )
           },
        ) for i, description in enumerate(grocery_items_descriptions)
    ],
)


_query_response = client.query_points(
    collection_name="bm25_vectors_collection",
    using="bm25_sparse_vector",
    limit=3,
    query=models.Document(  #to run FastEmbed under the hood
        text="cheese",
        model="Qdrant/bm25"
    ),
    with_vectors=True,
)
print(_query_response)