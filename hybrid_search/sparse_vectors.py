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

collection_name = "sparse_vectors_collection_1"

# Create the collection with sparse vectors
client.create_collection(
    collection_name=collection_name,
    sparse_vectors_config={ #vector named "sparse_vector"
        "sparse_vector": models.SparseVectorParams(),
    },
)


collection_name = "sparse_vectors_collection_custom_index"

client.create_collection(
    collection_name=collection_name,
    sparse_vectors_config={
        "sparse_vector": models.SparseVectorParams(
            index=models.SparseIndexParams( #inverted index parameters
                full_scan_threshold=0, #full scan search, not using inverted index
                on_disk=False, #where inverted index is stored
                datatype=models.VectorStorageDatatype("float32") #precision of values stored in inverted index

            )
        ),
    },
)

# Insert vectors into the collection
client.upsert(
    collection_name=collection_name,
    points=[
        models.PointStruct(
            id=1,
            payload={},
            vector={ #vector named "sparse_vector"
                "sparse_vector": models.SparseVector(
                    indices=[1, 2, 3], #uint32, from 0 to 4_294_967_295
                    values=[0.2, -0.2, 0.2] #stored as floats
                )
            },
        ),
        models.PointStruct(
            id=2,
            payload={},
            vector={ #vector named "sparse_vector"
                "sparse_vector": models.SparseVector(
                    indices=[1, 5], #uint32, from 0 to 4_294_967_295
                    values=[0.1, 0.1] #stored as floats
                )
            },
        ),
    ],
)

_query_result = client.query_points(
    collection_name=collection_name,
    using="sparse_vector",  # we need to specify the name of our sparse vectors to search against them
    limit=1,                # return the top 1 most similar result
    query=models.SparseVector(
        indices=[1, 3],
        values=[1, 1]
    ),
    with_vectors=True # to see the top 1 most similar vector
)

print(_query_result)

# Let’s understand why we got Point 1 as the answer.
#
# In the collection, we have two points:
#
# Point 1 has three non-zero values: values = [0.2, -0.2, 0.2] with indices = [1, 2, 3]
# Point 2 has two non-zero values: values = [0.1, 0.1] with indices = [1, 5]
# Our query has indices = [1, 3] with corresponding values = [1, 1].
#
# The similarity score for sparse vectors is calculated by comparing only the matching indices shared between the query and the points: [1, 3] for Point 1, and [1] for Point 2.
#
# We multiply the corresponding values and sum them up:
#
# score(query, Point 1) = 1 * 0.2 + 1 * 0.2 = 0.4
# score(query, Point 2) = 1 * 0.1 = 0.1
# Since 0.4 is higher than 0.1, Point 1 is more similar to our query.