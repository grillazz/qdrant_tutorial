import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from rich import print

load_dotenv()

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

print(client.get_collections())

# Define the collection name
collection_name = "grillazz_music_3_collection"
#
# # Create the collection with specified vector parameters
client.create_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(
        size=4,  # Dimensionality of the vectors
        distance=models.Distance.COSINE  # Distance metric for similarity search
    )
)

collections = client.get_collections()
print("Existing collections:", collections)


# I use 4-dimensional vectors per song for genre-like scoring. pop , rock , jazz, electronic.
# The vector values are just for demonstration purposes and don't represent real embeddings.
songs_data = [
    {
        "band": "megadeth",
        "album": "megadeth",
        "song": "I Am War",
        "vector": [0.05, 0.90, 0.02, 0.03],
        "payload": {"category": "rock"},
    },
    {
        "band": "david gilmour",
        "album": "the wall",
        "song": "Comfortably Numb",
        "vector": [0.4, 0.4, 0.1, 0.1],
        "payload": {"category": "rock"},
    },
    {
        "band": "marillion",
        "album": "misplaced childhood",
        "song": "Kayleigh",
        "vector": [0.3, 0.4, 0.2, 0.1],
        "payload": {"category": "rock"},
    },
    {
        "band": "depeche mode",
        "album": "violator",
        "song": "Personal Jesus",
        "vector": [0.15, 0.35, 0.05, 0.85],
        "payload": {"category": "electronic"},
    },
    {
        "band": "norah jones",
        "album": "come away with me",
        "song": "the nearness of you",
        "vector": [0.3, 0.1, 0.7, 0.05],
        "payload": {"category": "jazz"},
    },
    {
        "band": "bts",
        "album": "be",
        "song": "dynamite",
        "vector": [0.9, 0.1, 0.05, 0.15],
        "payload": {"category": "pop"},
    },
]



# Define the vectors to be inserted
points = [
    models.PointStruct(
        id=idx,
        vector=item["vector"],
        payload={
            "band": item["band"],
            "album": item["album"],
            "song": item["song"],
            "category": item["payload"]["category"],
        },
    )
    for idx, item in enumerate(songs_data, start=1)
]

# Insert vectors into the collection
client.upsert(
    collection_name=collection_name,
    points=points,
)

collection_info = client.get_collection(collection_name)
print("Collection info:", collection_info)

# Create payload index required by strict-mode filtering (keyword match on `category`).
if "category" not in collection_info.payload_schema:
    client.create_payload_index(
        collection_name=collection_name,
        field_name="category",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )


query_vector = [0.08, 0.64, 0.33, 0.28]


search_filter = models.Filter(
    must=[
        models.FieldCondition(
            key="category",
            match=models.MatchValue(
                value="rock",
            )
        )
    ]
)


search_results = client.query_points(
    collection_name=collection_name,
    query=query_vector,
    query_filter=search_filter,
    limit=5  # Return the top 1 most similar vector
)

print("Search results:", search_results)