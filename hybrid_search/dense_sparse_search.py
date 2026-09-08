
from qdrant_client import QdrantClient, models
import os
import uuid

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=300,
)

# Define the collection name
collection_name = "hybrid_search_demo_4"

# Create our collection with both sparse (bm25) and dense vectors
client.create_collection(
    collection_name=collection_name,
    vectors_config={
        "dense": models.VectorParams(
            distance=models.Distance.COSINE,
            size=384,
        ),
    },
    sparse_vectors_config={
        "sparse": models.SparseVectorParams(
            modifier=models.Modifier.IDF
        )
    }
)

#
# documents = [
#     "Aged Gouda develops a crystalline texture and nutty flavor profile after 18 months of maturation.",
#     "Mature Gouda cheese becomes grainy and develops a rich, buttery taste with extended aging.",
#     "Brie cheese features a soft, creamy interior surrounded by an edible white rind.",
#     "This French cheese has a flowing, buttery center encased in a bloomy white crust.",
#     "Fresh mozzarella pairs beautifully with ripe tomatoes and basil leaves.",
#     "Classic Margherita pizza topped with tomato sauce, mozzarella, and fresh basil.",
#     "Parmesan requires at least 12 months of cave aging to develop its signature sharp taste.",
#     "Parmigiano-Reggiano's distinctive piquant flavor comes from extended maturation in controlled environments.",
#     "Grilled cheese sandwiches are the ultimate American comfort food for cold winter days.",
#     "Croque Monsieur combines ham and Gruyère in France's answer to the toasted cheese sandwich.",
# ]

documents = [
    "Dojrzała gouda po 18 miesiącach dojrzewania uzyskuje krystaliczną teksturę i orzechowy profil smakowy.",
    "Dojrzały ser gouda staje się ziarnisty i zyskuje bogaty, maślany smak w wyniku długiego leżakowania.",
    "Ser brie ma miękkie, kremowe wnętrze otoczone jadalną, białą skórką.",
    "Ten francuski ser ma płynne, maślane wnętrze zamknięte w białej, kwitnącej skórce.",
    "Świeża mozzarella świetnie komponuje się z dojrzałymi pomidorami i liśćmi bazylii.",
    "Klasyczna pizza margherita z sosem pomidorowym, mozzarellą i świeżą bazylią.",
    "Parmezan wymaga co najmniej 12 miesięcy dojrzewania w piwnicy, aby rozwinąć swój charakterystyczny ostry smak.",
    "Wyrazisty, pikantny smak Parmigiano-Reggiano pochodzi z długiego dojrzewania w kontrolowanych warunkach.",
    "Kanapki z grillowanym serem to amerykańskie danie comfort food idealne na zimowe dni.",
    "Croque Monsieur łączy szynkę i gruyère – to francuska odpowiedź na zapiekaną kanapkę z serem.",
]


client.upsert(
    collection_name=collection_name,
    points=[
        models.PointStruct(
            id=uuid.uuid4().hex,
            vector={
                "dense": models.Document(
                    text=doc,
                    model="sentence-transformers/all-MiniLM-L6-v2",
                ),
                "sparse": models.Document(
                    text=doc,
                    model="sdadas/polish-splade",
                ),
            },
            payload={"text": doc},
        )
        for i, doc in enumerate(documents)
    ]
)