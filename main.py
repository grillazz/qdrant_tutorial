from qdrant_client import QdrantClient

qdrant_client = QdrantClient(
    url="https://5b8c95d9-b91c-4923-acbe-47469e005ffe.europe-west3-0.gcp.cloud.qdrant.io:6333", 
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6OWRhNjI4ZGUtNzZlNS00Mjg2LThlODEtMWViZWNmMTQxNWYwIn0.S3OJGtsrkjGsdJLd9y5MMKYz6S5-zSSGOjev-Fl3YvU",
)

print(qdrant_client.get_collections())

