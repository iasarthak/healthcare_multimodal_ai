from qdrant_client import QdrantClient, models
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.storage.storage_context import StorageContext

def save_embeddings_to_db(image_docs, text_embeddings, image_embeddings):
    # Initialize Qdrant client
    client = QdrantClient(
        host="localhost",  # IP address of the Docker container
        port=6333  # Qdrant port exposed in the container
    )

    # Check if collection exists, if not, create it
    if not client.collection_exists("text_imageai"):
        client.create_collection(
            collection_name="text_imageai",
            vectors_config={
                "image": models.VectorParams(size=512, distance=models.Distance.COSINE),
                "text": models.VectorParams(size=len(text_embeddings[0]), distance=models.Distance.COSINE),
            }
        )

    # Prepare `llama_index` storage context for managing embeddings
    qdrant_store = QdrantVectorStore(client, collection_name="text_imageai")
    storage_context = StorageContext.from_defaults(vector_store=qdrant_store)

    # Upload text and image embeddings to Qdrant with payload
    for idx in range(len(image_docs)):
        storage_context.vector_store.add_vectors(
            vectors={"text": text_embeddings[idx], "image": image_embeddings[idx]},
            payload=image_docs[idx]
        )
