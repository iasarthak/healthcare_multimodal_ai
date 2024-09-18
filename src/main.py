import json
from fastembed import TextEmbedding
from qdrant_client import QdrantClient

from src.create_data_embeddings import search_similar_text, create_embeddings
from src.embeddings_utils import convert_text_to_embeddings, TEXT_MODEL_NAME
from src.gpt_utils import GPTClient
from config import Config

COLLECTION_NAME = "medical_images_text"


class MultimodalRAGSystem:
    def __init__(self):
        self.gpt_client = GPTClient()
        self.qdrant_client = create_embeddings()

    def process_query(self, query, top_k=3):
        # 1. Generate embedding for the query
        search_results = search_similar_text(self.qdrant_client, query, limit=top_k)

        # 2. Prepare context and image paths for GPT
        retrieved_context = "\n".join([f"Caption: {result.payload['caption']}" for result in search_results])
        image_paths = [result.payload['image_path'] for result in search_results]

        # 3. Query GPT with the context and images
        gpt_response = self.gpt_client.query(query, retrieved_context, image_paths)

        # 4. Process and return the response
        return self.gpt_client.process_response(gpt_response)


# Example usage
if __name__ == "__main__":
    system = MultimodalRAGSystem()
    user_query = "I'm having pain in my lower back. What could be the issue based on these images?"
    response = system.process_query(user_query)
    print(f"User Query: {user_query}")
    print(f"System Response: {response}")
