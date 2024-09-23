import json
from fastembed import TextEmbedding
from qdrant_client import QdrantClient

from src.create_data_embeddings import search_similar_text, create_embeddings, search_similar_image, merge_results
from src.embeddings_utils import convert_text_to_embeddings, TEXT_MODEL_NAME
from src.gpt_utils import GPTClient
from config import Config

COLLECTION_NAME = "medical_images_text"


class MultimodalRAGSystem:
    def __init__(self):
        self.gpt_client = GPTClient()
        self.qdrant_client = create_embeddings()

    def process_query(self, query, query_image_path=None, top_k=3):
        # 1. Text-based search if query is textual
        search_results_text = search_similar_text(self.qdrant_client, query, limit=top_k)

        # 2. Image-based search if a query image is provided
        search_results_image = []
        if query_image_path:  # Only perform image retrieval if an image path is provided
            search_results_image = search_similar_image(self.qdrant_client, query_image_path, limit=top_k)

        # 3. Combine both results - merging text and image results
        combined_results = merge_results(search_results_text, search_results_image)

        # 4. Prepare context and image paths for GPT
        retrieved_context = "\n".join([f"Caption: {result.payload['caption']}" for result in combined_results])
        image_paths = [result.payload['image_path'] for result in combined_results if 'image_path' in result.payload]

        # 5. Query GPT with the context and images
        gpt_response = self.gpt_client.query(query, combined_results, query_image_path)

        # 6. Process and return the response
        return self.gpt_client.process_response(gpt_response)


# Example usage
if __name__ == "__main__":
    system = MultimodalRAGSystem()
    user_query = ("The patient has been experiencing acute neck pain for several weeks. I’m attaching a relevant scan "
                  "of the affected area. "
                  "Additionally, I am attaching several relevant images and corresponding analyses, which could help "
                  "you in diagnosis. They are not of the same patient but can be used as context."
                  "The first image is of the patient and the rest are extracted context."
                  "Can you analyze the scan and provide insights based on the patient’s "
                  "symptoms and the attached scan?"
                  "Just give a concise analysis about the patient and not other context attached.")
    user_image_path = "/Users/sarthak/Documents/Work/Personal_Projects/healthcare_multimodal_ai/data/ROCO_80642_neck_test.jpg"  # Example path to an image file

    response = system.process_query(user_query, query_image_path=user_image_path)
    print(f"User Query: {user_query}")
    print(f"System Response: {response}")
