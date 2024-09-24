from typing import List
from fastembed import TextEmbedding, ImageEmbedding

TEXT_MODEL_NAME = "Qdrant/clip-ViT-B-32-text"
IMAGE_MODEL_NAME = "Qdrant/clip-ViT-B-32-vision"


def convert_text_to_embeddings(documents: List[str], embedding_model: str = TEXT_MODEL_NAME) -> List:
    text_embedding_model = TextEmbedding(model_name=embedding_model)
    text_embeddings = list(text_embedding_model.embed(documents))  # Returns a generator of embeddings
    return text_embeddings


def convert_image_to_embeddings(images: List[str], embedding_model: str = IMAGE_MODEL_NAME) -> List:
    image_model = ImageEmbedding(model_name=embedding_model)
    images_embedded = list(image_model.embed(images))
    return images_embedded


def search_similar_text(collection_name, client, query, limit=3):
    text_model = TextEmbedding(model_name=TEXT_MODEL_NAME)
    search_query = text_model.embed([query])
    search_results = client.search(
        collection_name=collection_name,
        query_vector=('text', list(search_query)[0]),
        with_payload=['image_path', 'caption'],
        limit=limit,
    )
    return search_results


def search_similar_image(collection_name, client, query_image_path, limit=3):
    # Convert the query image into an embedding using the same model used for image embeddings
    image_embedding_model = ImageEmbedding(model_name=IMAGE_MODEL_NAME)

    # Embed the provided query image (assumed to be a file path)
    query_image_embedding = list(image_embedding_model.embed([query_image_path]))[0]  # Embedding for the query image

    # Perform the similarity search in the Qdrant collection for image embeddings
    search_results = client.search(
        collection_name=collection_name,
        query_vector=('image', query_image_embedding),
        with_payload=['image_path', 'caption'],  # Fetch image paths and captions as metadata
        limit=limit,
    )
    return search_results


def merge_results(text_results, image_results):
    # Combine based on some metadata, or simply concatenate
    combined_results = text_results + image_results
    return combined_results



if __name__ == "__main__":
    image_path = ["/Users/sarthak/Documents/Work/Personal_Projects/healthcare_multimodal_ai/data/images/ROCO_81399.jpg"]
    image_embedding = convert_image_to_embeddings(image_path)
