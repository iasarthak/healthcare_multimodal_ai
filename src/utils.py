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


if __name__ == "__main__":
    image_path = ["/Users/sarthak/Documents/Work/Personal_Projects/healthcare_multimodal_ai/data/images/ROCO_81399.jpg"]
    image_embedding = convert_image_to_embeddings(image_path)
