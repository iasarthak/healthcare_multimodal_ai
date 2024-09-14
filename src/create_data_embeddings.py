import os

import pandas as pd
from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models

from src.utils import convert_text_to_embeddings, convert_image_to_embeddings

data_path = '/Users/sarthak/Documents/Work/Personal_Projects/healthcare_multimodal_ai/data/'
COLLECTION_NAME = "medical_images_text"
TEXT_MODEL_NAME = "Qdrant/clip-ViT-B-32-text"


def create_embeddings():
    # Read captions txt data
    path = data_path + 'captions.txt'
    caption_df = pd.read_csv(path, sep='\t', header=None, names=['image_id', 'caption'])

    # Read images
    image_directory = os.listdir(data_path + 'images')

    # Filter out images that are not in the captions
    images = []
    for image in image_directory:
        if image.split('.')[0] in caption_df['image_id'].values:
            images.append(image)

    # Create image_id, caption, image_path list of dictionaries
    image_docs = []
    for image in images:
        image_id = image.split('.')[0]
        caption = caption_df[caption_df['image_id'] == image_id]['caption'].values[0]
        image_path = data_path + 'images/' + image
        image_docs.append({'image_id': image_id, 'caption': caption, 'image_path': image_path})

    # Convert text to embeddings using Fastembed and images to embeddings using CLIP
    embeddings = convert_text_to_embeddings([doc['caption'] for doc in image_docs])
    for idx, embedding in enumerate(embeddings):
        image_docs[idx]['caption_embedding'] = embedding

    # Convert image to embeddings using CLIP
    image_embeddings = []
    for doc in image_docs:
        image_embeddings.append(convert_image_to_embeddings(doc['image_path']))

    for idx, embedding in enumerate(image_embeddings):
        image_docs[idx]['image_embedding'] = embedding

    # Save the embeddings to vector database
    client = QdrantClient("http://localhost:6333")

    text_model = TextEmbedding(model_name=TEXT_MODEL_NAME)
    text_embeddings_size = text_model._get_model_description(TEXT_MODEL_NAME)["dim"]

    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "image": models.VectorParams(size=512, distance=models.Distance.COSINE),
                "text": models.VectorParams(size=text_embeddings_size, distance=models.Distance.COSINE),
            }
        )
    client.upload_points(
        collection_name=COLLECTION_NAME,
        points=[
            models.PointStruct(
                id=doc['image_id'],
                vector={
                    "text": doc['caption_embedding'],
                    "image": doc['image_embedding'],
                },
                payload=doc  # original image and its caption
            )
            for doc in image_docs
        ]
    )


if __name__ == "__main__":
    create_embeddings()
