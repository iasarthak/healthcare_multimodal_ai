import os
import uuid

import pandas as pd
from fastembed import TextEmbedding, ImageEmbedding
from qdrant_client import QdrantClient, models

from src.embeddings_utils import convert_text_to_embeddings, convert_image_to_embeddings, TEXT_MODEL_NAME, \
    IMAGE_MODEL_NAME

DATA_PATH = '/Users/sarthak/Documents/Work/Personal_Projects/healthcare_multimodal_ai/data/'
COLLECTION_NAME = "medical_images_text"


def create_uuid_from_image_id(image_id):
    NAMESPACE_UUID = uuid.UUID('12345678-1234-5678-1234-567812345678')
    return str(uuid.uuid5(NAMESPACE_UUID, image_id))


def search_similar_text(client, query, limit=3):
    text_model = TextEmbedding(model_name=TEXT_MODEL_NAME)
    search_query = text_model.embed([query])
    search_results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=('text', list(search_query)[0]),
        with_payload=['image_path', 'caption'],
        limit=limit,
    )
    return search_results


def search_similar_image(client, query_image_path, limit=3):
    # Convert the query image into an embedding using the same model used for image embeddings
    image_embedding_model = ImageEmbedding(model_name=IMAGE_MODEL_NAME)

    # Embed the provided query image (assumed to be a file path)
    query_image_embedding = list(image_embedding_model.embed([query_image_path]))[0]  # Embedding for the query image

    # Perform the similarity search in the Qdrant collection for image embeddings
    search_results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=('image', query_image_embedding),
        with_payload=['image_path', 'caption'],  # Fetch image paths and captions as metadata
        limit=limit,
    )
    return search_results


def merge_results(text_results, image_results):
    # Combine based on some metadata, or simply concatenate
    combined_results = text_results + image_results
    return combined_results


def create_embeddings():
    # Read captions txt data
    path = DATA_PATH + 'captions.txt'
    caption_df = pd.read_csv(path, sep='\t', header=None, names=['image_id', 'caption'])

    # Read images
    image_directory = os.listdir(DATA_PATH + 'images')

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
        image_path = DATA_PATH + 'images/' + image
        image_docs.append({'image_id': image_id, 'caption': caption, 'image_path': image_path})

    # Convert text to embeddings using Fastembed and images to embeddings using CLIP
    captions = [doc['caption'] for doc in image_docs]
    embeddings = convert_text_to_embeddings(captions)
    for idx, embedding in enumerate(embeddings):
        image_docs[idx]['caption_embedding'] = embedding

    # Convert image to embeddings using CLIP
    image_embeddings = convert_image_to_embeddings([doc['image_path'] for doc in image_docs])

    for idx, embedding in enumerate(image_embeddings):
        image_docs[idx]['image_embedding'] = embedding

    # Save the embeddings to vector database
    client = QdrantClient(":memory:")

    text_model = TextEmbedding(model_name=TEXT_MODEL_NAME)
    text_embeddings_size = text_model._get_model_description(TEXT_MODEL_NAME)["dim"]

    image_model = ImageEmbedding(model_name=IMAGE_MODEL_NAME)
    image_embeddings_size = image_model._get_model_description(IMAGE_MODEL_NAME)["dim"]

    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "image": models.VectorParams(size=image_embeddings_size, distance=models.Distance.COSINE),
                "text": models.VectorParams(size=text_embeddings_size, distance=models.Distance.COSINE),
            }
        )
    client.upload_points(
        collection_name=COLLECTION_NAME,
        points=[
            models.PointStruct(
                # Convert image_id to UUID
                id=create_uuid_from_image_id(doc['image_id']),
                vector={
                    "text": doc['caption_embedding'],
                    "image": doc['image_embedding'],
                },
                payload={
                    "image_id": doc['image_id'],
                    "caption": doc['caption'],
                    "image_path": doc['image_path']
                }
            )
            for doc in image_docs
        ]
    )
    return client


if __name__ == "__main__":
    create_embeddings()
