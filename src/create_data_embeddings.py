import os
import uuid

import pandas as pd
from fastembed import TextEmbedding, ImageEmbedding
from qdrant_client import QdrantClient, models

from src.embeddings_utils import convert_text_to_embeddings, convert_image_to_embeddings, TEXT_MODEL_NAME, \
    IMAGE_MODEL_NAME

DATA_PATH = '/Users/sarthak/Documents/Work/Personal_Projects/healthcare_multimodal_ai/data/'


def create_uuid_from_image_id(image_id):
    NAMESPACE_UUID = uuid.UUID('12345678-1234-5678-1234-567812345678')
    return str(uuid.uuid5(NAMESPACE_UUID, image_id))


def create_embeddings(collection_name):
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

    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "image": models.VectorParams(size=image_embeddings_size, distance=models.Distance.COSINE),
                "text": models.VectorParams(size=text_embeddings_size, distance=models.Distance.COSINE),
            }
        )
    client.upload_points(
        collection_name=collection_name,
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
