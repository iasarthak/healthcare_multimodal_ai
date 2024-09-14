# Convert text to embeddings using Fastembed
import clip
import torch
from PIL import Image
from fastembed import TextEmbedding

TEXT_MODEL_NAME = "Qdrant/clip-ViT-B-32-text"


def convert_text_to_embeddings(text_document):
    embedding_model = TEXT_MODEL_NAME
    text_embedding_model = TextEmbedding(model_name=embedding_model)
    text_embeddings = text_embedding_model.embed(text_document)  # Returns a generator of embeddings
    return text_embeddings


def convert_image_to_embeddings(image_path: str):
    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Load the image
    image = Image.open(image_path)
    image = preprocess(image).unsqueeze(0).to(device)

    return image


if __name__ == "__main__":
    # text_docs = ["Hello, world!", "This is a test sentence."]
    # embeddings = convert_text_to_embeddings(text_docs)
    # for embedding in embeddings:
    #     print(embedding)

    image_path = "/Users/sarthak/Documents/Work/Personal_Projects/healthcare_multimodal_ai/data/images/ROCO_00796.jpg"
    image_embedding = convert_image_to_embeddings(image_path)

    descriptions = [
        "A Pictrue of a Tiger and Girl on Rocks",
        "A picture of Donkey and a Man",
        "A picture of a red car",
        "A picture of a Sparrow and Butterfly",
        "A picture of Animal and Human",
    ]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    text = clip.tokenize(descriptions).to(device)
    with torch.no_grad():
        logits_per_image, logits_per_text = model(image_embedding, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    results = dict(zip(descriptions, map(lambda x: x * 100, probs[0])))
    results = {k: v for k, v in sorted(results.items(), key=lambda item: item[1], reverse=True)}  # Sorted Results
    for text, percentage in results.items():
        print(f"Description: {text}, Similarity: {percentage}")
