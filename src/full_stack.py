#%%
import openai

def configure_openai_api(api_key):
    openai.api_key = api_key

def generate_text_from_embeddings(embeddings):
    response = openai.Completion.create(
        model="text-davinci-003", #choose model
        prompt="Generate text from embeddings: " + str(embeddings),
        max_tokens=100
    )
    return response.choices[0].text



#%%
def main():
    # Part 1: Create Embeddings using llama_index
    image_docs, text_embeddings, image_embeddings = create_embeddings()

    # Part 2: Save Embeddings to Qdrant
    save_embeddings_to_db(image_docs, text_embeddings, image_embeddings)

    # Part 3: Configure OpenAI API and Generate Content
    configure_openai_api("your_openai_api_key")
    generated_text = generate_text_from_embeddings(text_embeddings)
    print(generated_text)

if __name__ == "__main__":
    main()

