#%%
import gradio as gr
from PIL import Image
import os



# Function to search and display images based on a query
def search_images(query):
    # Perform the search query
    retrieval_results = retriever.retrieve(query)
    response = query_engine.query(query)

    # Prepare the display output
    image_paths = []
    captions = []
    for result in retrieval_results:
        # Robustly handle image metadata retrieval
        image_path = response.metadata.get(result, {}).get('image', None)
        caption = response.response
        if image_path and os.path.exists(image_path):
            image_paths.append(image_path)
        captions.append(caption)

    return image_paths, captions

# Gradio interface function
def gradio_interface(query):
    image_paths, captions = search_images(query)

    # Efficient image loading
    pil_images = [Image.open(img_path) for img_path in image_paths if os.path.exists(img_path)]

    return pil_images, captions

# Create the Gradio app
iface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Textbox(label="Query"),
    outputs=[gr.Gallery(label="Retrieved Images"), gr.Textbox(label="Captions")],
    title="Enhanced Healthcare Diagnostics Search App",
    description="Enter a query to search for images and their associated captions."
)

# Launch the Gradio app
iface.launch(debug=True)

#%%
## search_images(query):
# -Takes a search query as input.
# -Uses a retrieval engine (retriever.retrieve()) to find relevant images.
# -Uses a query engine (query_engine.query()) to retrieve captions.
# -Collects the image paths and captions, ensuring each image path exists. -Returns lists of image paths and captions.
## gradio_interface(query):
# -Calls search_images() to get the images and captions based on the user’s input query. -Converts the images to a format that can be displayed (PIL format). -Returns the images and captions, which Gradio displays.
## Gradio Interface:
# -Defines the inputs (textbox for query) and outputs (gallery for images and textbox for captions).
# -Launches the Gradio app with a public-facing interface where users can type in a query and view the results.