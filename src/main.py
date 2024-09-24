import gradio as gr
from PIL import Image
from multimodal_rag_system import MultimodalRAGSystem

# Initialize the MultimodalRAGSystem
system = MultimodalRAGSystem()


# Define the Gradio function that will process the user input and image
def chatbot_interface(user_query, user_image=None):
    if user_image is not None:
        # Convert numpy array to a PIL Image and save it temporarily
        user_image = Image.fromarray(user_image)
        user_image_path = ("/Users/sarthak/Documents/Work/Personal_Projects/healthcare_multimodal_ai/data"
                           "/user_input_image.jpg")
        user_image.save(user_image_path)
    else:
        user_image_path = None

    # Get the response from the Multimodal AI system
    response = system.process_query(user_query, query_image_path=user_image_path)
    return response


# Create the Gradio interface with text input and image input
interface = gr.Interface(
    fn=chatbot_interface,
    inputs=[gr.components.Textbox(lines=5, label="User Query"), gr.components.Image(label="Upload Medical Image")],
    outputs=gr.components.Textbox(),
    title="Multimodal Medical Assistant",
    description="Ask medical-related questions and upload relevant medical images for analysis."
)

# Launch the Gradio interface
if __name__ == "__main__":
    interface.launch()
