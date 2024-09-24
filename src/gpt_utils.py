import base64
import json

import requests

from config import Config


class GPTClient:
    def __init__(self):
        self.api_key = Config.OPENAI_API_KEY
        self.api_url = "https://api.openai.com/v1/chat/completions"

    def query(self, prompt, retrieved_contexts, user_image=None):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        chatbot_role = """
        You are a radiologist with an experience of 30 years. 
        You analyse medical scans and text, and help diagnose underlying issues.
        """

        # Initialize message structure with the system prompt
        messages = [
            {"role": "system", "content": chatbot_role},
            {"role": "user", "content": [
                {"type": "text", "text": prompt}
            ]}
        ]

        # Add the user-uploaded image (if any)
        if user_image:
            with open(user_image, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                messages[1]["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                })

        messages[1]["content"].append({
            "type": "text",
            "text": "Additional context that you may use as a reference. Use them if you feel they are relevant to "
                    "the case."
                    "NOTE: They are not the patient's images. They are of other patients which can be used as a "
                    "reference, if required."
        })

        # Add the retrieved contexts, which should include both captions and corresponding images
        for context in retrieved_contexts:
            caption = context.payload['caption']
            image_path = context.payload['image_path']

            # Add the caption to the message
            messages[1]["content"].append({
                "type": "text",
                "text": f"Caption: {caption}"
            })

            # Add the corresponding image to the message
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                messages[1]["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                })

        # Prepare the payload for the GPT API
        data = {
            "model": "gpt-4o",
            "messages": messages,
            "max_tokens": 600
        }

        # Send the request to the API
        response = requests.post(self.api_url, headers=headers, data=json.dumps(data))
        return response.json()

    def process_response(self, response):
        if 'choices' in response and len(response['choices']) > 0:
            return response['choices'][0]['message']['content']
