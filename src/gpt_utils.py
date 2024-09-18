import base64
import json

import requests

from config import Config


class GPTClient:
    def __init__(self):
        self.api_key = Config.OPENAI_API_KEY
        self.api_url = "https://api.openai.com/v1/chat/completions"

    def query(self, prompt, retrieved_text, images=None):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        messages = [
            {"role": "system", "content": "You are a helpful assistant analyzing medical reports and images."},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "text", "text": f"Retrieved context: {retrieved_text}"}
            ]}
        ]

        if images:
            for image_path in images:
                with open(image_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                    messages[1]["content"].append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    })

        data = {
            "model": "gpt-4o",
            "messages": messages,
            "max_tokens": 300
        }

        response = requests.post(self.api_url, headers=headers, data=json.dumps(data))
        return response.json()

    def process_response(self, response):
        if 'choices' in response and len(response['choices']) > 0:
            return response['choices'][0]['message']['content']
