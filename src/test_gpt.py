#%%
import os
import uuid
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#%%
from src.gpt_utils import GPTClient

# Instance of GPTClient
gpt_client = GPTClient()

# Test Query
test_prompt = "What are the symptoms of a herniated disk in the neck?"
response = gpt_client.query(test_prompt, retrieved_contexts=[])

# Check the processed response
if response:
    output = gpt_client.process_response(response)
    print("\n Final GPT Response:", output)
else:
    print("No response received from API")

# %%