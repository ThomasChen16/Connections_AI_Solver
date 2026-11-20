import os
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv() # Load environment variables from .env file

class Qwen3Embedding:
    # Load environment variables from .env file
    def __init__(self):
        self.client = OpenAI(
            # Load API key from environment variable
            api_key=os.getenv("NRP_API_KEY"),
            # Load base URL from environment variable
            base_url="https://ellm.nrp-nautilus.io/v1"
        )
        
    def encode(self, words, normalize_embeddings=True):
        # Request embeddings from NRP
        response = self.client.embeddings.create(
            model="qwen3-embedding",
            input=words
        )
        
        # Convert embeddings to a NumPy array
        embeddings = np.array(
            [item.embedding for item in sorted(
                response.data, key=lambda item: item.index
            )],
            dtype=np.float32
        )
        
        # Normalized each embedding vector
        if normalize_embeddings:
            norms = np.linalg.norm(
                embeddings, axis=1, keepdims=True
            )
            # Avoid division by zero by using a small epsilon value
            embeddings = embeddings / np.maximum(norms, 1e-12)
            
        return embeddings
    
# Load the Qwen3 embedding model
def load_model_qwen3():
    return Qwen3Embedding()
        