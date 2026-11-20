from sentence_transformers import SentenceTransformer # Model for generating sentence embeddings

# Load the MiniLM model
def load_model_minilm():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")