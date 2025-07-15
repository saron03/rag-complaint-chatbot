# src/recreate_vector_store.py
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import time
import os
import torch

# Load metadata
start_time = time.time()
print("Loading chunks_metadata.csv...")
try:
    metadata = pd.read_csv("vector_store/chunks_metadata.csv")
except FileNotFoundError:
    raise FileNotFoundError("chunks_metadata.csv not found in ../vector_store/")

# Print columns for verification
print("Columns in chunks_metadata.csv:", list(metadata.columns))

# Use a subset for testing
subset_size = 1000  # Keep for speed
metadata = metadata.head(subset_size)
print(f"Using subset of {subset_size} rows for faster processing")

# Initialize embedding model
print("Initializing embedding model...")
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
)

# Prepare documents and metadata
try:
    documents = metadata["text"].tolist()
    metadatas = metadata[["complaint_id", "product"]].to_dict("records")
except KeyError as e:
    raise KeyError(f"Column not found in chunks_metadata.csv: {str(e)}. Available columns: {list(metadata.columns)}")

# Create FAISS index
print("Creating FAISS index...")
vector_store = FAISS.from_texts(documents, embedding_model, metadatas=metadatas)

# Save FAISS index with absolute path
print("Saving FAISS index...")
save_path = "C:/Users/saron/OneDrive/Desktop/kifya/week6/rag-complaint-chatbot/vector_store"
index_name = "complaints_index"
try:
    vector_store.save_local(save_path, index_name=index_name)
    faiss_path = os.path.join(save_path, f"{index_name}.faiss")
    pkl_path = os.path.join(save_path, f"{index_name}.pkl")
    if os.path.exists(faiss_path) and os.path.exists(pkl_path):
        print(f"FAISS vector store saved to {faiss_path} and {pkl_path}")
    else:
        print(f"Error: One or both files not found after saving: {faiss_path}, {pkl_path}")
except Exception as e:
    print(f"Error saving FAISS index: {str(e)}")

print(f"Time taken: {time.time() - start_time:.2f} seconds")