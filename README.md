## Task 2 — Text Chunking, Embedding & Indexing

### Objective
This step converts cleaned consumer complaint narratives into vector embeddings for semantic search.

---

### What We Did
- **Chunking:** Split long complaint texts into smaller overlapping chunks (using LangChain’s `RecursiveCharacterTextSplitter`).
- **Embedding:** Used the `sentence-transformers/all-MiniLM-L6-v2` model — small, fast, and effective for short text.
- **Indexing:** Created a FAISS vector index for fast similarity search, with complaint ID and product metadata for traceability.

---

### Outputs
- **FAISS Index:** `vector_store/complaints_index.faiss`
- **Metadata:** `vector_store/chunks_metadata.csv`
- **Notebook:** `notebooks/02_chunking_embedding_indexing.ipynb`

---

### Notes
- Final `chunk_size`: *200 characters*
- Final `chunk_overlap`: *50 characters*
- This balance preserves context while keeping embeddings meaningful.
