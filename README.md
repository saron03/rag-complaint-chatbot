## Task 3 — RAG Core Logic & Evaluation

This module implements the core Retrieval-Augmented Generation (RAG) system for answering customer complaint questions.

---

## What’s Included

- `rag_pipeline.py` — Contains the `RAGPipeline` class:  
  - Embeds user questions  
  - Retrieves top-k relevant complaint chunks (FAISS)  
  - Uses a prompt template  
  - Generates answers with a local LLM (`distilgpt2`)

- `preprocess_data.py` — Preprocesses complaint data before vectorization.

- `requirements.txt` — Dependencies for LangChain, Transformers, FAISS, etc.

---