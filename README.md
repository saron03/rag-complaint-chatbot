# Intelligent Complaint Analysis for Financial Services

**Building a RAG-Powered Chatbot to Turn Customer Feedback into Actionable Insights**

---

## Project Overview

CrediTrust Financial is a fast-growing digital finance company serving East African markets with products like Credit Cards, Personal Loans, Buy Now, Pay Later (BNPL), Savings Accounts, and Money Transfers. This project builds an internal AI tool — a Retrieval-Augmented Generation (RAG) chatbot — to transform large volumes of unstructured customer complaints into actionable, evidence-backed insights.  

This enables product managers, support, and compliance teams to quickly understand emerging issues, reducing the time from days to minutes.

---

## Key Features

- **Data Exploration & Preprocessing**  
  Clean and filter millions of real consumer complaint narratives from the Consumer Financial Protection Bureau (CFPB).

- **Text Chunking & Embedding**  
  Split long complaint texts into manageable chunks, embed them using a state-of-the-art transformer model, and index them with a fast vector search engine (FAISS).

- **Semantic Search & Retrieval**  
  Retrieve relevant complaint excerpts based on natural language queries.

- **Generative Answering**  
  Use a Large Language Model (LLM) to generate concise, contextual answers grounded in real customer feedback.

- **Multi-Product Querying**  
  Support queries across five major financial products.

- **Interactive Chat Interface**  
  A user-friendly interface built with Gradio/Streamlit for non-technical users to interact with the chatbot, view answers, and verify sources.

---

## Project Structure
```
rag-complaint-chatbot
├── data/
│ └── filtered_complaints.csv # Cleaned and filtered complaint dataset
├── notebooks/
│ ├── 01_data_eda_preprocessing.ipynb
│ └── 02_chunking_embedding_indexing.ipynb
├── src/
│ ├── chunk_embed_index.py # Scripts for chunking, embedding, and indexing
│ ├── rag_pipeline.py # RAG retrieval and generation logic
│ └── app.py # Gradio/Streamlit chat interface
├── vector_store/ # Persisted FAISS index and metadata (gitignored)
├── reports/
│ ├── eda_summary.md
│ └── task2_chunking_embedding.md
├── requirements.txt
└── README.md
```

---

## Installation & Setup

1. Clone the repository:

   ```bash
   git clone git@github.com:saron03/rag-complaint-chatbot.git
   cd rag-complaint-chatbot
   ```
2. Create and activate a virtual environment (optional but recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    venv\Scripts\activate     # Windows
    ```

3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    Download the CFPB complaint dataset (link provided in project documentation) and place it in data/.
    ```

## Usage
- Step 1: Data Preprocessing & EDA

    * Run the notebook or script to clean and filter complaint narratives:

        ```bash
        jupyter notebook notebooks/01_data_eda_preprocessing.ipynb
        ```
    * This outputs `data/filtered_complaints.csv`.

- Step 2: Chunking, Embedding & Indexing
    * Run the chunking and embedding script or notebook:

        ```bash
        python src/chunk_embed_index.py
    * This creates a FAISS index and metadata files saved in vector_store/.

- Step 3: Running the RAG Pipeline
    * Use the retrieval and generation module in `src/rag_pipeline.py` to query complaints and generate answers.

- Step 4: Launching the Chat Interface
    * Run the chat app with:

        ```bash
        python src/app.py
        ```
    * This opens a web interface where users can ask questions and get answers backed by real complaint data.

## Model Choices

- **Embedding Model** : `sentence-transformers/all-MiniLM-L6-v2`

    Selected for its balance of speed, lightweight footprint, and strong semantic search performance on short texts.

- **Vector Store** : `FAISS`

    Enables fast similarity search over thousands of complaint chunks.

- **LLM for Generation**: Configurable (e.g., Hugging Face Transformers, LangChain integrations)

---

## Contribution & Collaboration
Contributions and improvements are welcome via pull requests.

## License
MIT License

