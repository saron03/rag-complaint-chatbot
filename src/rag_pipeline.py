import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from transformers import pipeline
import torch
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self, vector_store_path, index_name="complaints_index.faiss", embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"):
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model_name,
                model_kwargs={"local_files_only": True}
            )
            logger.info(f"Initialized embedding model: {embedding_model_name}")
        except Exception as e:
            logger.error(f"Error initializing embedding model: {str(e)}")
            raise

        try:
            self.vector_store = FAISS.load_local(
                folder_path=vector_store_path,
                embeddings=self.embeddings,
                index_name=index_name.replace(".faiss", ""),
                allow_dangerous_deserialization=True
            )
            logger.info(f"Loaded FAISS vector store from {vector_store_path}")
        except Exception as e:
            logger.error(f"Error loading FAISS index {index_name}: {str(e)}")
            raise FileNotFoundError(f"Error loading FAISS index {index_name} from {vector_store_path}: {str(e)}")

        try:
            self.llm = pipeline(
                "text-generation",
                model="facebook/opt-350m",  # Non-gated, better than distilgpt2
                # model="mistralai/Mixtral-8x7B-Instruct-v0.1",  # Uncomment after Hugging Face login
                device=0 if torch.cuda.is_available() else -1,
                max_new_tokens=100,
                temperature=0.7,
                top_p=0.9
            )
            logger.info("Initialized LLM: facebook/opt-350m")
        except Exception as e:
            logger.error(f"Error initializing LLM: {str(e)}")
            raise

        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
You are a financial analyst assistant for CrediTrust. Summarize the key issues from the complaint data below to answer the question in 50-100 words. Focus on the specific product and issue asked. If the data is irrelevant, say: "I don't have enough information to answer this question." Do not repeat the question, avoid speculation, and ensure the answer is concise and clear.

**Question**: {question}
**Complaint Data**: {context}

**Answer**:
"""
        )

    def retrieve(self, question, k=8, product_filter=None):
        results = self.vector_store.similarity_search_with_score(question, k=k*2)
        logger.info(f"Retrieved {len(results)} chunks for question: {question}")
        
        retrieved_chunks = [(doc.page_content, doc.metadata, score) for doc, score in results]
        
        if product_filter:
            filtered_chunks = [
                (text, meta, score) for text, meta, score in retrieved_chunks 
                if meta.get('product', '').lower() == product_filter.lower()
            ][:k]
            if filtered_chunks:
                retrieved_chunks = filtered_chunks
                logger.info(f"Filtered to {len(retrieved_chunks)} {product_filter} chunks")
            else:
                logger.warning(f"No {product_filter} chunks found, using top {k} chunks")
        
        retrieved_chunks = retrieved_chunks[:k]
        for _, meta, score in retrieved_chunks:
            logger.debug(f"Chunk ID: {meta.get('complaint_id', 'N/A')}, Product: {meta.get('product', 'N/A')}, Score: {score:.2f}")
        
        return [(text, meta) for text, meta, _ in retrieved_chunks]

    def generate_answer(self, question, product_filter=None):
        chunks = self.retrieve(question, k=8, product_filter=product_filter)
        context = "\n".join([chunk[0] for chunk in chunks])
        
        debug_info = {
            "retrieved_chunks": [(chunk, meta, self.vector_store.similarity_search_with_score(question, k=1)[0][1]) 
                                 for chunk, meta in chunks]
        }
        
        prompt = self.prompt_template.format(context=context, question=question)
        
        try:
            result = self.llm(prompt, max_new_tokens=100, do_sample=True, temperature=0.7)
            answer = result[0]["generated_text"].replace(prompt, "").strip()
            logger.info(f"Generated answer for question: {question}")
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            answer = "Error generating answer."
        
        return {
            "question": question,
            "answer": answer,
            "retrieved_chunks": chunks,
            "debug_info": debug_info
        }