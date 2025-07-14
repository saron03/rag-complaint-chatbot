import pandas as pd
from rag_pipeline import RAGPipeline
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_rag():
    try:
        rag = RAGPipeline(
            vector_store_path="C:/Users/saron/OneDrive/Desktop/kifya/week6/rag-complaint-chatbot/vector_store",
            index_name="complaints_index.faiss",
            embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    except Exception as e:
        logger.error(f"Error initializing RAG pipeline: {str(e)}")
        return

    questions = [
        ("Why are people unhappy with BNPL?", "bnpl"),
        ("What are the common issues with credit card billing?", "credit card"),
        ("Are there any fraud-related complaints for money transfers?", "money transfer"),
        ("What problems do customers face with savings accounts?", "savings account"),
        ("How do personal loan complaints differ from BNPL complaints?", "personal loan"),
        ("Why do customers complain about customer service in BNPL?", "bnpl"),
        ("Are there any issues with mobile app functionality for credit cards?", "credit card")
    ]

    eval_results = []

    for question, product_filter in questions:
        try:
            result = rag.generate_answer(question, product_filter=product_filter)
            answer = result["answer"]
            retrieved_chunks = result["retrieved_chunks"]

            # Assign quality score based on relevance and coherence
            if "I don't have enough information" in answer or "not a consumer" in answer or "default rate" in answer:
                quality_score = 1
                comments = "Answer is irrelevant or incoherent, likely due to LLM limitations or insufficient relevant chunks."
            elif len(retrieved_chunks) < 2 or not all(product_filter in meta['product'].lower() for _, meta in retrieved_chunks[:2]):
                quality_score = 2
                comments = "Answer is partially relevant but lacks depth due to limited or irrelevant retrieved chunks."
            elif len(answer.split()) < 50 or len(answer.split()) > 100:
                quality_score = 3
                comments = "Answer addresses the question but is too brief or verbose, missing key details."
            else:
                quality_score = 4
                comments = "Answer is relevant and mostly clear but could be more concise or detailed."

            sources = [
                f"Complaint ID: {meta['complaint_id']}, Product: {meta['product']}, Text: {text[:50]}..."
                for text, meta in retrieved_chunks[:2]
            ]

            eval_results.append({
                "Question": question,
                "Generated Answer": answer,
                "Retrieved Sources": sources,
                "Quality Score": quality_score,
                "Comments": comments
            })
            logger.info(f"Evaluated question: {question}")
        except Exception as e:
            logger.error(f"Error evaluating question '{question}': {str(e)}")
            eval_results.append({
                "Question": question,
                "Generated Answer": "Error generating answer.",
                "Retrieved Sources": [],
                "Quality Score": 1,
                "Comments": f"Error: {str(e)}"
            })

    # Create evaluation table
    eval_df = pd.DataFrame(eval_results)

    # Ensure reports directory exists
    os.makedirs("C:/Users/saron/OneDrive/Desktop/kifya/week6/rag-complaint-chatbot/reports", exist_ok=True)

    # Save to Markdown file
    output_path = "C:/Users/saron/OneDrive/Desktop/kifya/week6/rag-complaint-chatbot/reports/evaluation_table.md"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# RAG Pipeline Evaluation Table\n\n")
        f.write(eval_df.to_markdown(index=False))
        f.write("\n\nGenerated on: 2025-07-09\n")

    # Print table for verification
    print("\nEvaluation Table:")
    print(eval_df.to_markdown(index=False))
    print(f"\nTable saved to: {output_path}")

if __name__ == "__main__":
    evaluate_rag()