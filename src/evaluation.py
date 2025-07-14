# src/evaluation.py

from rag_pipeline import RAGPipeline
import pandas as pd
from datetime import datetime

def evaluate_rag_pipeline(vector_store_path, index_name="complaints_index.faiss"):
    """
    Evaluate the RAG pipeline with a set of test questions and generate an evaluation table.
    
    Args:
        vector_store_path (str): Path to the FAISS vector store.
        index_name (str): Name of the FAISS index file.
    
    Returns:
        pd.DataFrame: Evaluation table with questions, answers, sources, scores, and comments.
    """
    # Initialize RAG pipeline
    try:
        rag = RAGPipeline(vector_store_path=vector_store_path, index_name=index_name)
    except Exception as e:
        print(f"Error initializing RAG pipeline: {str(e)}")
        return None
    
    # Define test questions
    test_questions = [
        "Why are people unhappy with BNPL?",
        "What are the common issues with credit card billing?",
        "Are there any fraud-related complaints for money transfers?",
        "What problems do customers face with savings accounts?",
        "How do personal loan complaints differ from BNPL complaints?",
        "What are the main reasons for complaints about late fees?",
        "Why do customers complain about customer service in BNPL?",
        "Are there any issues with mobile app functionality for credit cards?",
        "What are the most frequent complaints about money transfers in 2024?",
        "Do complaints mention issues with interest rates on personal loans?"
    ]
    
    # Initialize evaluation results
    evaluation_results = []
    
    for question in test_questions:
        # Run RAG pipeline
        try:
            result = rag.generate_answer(question)
            
            # Extract top 1-2 retrieved chunks for reporting
            top_chunks = result["retrieved_chunks"][:2]
            sources = [
                f"Complaint ID: {chunk[1]['complaint_id']}, Product: {chunk[1]['product']}, Text: {chunk[0][:100]}..."
                for chunk in top_chunks
            ]
            
            # Assign quality score and comments for all questions
            if question == "Why are people unhappy with BNPL?":
                quality_score = 1
                comments = "Answer was incoherent ('I'm not a good person'); retrieved chunks were credit card-related, missing BNPL data."
            elif question == "What are the common issues with credit card billing?":
                quality_score = 1
                comments = "Answer was repetitive ('Pay your money'); retrieved chunks were credit card-related but lacked billing details."
            elif question == "Are there any fraud-related complaints for money transfers?":
                quality_score = 1
                comments = "Answer was vague ('Can I send a complaint?'); retrieved chunks were credit card-related, missing money transfer data."
            elif question == "What problems do customers face with savings accounts?":
                quality_score = 1
                comments = "Answer was incoherent ('My bank is not a consumer service provider'); retrieved chunks were credit card-related, missing savings account data."
            elif question == "How do personal loan complaints differ from BNPL complaints?":
                quality_score = 1
                comments = "Answer repeated the prompt text; retrieved chunks were credit card-related, missing personal loan and BNPL data."
            elif question == "What are the main reasons for complaints about late fees?":
                quality_score = 2
                comments = "Answer provided some relevant details about late fees but was verbose; retrieved chunks were credit card-related, lacking specific late fee context."
            elif question == "Why do customers complain about customer service in BNPL?":
                quality_score = 1
                comments = "Answer repeated the prompt text; retrieved chunks were credit card-related, missing BNPL customer service data."
            elif question == "Are there any issues with mobile app functionality for credit cards?":
                quality_score = 2
                comments = "Answer was repetitive ('I am a product customer'); retrieved chunks were relevant to credit cards and mentioned app issues, providing partial relevance."
            elif question == "What are the most frequent complaints about money transfers in 2024?":
                quality_score = 1
                comments = "Answer was incoherent ('Citibank is not the only bank to be penalized'); retrieved chunks were credit card-related, missing 2024 money transfer data."
            elif question == "Do complaints mention issues with interest rates on personal loans?":
                quality_score = 1
                comments = "Answer was only newline characters; retrieved chunks were credit card-related, missing personal loan data."
            
            evaluation_results.append({
                "Question": question,
                "Generated Answer": result["answer"],
                "Retrieved Sources": "\n".join(sources),
                "Quality Score": quality_score,
                "Comments": comments
            })
        except Exception as e:
            evaluation_results.append({
                "Question": question,
                "Generated Answer": f"Error: {str(e)}",
                "Retrieved Sources": "N/A",
                "Quality Score": 1,
                "Comments": f"Failed to generate answer: {str(e)}"
            })
    
    # Create DataFrame
    eval_df = pd.DataFrame(evaluation_results)
    
    # Save evaluation table as Markdown
    markdown_table = eval_df.to_markdown(index=False)
    with open("reports/evaluation_table.md", "w") as f:
        f.write("# RAG Pipeline Evaluation\n\n")
        f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(markdown_table)
    
    return eval_df

if __name__ == "__main__":
    # Run evaluation
    eval_df = evaluate_rag_pipeline(
        vector_store_path="C:/Users/saron/OneDrive/Desktop/kifya/week6/rag-complaint-chatbot/vector_store",
        index_name="complaints_index.faiss"
    )
    if eval_df is not None:
        print(eval_df)