import gradio as gr
from src.rag_pipeline import RAGPipeline
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from huggingface_hub import snapshot_download
from huggingface_hub import snapshot_download

snapshot_download(repo_id="meta-llama/Llama-2-7b-hf", repo_type="model")

# Initialize your RAG pipeline (update the paths if needed)
rag = RAGPipeline(
    vector_store_path="C:/Users/Saron/MyWorkSpace/kifya/week6/rag-complaint-chatbot/vector_store"
)

# Define the Gradio function
def answer_question(question, product_filter):
    if not question.strip():
        return "Please enter a question.", ""
    result = rag.generate_answer(question, product_filter)
    answer = result["answer"]
    sources = result["retrieved_chunks"]

    sources_text = "\n\n".join(
        [f"• {chunk[0][:200]}..." for chunk in sources]  # Show first 200 chars
    )
    return answer, sources_text

# Create Gradio Blocks UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🗂️ Complaint Chatbot\nAsk any question about customer complaints.")
    with gr.Row():
        question = gr.Textbox(label="Your Question", placeholder="Type your question here...", lines=2)
        product = gr.Textbox(label="Filter by Product (optional)", placeholder="e.g. Credit card")

    with gr.Row():
        ask_btn = gr.Button("Ask")
        clear_btn = gr.Button("Clear")

    answer = gr.Textbox(label="AI Answer", placeholder="Answer will appear here...", lines=4)
    sources = gr.Textbox(label="Retrieved Sources", placeholder="Relevant complaint excerpts...", lines=8)

    ask_btn.click(answer_question, inputs=[question, product], outputs=[answer, sources])
    clear_btn.click(lambda: ("", "", ""), inputs=None, outputs=[question, product, answer, sources])

# Launch
if __name__ == "__main__":
    demo.launch()
