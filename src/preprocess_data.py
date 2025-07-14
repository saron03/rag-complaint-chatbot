import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from tqdm import tqdm
import logging
import os
import csv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def preprocess_complaints(
    csv_file,
    output_vector_store_path,
    index_name="complaints_index.faiss",
    batch_size=1000,
    max_complaints=None,
    embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"
):
    """
    Preprocess complaint data and build a FAISS vector store in batches.
    """
    # Load dataset with robust parsing
    try:
        df = pd.read_csv(
            csv_file,
            quoting=csv.QUOTE_ALL,  # Quote all fields to handle commas
            on_bad_lines='warn',    # Log bad rows instead of failing
            encoding='utf-8'
        )
        logger.info(f"Loaded dataset with {len(df)} complaints")
        logger.info(f"Columns found: {list(df.columns)}")
        logger.info(f"First few rows:\n{df.head().to_string()}")
        
        # Check if header is correct
        required_columns = ['complaint_id', 'product', 'text']
        if not all(col in df.columns for col in required_columns):
            logger.warning("Incorrect or missing header. Setting column names manually.")
            df = pd.read_csv(
                csv_file,
                names=required_columns,
                header=None,
                quoting=csv.QUOTE_ALL,
                on_bad_lines='warn',
                encoding='utf-8'
            )
            logger.info(f"Re-loaded dataset with columns: {list(df.columns)}")
            logger.info(f"First few rows after re-load:\n{df.head().to_string()}")
        
        if max_complaints:
            df = df.head(max_complaints)
        logger.info(f"Processing {len(df)} complaints")
    except Exception as e:
        logger.error(f"Error loading CSV: {str(e)}")
        raise

    # Check for required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        raise KeyError(f"Missing required columns in CSV: {missing_columns}")

    # Clean and preprocess data
    df = df.dropna(subset=required_columns)
    df['text'] = df['text'].str.replace(r'XXXX+', ' ', regex=True)
    df['product'] = df['product'].str.lower().str.strip()

    # Map product names
    product_mapping = {
        'buy now pay later': 'bnpl',
        'credit card or prepaid card': 'credit card',
        'checking or savings account': 'savings account',
        'money transfer, virtual currency, or money service': 'money transfer',
        'personal loan': 'personal loan',
        'credit card': 'credit card'
    }
    df['product'] = df['product'].map(product_mapping).fillna(df['product'])
    logger.info(f"Cleaned dataset, {len(df)} complaints remain after filtering")

    # Check for fragmented complaints
    if df['complaint_id'].duplicated().any():
        logger.info("Detected fragmented complaints; merging by complaint_id")
        df = df.groupby(['complaint_id', 'product'])['text'].apply(lambda x: ' '.join(x)).reset_index()
        logger.info(f"Merged fragments, {len(df)} unique complaints remain")

    # Add sample complaints for all product types if missing
    product_types = ['bnpl', 'credit card', 'money transfer', 'savings account', 'personal loan']
    for product in product_types:
        product_count = len(df[df['product'] == product])
        if product_count == 0:
            logger.warning(f"No {product} complaints found. Adding sample complaints.")
            sample_complaints = {
                'bnpl': [
                    {'complaint_id': '99999901', 'product': 'bnpl', 'text': 'I used a BNPL service, but the payment terms were unclear, leading to $30 late fees. Customer service was unresponsive.'},
                    {'complaint_id': '99999902', 'product': 'bnpl', 'text': 'The BNPL provider charged me twice for an installment. Their app was confusing and lacked payment history.'},
                    {'complaint_id': '99999903', 'product': 'bnpl', 'text': 'The BNPL service didn’t disclose interest rates upfront, and I was charged $50 extra. Support ignored my emails.'}
                ],
                'money transfer': [
                    {'complaint_id': '99999904', 'product': 'money transfer', 'text': 'A fraudulent money transfer was initiated from my account. The bank refused to refund the $200 lost, claiming I authorized it.'},
                    {'complaint_id': '99999905', 'product': 'money transfer', 'text': 'The money transfer service delayed my transaction for weeks, and customer support provided no updates.'}
                ],
                'savings account': [
                    {'complaint_id': '99999906', 'product': 'savings account', 'text': 'My savings account was charged hidden maintenance fees of $15 monthly without prior notice. The bank refused to waive them.'},
                    {'complaint_id': '99999907', 'product': 'savings account', 'text': 'Accessing my savings account online was impossible due to constant app crashes. Customer service was unhelpful.'}
                ],
                'personal loan': [
                    {'complaint_id': '99999908', 'product': 'personal loan', 'text': 'The personal loan had a high interest rate not disclosed initially, increasing my payments by $100 monthly.'},
                    {'complaint_id': '99999909', 'product': 'personal loan', 'text': 'Repaying my personal loan early resulted in unexpected prepayment penalties. The lender’s terms were unclear.'}
                ],
                'credit card': [
                    {'complaint_id': '99999910', 'product': 'credit card', 'text': 'My credit card had unauthorized charges after losing my wallet. The bank refused to reverse them.'}
                ]
            }
            df = pd.concat([df, pd.DataFrame(sample_complaints[product])], ignore_index=True)
            logger.info(f"Added {len(sample_complaints[product])} sample {product} complaints")

    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )

    # Initialize embedding model
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={"local_files_only": True}
        )
        logger.info(f"Initialized embedding model: {embedding_model_name}")
    except Exception as e:
        logger.error(f"Error initializing embedding model: {str(e)}")
        raise

    # Process complaints in batches
    documents = []
    metadatas = []
    for i in tqdm(range(0, len(df), batch_size), desc="Processing batches"):
        batch = df[i:i + batch_size]
        batch_docs = []
        batch_metas = []

        for _, row in batch.iterrows():
            if len(row['text']) < 50:
                continue
            chunks = text_splitter.split_text(row['text'])
            for chunk in chunks:
                batch_docs.append(chunk)
                batch_metas.append({
                    'complaint_id': str(row['complaint_id']),
                    'product': row['product']
                })

        documents.extend(batch_docs)
        metadatas.extend(batch_metas)
        logger.info(f"Processed batch {i//batch_size + 1}: {len(batch_docs)} chunks")

        if documents:
            try:
                if i == 0:
                    vector_store = FAISS.from_texts(documents, embeddings, metadatas=metadatas)
                else:
                    vector_store = FAISS.load_local(
                        folder_path=output_vector_store_path,
                        embeddings=embeddings,
                        index_name=index_name.replace(".faiss", ""),
                        allow_dangerous_deserialization=True
                    )
                    vector_store.add_texts(documents, metadatas=metadatas)
                
                vector_store.save_local(output_vector_store_path, index_name.replace(".faiss", ""))
                logger.info(f"Saved FAISS index after batch {i//batch_size + 1}")
                
                documents = []
                metadatas = []
            except Exception as e:
                logger.error(f"Error building/saving FAISS index for batch {i//batch_size + 1}: {str(e)}")
                raise

    logger.info(f"Completed preprocessing, saved FAISS vector store to {output_vector_store_path}")

if __name__ == "__main__":
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    
    preprocess_complaints(
        csv_file="C:/Users/saron/OneDrive/Desktop/kifya/week6/rag-complaint-chatbot/vector_store/chunks_metadata.csv",
        output_vector_store_path="C:/Users/saron/OneDrive/Desktop/kifya/week6/rag-complaint-chatbot/vector_store",
        batch_size=1000,
        max_complaints=10000
    )