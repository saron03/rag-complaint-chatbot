Paste your rich t

## **EDA Summary — Task 1**

**Dataset Overview:**  
I analyzed the Consumer Financial Protection Bureau (CFPB) complaint dataset, which initially contained **9,609,797** complaint records with 18 columns, including product information, narrative text, company details, and submission metadata.

**Key Findings:**

- Out of ~9.6 million complaints, only about **3 million** include a **Consumer complaint narrative**. This means roughly **66%** of records do not contain the detailed text needed for our RAG-based chatbot.
- For my focus on five core products (_Credit Cards, Personal Loans, BNPL, Savings Accounts, Money Transfers_), I found narrative data only for **Credit Cards** and **Money Transfers** in the sample I filtered so far — resulting in **82,164** usable complaints: **80,667** for Credit Cards and **1,497** for Money Transfers.
- Narrative lengths vary widely, with many narratives being short but informative. Further cleaning will help improve the quality of embeddings for semantic search.


**Next Steps:**  
Based on this EDA, I filtered out complaints with no narratives and limited the dataset to the products I'm targeting. The cleaned dataset of **~82,000 complaints** will be used to build the vector database for our RAG pipeline. Additional cleaning — like lowercasing text and removing special characters — was applied to ensure better semantic embeddings and more reliable retrieval performance.

* * *

✅ **Deliverable:** Cleaned data saved as `data/filtered_complaints.csv` — ready for embedding in the next stage.