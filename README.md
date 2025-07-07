## **Task 1 — Exploratory Data Analysis & Preprocessing**

### ✅ **Objective**

In this task, I explored and prepared the **CFPB consumer complaint dataset** to create a clean, filtered dataset for our **RAG-powered complaint-answering chatbot**.

* * *

### **What I Did**

* Loaded the full CFPB complaints dataset (**~9.6 million rows**).
* Analyzed complaint distribution across product categories.
* Checked narrative availability: found ~3M complaints with narratives.
* Filtered to keep only 5 key products:
    * **Credit Cards**
    * **Personal Loans**
    * **Buy Now, Pay Later (BNPL)**
    * **Savings Accounts**
    * **Money Transfers**
* Removed complaints without narratives.
* Cleaned text (lowercasing, special characters removal) to prepare for embedding.


* * *

### **Key Insights**

* Final usable data: **~82,000** complaints with detailed narratives (mainly Credit Cards + Money Transfers).
* The data is now suitable for building vector embeddings for semantic search.


* * *

### **Outputs**

* Filtered Data: `data/filtered_complaints.csv`
* Notebook: `notebooks/01_eda_preprocessing.ipynb`
   
* **EDA Summary:** `reports/eda_summary.md`