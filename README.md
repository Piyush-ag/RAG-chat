Here’s your content converted into Markdown format:

# Tech Notes
**Tech Notes:** [here](https://docs.google.com/document/d/19BR5yyjkEyBbxJGNeeTxqNOojOVnvbB2ETjn1antwX4/edit?usp=sharing)
**Hallucination and Boundary Test:** [here](https://docs.google.com/document/d/1idHrVoeZVzvd5H-JuOjWftX2F4bc4BSN2H6985oSsBc/edit?usp=sharing)

---

## Approach

This project is based on a **Retrieval-Augmented Generation (RAG)** pipeline that integrates document retrieval with **GPT-4o** to enable intelligent interactions with PDFs. Users can upload PDFs, ask context-based questions, and receive accurate, source-backed responses.

---

## Model

The system uses a combination of models and tools:

- **GPT-4o (ChatOpenAI)** – A powerful language model for contextual understanding and structured responses.
- **FAISS (Facebook AI Similarity Search)** – A vector-based retrieval mechanism for efficient document chunk searching.
- **OpenAI Embeddings (text-embedding-3-large)** – Converts document text into embeddings to improve retrieval accuracy.
- **PyMuPDF (fitz)** – Extracts raw text from PDFs.
- **Streamlit** – Provides a simple and interactive UI for users to upload and query PDFs.

---

## System Prompt

To ensure structured and contextually relevant responses, we use a predefined system persona:

> *“You are a financial analyst AI specializing in SEC filings. Provide concise, sourced, and structured answers. Always cite document sections where applicable.”*

This keeps the chatbot focused, informative, and aligned with factual sources.

---

## Architecture

### **1. PDF Processing**
- PDFs are uploaded through **Streamlit**.
- Text is extracted using **PyMuPDF**.
- The text is divided into sections using a **chunking function (chunk_by_headings)**, ensuring logical segmentation.
- The document chunks are embedded using **OpenAI Embeddings** and stored in **FAISS** for efficient retrieval.

### **2. Document Retrieval**
- The **FAISS vector store** is used for similarity-based searches.
- The retrieval method uses **Maximal Marginal Relevance (MMR)** to ensure that retrieved chunks are both relevant and diverse.

### **3. Query Processing & Response Generation**
- User queries are logged in session state.
- The **retriever fetches** the most relevant document chunks from FAISS.
- **GPT-4o** generates a response by integrating the query with the retrieved document chunks.
- The response is **streamed back in real-time** to provide a smooth chat experience.

### **4. User Interaction & Chat History**
- Users interact with the system via **Streamlit**.
- **Session state stores chat history** for continuity.
- **Retrieved document sections are displayed** for transparency, allowing users to verify sources.

---

## Insights & Challenges

### **What Worked Well**
**Accurate Document Retrieval** – The **MMR-based retriever** ensures relevant and diverse results.  
**Structured Answers with Citations** – The **system prompt** enforces clarity and verifiable sourcing.  
**User-Friendly UI** – **Streamlit** makes interactions seamless.  
**Scalability** – **FAISS** allows fast and efficient document retrieval, even with large datasets.  

### **Challenges & Areas for Improvement**
**Handling Noisy PDF Data** – Some PDFs contain **OCR errors**, making text extraction unreliable.  
**Chunking Strategy Optimization** – Larger chunks provide more context but **increase token usage**. A balance needs to be found.  
**Mitigating Hallucinations** – **GPT-4o** sometimes generates **unverifiable information**. Improving document citation and chunk ranking is a priority.  
**Latency Issues** – **Vector search and GPT-4o processing introduce delays**. Implementing **caching** for frequently queried embeddings could help.  

---

## Strengths & Weaknesses

| **Feature**             | **Strengths**                                       | **Weaknesses**                                      |
|-------------------------|-----------------------------------------------------|-----------------------------------------------------|
| **Document Retrieval**  | Fast and relevant search using **FAISS & MMR**      | Performance may degrade if document structure is inconsistent |
| **Response Quality**    | Provides **structured, well-sourced answers**       | **Hallucination risk** if documents lack relevant content |
| **Scalability**         | **FAISS enables quick searches** across large document collections | **High token usage** can be expensive |
| **User Experience**     | **Intuitive Streamlit-based UI**                    | **Dependent on API latency** and model response time |

---

## Future Enhancements

**Hybrid Retrieval** – Combine **dense and sparse retrieval** methods to improve search accuracy.  
**Better Document Parsing** – Improve **text extraction techniques** for PDFs with **OCR errors** or **non-standard formatting**.  
**Query Expansion** – Use **synonyms and keyword expansion** to improve search relevance.  
**Optimized Chunking** – Experiment with **semantic chunking** rather than fixed-size segmentation for better retrieval efficiency.  

