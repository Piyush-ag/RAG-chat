import os
import fitz  # PyMuPDF for PDFs
import streamlit as st
import tiktoken
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.docstore.document import Document
import re
import tempfile
import time
import pdfplumber

# Set OpenAI API Key
os.environ["OPENAI_API_KEY"] = "your_own_api"

# Streamlit UI Configuration
st.set_page_config(page_title="Chat with Your PDFs (GPT-4o)", layout="wide")
st.title("Chat with Your PDFs (GPT-4o)")

# Initialize Session State for Chat History & Vector Store
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # Stores messages in order
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None  # Store FAISS Index

# File Upload UI
uploaded_files = st.file_uploader("Upload PDF Files", type=["pdf"], accept_multiple_files=True)

# Function to extract text from PDFs
def extract_text(filepath):
    """Extracts text from a PDF using PDFPlumber."""
    text = ""
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n" if page.extract_text() else ""
    return text

# Function to chunk text using `chunk_by_headings`
def chunk_by_headings(text, chunk_size=256, overlap=50):
    sections = re.split(r"\n(Item\s+\d+\.?.*?)\n", text)  # Detects "Item 1", "Item 2", etc.
    chunks = []
    
    for i in range(1, len(sections), 2):
        section_title = sections[i].strip()
        section_text = sections[i+1].strip()
        
        tokens = section_text.split()
        for j in range(0, len(tokens), chunk_size - overlap):
            chunk = " ".join(tokens[j:j+chunk_size])
            chunks.append(f"{section_title}\n{chunk}")  # Keep section titles for context
    return chunks

if uploaded_files:
    st.success("✅ PDFs uploaded successfully. Processing files...")

    # Process PDFs if FAISS store doesn't exist in session state
    if st.session_state.vector_store is None:
        with st.spinner("Processing your PDFs..."):
            documents = []
            with tempfile.TemporaryDirectory() as temp_dir:
                for file in uploaded_files:
                    temp_file_path = os.path.join(temp_dir, file.name)
                    with open(temp_file_path, "wb") as f:
                        f.write(file.getbuffer())

                    # Extract text and chunk using `chunk_by_headings`
                    text = extract_text(temp_file_path)
                    chunks = chunk_by_headings(text)
                    
                    for chunk in chunks:
                        doc = Document(page_content=chunk, metadata={"source": file.name})
                        documents.append(doc)

            # Store documents in FAISS
            embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
            vectorstore = FAISS.from_documents(documents, embedding=embeddings)
            st.session_state.vector_store = vectorstore

        st.success("✅ PDFs processed! You can now start chatting.")

# Load Vectorstore
vectorstore = st.session_state.vector_store

if vectorstore:
    # **Use your existing retriever settings (MMR with Re-Ranking)**
    retriever = vectorstore.as_retriever(
        search_type="mmr",  # Maximal Marginal Relevance
        search_kwargs={"k": 20, "fetch_k": 50, "lambda_mult": 0.5}
    )

    # GPT-4o Chat Model
    llm = ChatOpenAI(model="gpt-4o", temperature=0.3, max_retries=10)

    # RAG Pipeline (Retrieval + GPT-4o)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    # Display Chat History (Sequentially)
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User Input Field
    query = st.chat_input("Ask a question about your PDFs:")
    
    if query:
        # Store and display user message immediately
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # AI Persona for Structured Answers
        persona = """
        You are a financial analyst AI specializing in SEC filings. Provide concise, sourced, and structured answers.
        Always cite document sections where applicable.
        """
        
        with st.spinner("Thinking..."):
            result = qa.invoke({"query": f"{persona}\nUser Query: {query}"})

            # Extract and format the response properly
            formatted_response = f"### **Answer:**\n{result['result']}"

            # Extract retrieved document chunks for display
            retrieved_chunks = result.get("source_documents", [])

        # Store assistant response in session history
        st.session_state.chat_history.append({"role": "assistant", "content": formatted_response})

        # Display AI Response with Streaming Effect
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            for chunk in formatted_response.split(" "):  # Stream word by word
                full_response += chunk + " "
                message_placeholder.markdown(formatted_response.replace("$", "\\$"))
                time.sleep(0.05)  # Simulate typing effect

        # Display Retrieved Chunks in an Expandable Section
        with st.expander("Retrieved Sections from Documents"):
            for i, doc in enumerate(retrieved_chunks):
                section_title = doc.page_content.replace("$", "\\$")
                st.markdown(f"** Chunk {i+1}:** {section_title}")
                st.markdown(f"** Source:** {doc.metadata.get('source', 'Unknown')}")
                st.markdown("---")
else:
    st.info("Please upload PDF files to begin.")