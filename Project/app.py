"""
app_local_free.py

Free-local PharmaIntel RAG application:
- Uses LangChain for scraping/indexing/RAG orchestration.
- Uses sentence-transformers (all-MiniLM-L6-v2) for local embeddings.
- Uses FAISS for local vector store storage.
- Uses Ollama (Mistral default) for local LLM chat/RAG responses.
- Uses Streamlit with enhanced CSS for a beautiful, modern UI.
"""

import os
import json
import streamlit as st
import shutil
from typing import List
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import ConversationalRetrievalChain 
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_community.llms import Ollama


# -----------------------
# Config & Theming
# -----------------------
st.set_page_config(
    page_title="PharmaIntel — Local RAG", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- Aesthetic Custom Theme Styling (CSS Injection) ---
st.markdown("""
<style>
/* Global Variables for Streamlit Theming */
:root {
    --primary-color: #1a75ff; /* Clinical Blue */
    --background-color: #ffffff; /* Clean White */
    --secondary-background-color: #f0f4f8; /* Light gray for sidebar/status */
    --text-color: #333333;
}

/* 1. Sidebar Styling: Make it a distinct, clean navigation panel */
[data-testid="stSidebar"] {
    background-color: var(--secondary-background-color); 
    color: var(--text-color);
    border-right: 2px solid var(--primary-color); /* A subtle blue line */
}

/* 2. Main Content Layout: Max width and cleaner padding */
.main .block-container { 
    padding-top: 2rem;
    padding-bottom: 5rem;
    max-width: 1400px; /* Wider content area */
}

/* 3. Typography and Headings */
h1, h2, h3, h4 {
    color: var(--primary-color); /* Highlight main headers */
    font-weight: 600;
}
.stMarkdown h1 {
    font-size: 2.5em;
    border-bottom: 2px solid #e0e0e0;
    padding-bottom: 10px;
}

/* 4. Primary Button Styling (More modern feel) */
.stButton>button {
    border-radius: 20px; /* Rounded pill shape */
    border: none;
    color: white;
    background-color: var(--primary-color);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); /* Subtle shadow */
    transition: all 0.2s ease-in-out;
}
.stButton>button:hover {
    background-color: #005ce6;
    transform: translateY(-2px); /* Lift effect on hover */
    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
}

/* 5. Chat Message Customization (Using st.chat_message) */
[data-testid="stChatMessage"] {
    padding: 10px 15px;
    border-radius: 12px;
    margin-bottom: 15px;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05); /* Light shadow for depth */
}

/* Styling the User's Message Bubble */
[data-testid="stChatMessage"][data-state="user"] {
    background-color: #e6f0ff; /* Light blue background */
    border-bottom-right-radius: 2px; /* Slight conversation bubble look */
    text-align: right;
}

/* Styling the Assistant's Message Bubble */
[data-testid="stChatMessage"][data-state="assistant"] {
    background-color: #ffffff; /* White background */
    border: 1px solid #e0e0e0;
    border-bottom-left-radius: 2px;
    text-align: left;
}

/* Hide the default Streamlit Header (already in your code, keeping it for completeness) */
[data-testid="stHeader"] {
    display: none;
}

</style>
""", unsafe_allow_html=True)


# Data sources (from your proposal)
DATA_SOURCES = {
    "Drugs.com": "https://www.drugs.com",
    "Medscape Drug Reference": "https://reference.medscape.com/drugs",
    "FDA DailyMed": "https://dailymed.nlm.nih.gov/dailymed"
}

# Where to persist FAISS index and metadata
PERSIST_DIR = os.path.join(os.getcwd(), "faiss_pharma_db")

# HuggingFace model for embeddings (local, free)
HF_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Ollama model name (ensure you pulled this using ollama)
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "mistral")  # change default if needed

# -----------------------
# Helpers
# -----------------------
def scrape_with_webbaseloader(url: str) -> str:
    loader = WebBaseLoader(url)
    docs = loader.load()
    text = "\n\n".join([d.page_content for d in docs if d.page_content])
    return text

def chunk_text(text: str, chunk_size=900, chunk_overlap=150) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

def build_embeddings_and_faiss(docs: List[Document], persist_dir: str):
    # Create embeddings object (uses sentence-transformers under the hood)
    hf = HuggingFaceEmbeddings(model_name=HF_EMBEDDING_MODEL, model_kwargs={"device": "cpu"})
    
    # Check and remove old index
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)
    os.makedirs(persist_dir, exist_ok=True)
    
    # Build FAISS vector store
    texts = [d.page_content for d in docs]
    metadatas = [d.metadata if d.metadata else {} for d in docs]
    faiss_db = FAISS.from_texts(texts, embedding=hf, metadatas=metadatas)
    
    # persist
    faiss_db.save_local(persist_dir)
    return faiss_db

@st.cache_resource
def load_faiss(persist_dir: str):
    if not os.path.exists(persist_dir) or not os.listdir(persist_dir):
        return None
    # Loading the embedding model once
    hf = HuggingFaceEmbeddings(model_name=HF_EMBEDDING_MODEL, model_kwargs={"device": "cpu"})
    faiss_db = FAISS.load_local(
        persist_dir, 
        hf, 
        allow_dangerous_deserialization=True # FIX: Necessary for loading local pickle files securely
    )
    return faiss_db

# -----------------------
# UI: Sidebar controls
# -----------------------
st.sidebar.markdown('## 💊 PharmaIntel: Local RAG')
mode = st.sidebar.selectbox("Select Application Mode", ["Chat (RAG)", "Scrape & Index"])
st.sidebar.markdown("---")

# Global Settings
selected_sources = st.sidebar.multiselect("Choose web sources to scrape", list(DATA_SOURCES.keys()), default=list(DATA_SOURCES.keys()))
ollama_model_input = st.sidebar.text_input("Ollama Model Name", value=OLLAMA_MODEL, help="E.g., mistral, llama2. Must be pulled locally.")
st.session_state.ollama_model = ollama_model_input
st.sidebar.markdown("---")

# Indexing Options
if mode == "Scrape & Index":
    st.sidebar.subheader("Index Options")
    use_jsonl_dataset = st.sidebar.checkbox("Use drug_dataset.jsonl", value=False, help="Load structured data instead of live scraping.")
    recreate_index = st.sidebar.checkbox("Force rebuild index", value=False, help="Delete existing FAISS index before creation.")
    st.sidebar.markdown("---")

# -----------------------
# Scrape & Index Logic
# -----------------------
if mode == "Scrape & Index":
    st.title("🌐 Scrape & Index Knowledge Base")
    st.info("Select a method below to build or rebuild your local knowledge vector store.")
    
    if st.button("Start Indexing Process", type="primary") or st.session_state.get('run_indexing_after_load', False):
        st.session_state['run_indexing_after_load'] = False
        all_docs = []
        
        with st.status("Starting Indexing...", expanded=True) as status:
            
            # --- 1. Load Documents ---
            if use_jsonl_dataset:
                jsonl_path = "drug_dataset.jsonl"
                status.update(label=f"Loading documents from **{jsonl_path}**...")
                if not os.path.exists(jsonl_path):
                    st.error(f"'{jsonl_path}' not found. Please ensure it is in the same directory.")
                    status.update(label="Indexing failed.", state="error")
                    st.stop()
                
                try:
                    with open(jsonl_path, "r", encoding="utf-8") as fh:
                        for line in fh:
                            rec = json.loads(line)
                            
                            text_parts = []
                            if rec.get("generic_name"):
                                text_parts.append(f"Generic name: {rec['generic_name']}")
                            if rec.get("brand_names"):
                                text_parts.append(f"Brand names: {rec['brand_names']}")

                            for f in ["indications", "mechanism", "dosage", "side_effects", "interactions"]:
                                if rec.get(f):
                                    text_parts.append(f"{f.capitalize()}: {rec[f]}")

                            text_blob = "\n\n".join(text_parts) if text_parts else rec.get("raw_excerpt", "")
                            
                            metadata = {
                                "source_url": rec.get("source_url"),
                                "site": rec.get("site"),
                            }
                            # Chunking structured data to fit into model context
                            docs_from_jsonl = chunk_text(text_blob)
                            for i, c in enumerate(docs_from_jsonl):
                                metadata_chunked = metadata.copy()
                                metadata_chunked["chunk_id"] = f"{rec.get('generic_name', 'data')}_{i}"
                                all_docs.append(Document(page_content=c, metadata=metadata_chunked))
                    
                    st.success(f"Loaded and chunked **{len(all_docs)}** sections from JSONL dataset.")
                except Exception as e:
                    st.error(f"Error parsing JSONL: {e}")
                    status.update(label="Indexing failed.", state="error")
                    st.stop()
                    
            else: # --- Live Scraping ---
                status.update(label="Starting live web scraping...")
                total_chunks = 0
                for name in selected_sources:
                    url = DATA_SOURCES[name]
                    status.markdown(f"- Scraping **{name}**: {url}...")
                    try:
                        text = scrape_with_webbaseloader(url)
                    except Exception as e:
                        st.error(f"Error scraping {url}: {e}")
                        text = ""
                        
                    if text:
                        chunks = chunk_text(text)
                        total_chunks += len(chunks)
                        for i, c in enumerate(chunks):
                            metadata = {"source": name, "url": url, "chunk_id": f"{name}_{i}"}
                            all_docs.append(Document(page_content=c, metadata=metadata))
                        status.markdown(f"**{len(chunks)}** chunks processed from **{name}**.")
                    else:
                        st.warning(f"No text scraped from {name}")

            if not all_docs:
                st.error("No documents collected — cannot build index.")
                status.update(label="Indexing failed.", state="error")
                st.stop()
            
            status.update(label=f"Total documents collected: **{len(all_docs)}**.")
            
            # --- 2. Build FAISS Index ---
            status.update(label=f"Building embeddings (model: **{HF_EMBEDDING_MODEL}**) and creating FAISS index...")
            
            if recreate_index and os.path.exists(PERSIST_DIR):
                shutil.rmtree(PERSIST_DIR)
                status.warning("Existing index removed (recreate requested).")
            
            try:
                build_embeddings_and_faiss(all_docs, PERSIST_DIR)
                status.update(label=f"Index built successfully and saved to **{PERSIST_DIR}**.", state="complete")
                st.balloons()
            except Exception as e:
                st.error(f"Error during FAISS & embeddings: {e}")
                status.update(label="Indexing failed.", state="error")


# -----------------------
# Chat (RAG) Logic
# -----------------------
else:
    st.title("🤖 Chat with Pharma Knowledge")
    
    # 1. Load Index
    faiss_db = load_faiss(PERSIST_DIR)
    if not faiss_db:
        st.warning("⚠️ No FAISS index found. Please run the **'Scrape & Index'** mode first.")
        st.stop()

    # 2. Configure LLM/Chain
    try:
        # Initializing the model (The @st.cache_resource decorator helps here)
        llm = Ollama(model=st.session_state.ollama_model, base_url="http://localhost:11434")
        
        # Build retriever
        retriever = faiss_db.as_retriever(search_kwargs={"k": 4})
        
        # Define a highly directive system prompt (Strict RAG Prompt)
        CUSTOM_PROMPT = """
        You are a highly specialized and accurate pharmaceutical knowledge assistant. 
        Your goal is to answer the user's question accurately and concisely.

        CRITICAL INSTRUCTION: You MUST use ONLY the provided context documents (Source Documents) to answer the question. 
        If the answer is not explicitly found within the context, you MUST politely state that you cannot find the specific information in the provided sources. 
        Do not use your external knowledge or browse the web. Do not include URLs in your final answer; they are only for reference.
        ---
        Context: {context}
        ---
        Question: {question}
        Answer:
        """
        STRICT_PROMPT = PromptTemplate(template=CUSTOM_PROMPT, input_variables=["context", "question"])
        
        # Conversational retrieval chain with strict prompt injection
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": STRICT_PROMPT} 
        )

    except Exception as e:
        st.error("❌ Ollama/LangChain Initialization Failed.")
        st.code(f"Error: {e}", language="python")
        st.info("Check 1: Is the Ollama desktop app running?")
        st.info(f"Check 2: Did you run `ollama pull {st.session_state.ollama_model}` in your terminal?")
        st.stop()
        
    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        
    st.markdown("---")
    
    # Display chat history using modern st.chat_message
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                with st.expander("📚 Sources Used"):
                    for d in message["sources"][:4]:
                        src = d.metadata.get("source", d.metadata.get("site", "Local Data"))
                        url = d.metadata.get("source_url", d.metadata.get("url", "N/A"))
                        st.markdown(f"**{src}** — {url}")
                        st.caption(d.page_content[:300].replace('\n', ' ').strip() + "...")
                        

    # Chat input and processing
    if query := st.chat_input("Ask about a drug, dosage, side effects, or interactions..."):
        # Display user query in chat
        with st.chat_message("user"):
            st.write(query)
            
        # Add query to history
        st.session_state.chat_history.append({"role": "user", "content": query})

        # Process the query with RAG chain
        with st.spinner("PharmaIntel is thinking..."):
            
            # Prepare history for the ConversationalRetrievalChain (needs tuple format)
            # The history structure is simplified here to avoid depth errors
            langchain_history = [(h['content'], h['content']) for h in st.session_state.chat_history if h['role'] != 'system']

            result = qa_chain({"question": query, "chat_history": langchain_history})
            answer = result["answer"]
            sources = result.get("source_documents", [])
            
            # Display assistant response
            with st.chat_message("assistant"):
                st.write(answer)
                
                # Display sources using the expander
                if sources:
                    # Collect sources and pass them to history for consistent display
                    source_list = []
                    with st.expander("📚 Sources Used"):
                        for d in sources[:4]:
                            src = d.metadata.get("source", d.metadata.get("site", "Local Data"))
                            url = d.metadata.get("source_url", d.metadata.get("url", "N/A"))
                            st.markdown(f"**{src}** — {url}")
                            st.caption(d.page_content[:300].replace('\n', ' ').strip() + "...")
                            source_list.append(d) # Keep a copy for saving to session state
                        
            # Add assistant response to history
            st.session_state.chat_history.append({"role": "assistant", "content": answer, "sources": sources})

    # Clear history button at the bottom of the main page
    st.markdown("---")
    if st.button("Clear Conversation History", help="Removes all chat history from this session."):
        st.session_state.chat_history = []
        st.rerun()

# Footer
st.sidebar.markdown("---")
st.sidebar.write("⚠️ **Disclaimer:** This RAG application is for **educational/informational use** only and is **not medical advice**.")
