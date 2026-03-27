import streamlit as st
import shutil
from rag_core import (
    process_all_pdfs,
    split_documents,
    EmbeddingManager,
    VectorStore,
    RAGRetriever
)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="📘 RAG PDF Chatbot",
    layout="wide"
)

# ============================================================
# CUSTOM CSS (Modern Chat UI)
# ============================================================

st.markdown("""
<style>
.chat-container {
    max-width: 900px;
    margin: auto;
}
.user-bubble {
   
    padding: 10px;
    border-radius: 12px;
    margin: 10px 0;
    text-align: left;
}
.bot-bubble {
    background-color: #000;
    padding: 14px;
    border-radius: 12px;
    margin: 10px 0;
}
.meta {
    font-size: 0.85em;
    color: #666;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================

st.title("📘 RAG PDF Chatbot")
st.caption("Ask questions and retrieve answers directly from your PDFs")

# ============================================================
# SIDEBAR CONTROLS
# ============================================================

st.sidebar.header("⚙️ Controls")

PDF_DIR = st.sidebar.text_input(
    "📂 PDF Directory",
    value=r"D:\Project\RagQnA\data"
)

TOP_K = st.sidebar.slider(
    "🔍 Number of Retrieved Chunks",
    min_value=1,
    max_value=10,
    value=5
)

if st.sidebar.button("🔄 Reset Vector Store"):
    shutil.rmtree("./vector_store", ignore_errors=True)
    st.sidebar.success("Vector store reset!")

# ============================================================
# LOAD PIPELINE (CACHED)
# ============================================================

@st.cache_resource
def load_pipeline(pdf_dir):
    docs = process_all_pdfs(pdf_dir)
    chunks = split_documents(docs)

    embedder = EmbeddingManager()
    embeddings = embedder.embed([c.page_content for c in chunks])

    vectorstore = VectorStore()
    vectorstore.add_documents(chunks, embeddings)

    retriever = RAGRetriever(vectorstore, embedder)
    return retriever

if st.sidebar.button("📥 Build / Load Knowledge Base"):
    with st.spinner("Indexing PDFs and building vector store..."):
        st.session_state["retriever"] = load_pipeline(PDF_DIR)
    st.sidebar.success("Knowledge base ready!")

# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

# ============================================================
# CHAT DISPLAY
# ============================================================

st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f"<div class='user-bubble'>{msg['content']}</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='bot-bubble'>{msg['content']}</div>",
            unsafe_allow_html=True
        )

st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# CHAT INPUT
# ============================================================

query = st.chat_input("Ask a question about your PDFs...")

if query:
    # Store user message
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    if "retriever" not in st.session_state:
        st.warning("Please build/load the knowledge base first.")
    else:
        with st.spinner("Searching relevant sections..."):
            results = st.session_state["retriever"].retrieve(query, TOP_K)

        # Build assistant response
        answer = "Here are the most relevant sections from the documents:\n\n"

        for i, r in enumerate(results, 1):
            answer += f"🔹 **Result {i} (Page {r['metadata'].get('page_number')})**\n"
            answer += r["content"][:600] + "\n\n"

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

        st.rerun()
