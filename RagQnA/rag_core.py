import os
import hashlib
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import chromadb


# ============================================================
# PDF INGESTION
# ============================================================

def process_all_pdfs(pdf_directory: str):
    documents = []
    pdf_dir = Path(pdf_directory)

    pdf_files = list(pdf_dir.glob("**/*.pdf"))

    for pdf_file in pdf_files:
        loader = PyPDFLoader(str(pdf_file))
        pages = loader.load()

        for page in pages:
            page.metadata.update({
                "source_file": pdf_file.name,
                "page_number": page.metadata.get("page"),
                "file_type": "pdf"
            })

        documents.extend(pages)

    return documents


# ============================================================
# CHUNKING
# ============================================================

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_documents(documents)


# ============================================================
# EMBEDDINGS
# ============================================================

class EmbeddingManager:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, show_progress_bar=False)


# ============================================================
# VECTOR STORE (DEDUP SAFE)
# ============================================================

class VectorStore:
    def __init__(self, persist_directory="./vector_store"):
        os.makedirs(persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection("pdf_documents")

    def add_documents(self, docs, embeddings):
        existing_ids = set(self.collection.get()["ids"])

        ids, texts, metadatas, embeds = [], [], [], []

        for doc, emb in zip(docs, embeddings):
            content_hash = hashlib.md5(
                doc.page_content.strip().encode("utf-8")
            ).hexdigest()

            if content_hash in existing_ids:
                continue

            ids.append(content_hash)
            texts.append(doc.page_content)
            metadatas.append(doc.metadata)
            embeds.append(emb.tolist())

        if ids:
            self.collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas,
                embeddings=embeds
            )


# ============================================================
# RETRIEVER (SEMANTIC DEDUP)
# ============================================================

class RAGRetriever:
    def __init__(self, vector_store, embedder):
        self.vector_store = vector_store
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int = 5):
        query_emb = self.embedder.embed([query])[0]

        results = self.vector_store.collection.query(
            query_embeddings=[query_emb.tolist()],
            n_results=15
        )

        final_results = []
        final_embeds = []

        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            doc_emb = self.embedder.embed([doc])[0]

            is_dup = False
            for e in final_embeds:
                if cosine_similarity([doc_emb], [e])[0][0] > 0.95:
                    is_dup = True
                    break

            if not is_dup:
                final_results.append({
                    "content": doc,
                    "metadata": meta,
                    "distance": dist
                })
                final_embeds.append(doc_emb)

            if len(final_results) == top_k:
                break

        return final_results
