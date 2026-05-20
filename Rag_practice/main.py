import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

app = FastAPI(title="Mini RAG Pipeline")

UPLOAD_FOLDER = "pdfs"
FAISS_FOLDER  = "faiss_db"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FAISS_FOLDER,  exist_ok=True)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

llm = OllamaLLM(model="mistral")

PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful assistant. Use only the context below to answer the question.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {question}

Answer:"""
)


@app.post("/upload")
async def upload(files: list[UploadFile] = File(...)):
    all_docs = []

    for file in files:
        path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(path, "wb") as f:
            f.write(await file.read())
        loader = PyPDFLoader(path)
        all_docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(all_docs)

    vector_db = FAISS.from_documents(chunks, embeddings)
    vector_db.save_local(FAISS_FOLDER)

    return {
        "message": "Documents ingested successfully",
        "files_uploaded": len(files),
        "total_chunks": len(chunks)
    }


class Question(BaseModel):
    question: str
    top_k: int = 3


@app.post("/ask")
async def ask(req: Question):

    if not os.path.exists(os.path.join(FAISS_FOLDER, "index.faiss")):
        raise HTTPException(
            status_code=400,
            detail="No documents uploaded yet. Please call /upload first."
        )

    # load faiss index
    vector_db = FAISS.load_local(
        FAISS_FOLDER,
        embeddings,
        allow_dangerous_deserialization=True
    )

    # retrieve top k relevant chunks
    relevant_chunks = vector_db.similarity_search(req.question, k=req.top_k)

    # join chunks into single context string
    context = "\n\n".join([doc.page_content for doc in relevant_chunks])

    # build prompt manually — no chain needed
    prompt = PROMPT.format(context=context, question=req.question)

    # send to LLM and get answer
    answer = llm.invoke(prompt)

    return {
        "question": req.question,
        "answer": answer
    }


@app.get("/health")
def health():
    return {"status": "ok"}
