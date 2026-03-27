## PDF Loader
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("ModuleofAppliedGeomorphologyfinal.pdf")
docs = loader.load()
docs
## Text Splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(docs)
documents

## Vector Embedding and Vector Store (HuggingFace MPNet)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Use MPNet embedding model from Hugging Face
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Build FAISS index
db = FAISS.from_documents(documents, embeddings)
db

## Query
query = "Who are the authors of 'ModuleofAppliedGeomorphologyfinal'?"
retrieved_results = db.similarity_search(query)
print("\nTop Result:\n", retrieved_results[0].page_content)

from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline

pipe = pipeline("text2text-generation", model="google/flan-t5-base", max_length=512)
llm = HuggingFacePipeline(pipeline=pipe)
llm

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context. 
Think step by step before providing a detailed answer. 
<context>
{context}
</context>
Question: {input}
""")

from langchain.chains.combine_documents import create_stuff_documents_chain

document_chain = create_stuff_documents_chain(llm, prompt)

retriever = db.as_retriever()
retriever

from langchain.chains import create_retrieval_chain

retrieval_chain = create_retrieval_chain(retriever, document_chain)
response = retrieval_chain.invoke({"input": "Glaciated areas and groundwater"})
response['answer']

# Final function to expose for Streamlit
def run_query(question: str):
    response = retrieval_chain.invoke({"input": question})
    return response['answer']

