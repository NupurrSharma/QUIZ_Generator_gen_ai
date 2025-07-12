import os
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

def ingest_documents(data_path="data"):
    documents = []

    for file in os.listdir(data_path):
        if file.endswith(".pdf"):
            reader = PdfReader(os.path.join(data_path, file))
            raw_text = ""
            for page in reader.pages:
                raw_text += page.extract_text()
            documents.append(raw_text)

    # Chunking
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    all_chunks = []
    for doc in documents:
        all_chunks += splitter.create_documents([doc])

    # Create embeddings and store FAISS index
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(all_chunks, embeddings)
    vectorstore.save_local("vector_store/quiz_index")

    return all_chunks
