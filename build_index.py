from langchain_community.vectorstores import FAISS
from chunker import load_and_chunk
from embeddings import embeddings

def build_faiss_index():
    chunks = load_and_chunk()
    vectorstore = FAISS.from_texts(chunks, embeddings)
    vectorstore.save_local("faiss_index")
    print("FAISS index saved")

if __name__ == "__main__":
    build_faiss_index()
