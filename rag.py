from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from embeddings import embeddings
from llm import llm

def load_qa_chain():
    vectorstore = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )

    return qa_chain

qa_chain = load_qa_chain()
