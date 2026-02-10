from fastapi import FastAPI
from pydantic import BaseModel
from rag import qa_chain

app = FastAPI(title="UCSD HDSI Chatbot")

class Question(BaseModel):
    question: str

@app.post("/chat")
def chat(q: Question):
    answer = qa_chain.run(q.question)
    return {"answer": answer}
