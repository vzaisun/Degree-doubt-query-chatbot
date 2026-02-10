from langchain_community.llms import Ollama

llm = Ollama(
    model="mistral",   # or "llama3:8b"
    temperature=0.2
)
