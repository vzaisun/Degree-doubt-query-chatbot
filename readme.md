# UCSD HDSI / MDS Chatbot

This project is a question-answering chatbot built to help UCSD HDSI / MDS master’s students understand their program requirements, courses, admissions, and progress-to-degree information.

The chatbot uses a Retrieval-Augmented Generation (RAG) approach and is built entirely using free and open-source tools. It runs locally and does not require any paid APIs or OpenAI keys.

---

## What this chatbot does

- Answers questions about UCSD HDSI / MDS programs
- Uses official UCSD public webpages as its knowledge source
- Retrieves relevant content before generating answers
- Runs completely on a local machine

---

## Data sources

The chatbot is built using content scraped from the following public UCSD websites:

- https://datascience.ucsd.edu/current-students/course-offerings/
- https://datascience.ucsd.edu/graduate/ms-program/
- https://datascience.ucsd.edu/graduate/graduate-admissions/
- https://datascience.ucsd.edu/graduate/ms-program/progress-to-degree/
- https://mds.ucsd.edu/program/index.html

---

## Tech stack and why it is used

- **Python**: Core implementation
- **Requests + BeautifulSoup**: Scraping public UCSD webpages
- **LangChain**: RAG pipeline and retrieval logic
- **SentenceTransformers**: Local text embeddings
- **FAISS**: Vector similarity search
- **Ollama**: Local large language model inference
- **FastAPI**: Backend API for chatbot queries
- **Uvicorn**: Server to run the FastAPI app

---

## Project structure

