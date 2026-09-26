"""
graph.py - LangGraph multi-agent RAG for UCSD MSDS advising.

Flow:
    question -> [router] --conditional--> courses_agent / program_agent /
                                          admissions_agent / progress_agent
                                          or out_of_scope -> END
             -> [grade_documents] --no relevant docs--> [rewrite_query] -> router (retry)
             -> [generate] -> [check_grounding] --not grounded--> [rewrite_query] (retry)
                                                --grounded-------> END
    After MAX_RETRIES failed attempts -> [refuse] -> END

Usage:
    python graph.py --build          # scrape pages + build FAISS index with metadata
    python graph.py "How many units do I need to graduate?"
"""

import sys
from typing import List, Literal, TypedDict

import requests
from bs4 import BeautifulSoup
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # older LangChain
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from embeddings import embeddings  # all-MiniLM-L6-v2 (existing file)
from llm import llm                # Ollama Mistral, temperature 0.2 (existing file)

INDEX_DIR = "faiss_index_v2"
MAX_RETRIES = 2
TOP_K = 4

# Each source page is tagged with a category so each specialist agent
# searches only its own slice of the knowledge base.
SOURCES = {
    "https://datascience.ucsd.edu/current-students/course-offerings/": "courses",
    "https://datascience.ucsd.edu/graduate/ms-program/": "program",
    "https://mds.ucsd.edu/program/index.html": "program",
    "https://datascience.ucsd.edu/graduate/graduate-admissions/": "admissions",
    "https://datascience.ucsd.edu/graduate/ms-program/progress-to-degree/": "progress",
}
CATEGORIES = ["courses", "program", "admissions", "progress"]


# --------------------------------------------------------------------------
# 1. Offline: build a FAISS index where every chunk keeps source + category
# --------------------------------------------------------------------------
def build_index() -> None:
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs: List[Document] = []
    for url, category in SOURCES.items():
        print(f"Scraping {url}")
        html = requests.get(url, timeout=30).text
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        for chunk in splitter.split_text(text):
            docs.append(Document(page_content=chunk,
                                 metadata={"source": url, "category": category}))
    FAISS.from_documents(docs, embeddings).save_local(INDEX_DIR)
    print(f"Saved {len(docs)} chunks to {INDEX_DIR}/")


_vectorstore = None


def get_vectorstore() -> FAISS:
    global _vectorstore
    if _vectorstore is None:
        # Safe here: we built this index ourselves (it is a local pickle).
        _vectorstore = FAISS.load_local(
            INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    return _vectorstore


# --------------------------------------------------------------------------
# 2. Shared graph state - every node reads it and returns a partial update
# --------------------------------------------------------------------------
class AdvisorState(TypedDict, total=False):
    question: str          # original user question (never changed)
    query: str             # search query (may be rewritten on retry)
    intent: str            # courses | program | admissions | progress | out_of_scope
    docs: List[Document]   # retrieved (then filtered) chunks
    answer: str
    grounded: bool
    retries: int


def ask_llm(prompt: str) -> str:
    out = llm.invoke(prompt)
    return getattr(out, "content", out).strip()  # works for LLM or chat model


def yes(text: str) -> bool:
    return text.strip().lower().startswith("yes")


# --------------------------------------------------------------------------
# 3. Nodes (agents)
# --------------------------------------------------------------------------
def router(state: AdvisorState) -> AdvisorState:
    """Router agent: classify the question's intent."""
    prompt = f"""You route questions for a UCSD MS Data Science advising assistant.
Pick exactly ONE label:
- courses: course offerings, electives, which classes are taught when
- program: degree requirements, units, core courses, program structure
- admissions: applying, deadlines, eligibility, application materials
- progress: progress to degree, milestones, time limits, academic standing
- out_of_scope: anything not about the UCSD MS Data Science program

Question: {state.get('query') or state['question']}
Answer with the label only."""
    raw = ask_llm(prompt).lower()
    intent = next((c for c in CATEGORIES + ["out_of_scope"] if c in raw), "program")
    return {"intent": intent, "query": state.get("query") or state["question"],
            "retries": state.get("retries", 0)}


def make_specialist(category: str):
    """Factory: one retrieval agent per category, searching only its pages."""
    def specialist(state: AdvisorState) -> AdvisorState:
        docs = get_vectorstore().similarity_search(
            state["query"], k=TOP_K, filter={"category": category}, fetch_k=40)
        return {"docs": docs}
    specialist.__name__ = f"{category}_agent"
    return specialist


def grade_documents(state: AdvisorState) -> AdvisorState:
    """Grader agent: drop retrieved chunks that don't help answer the question."""
    kept = []
    for d in state["docs"]:
        verdict = ask_llm(f"""Does this passage contain information useful for answering the question?
Question: {state['question']}
Passage: {d.page_content}
Answer yes or no.""")
        if yes(verdict):
            kept.append(d)
    return {"docs": kept}


def rewrite_query(state: AdvisorState) -> AdvisorState:
    """Rewriter agent: rephrase the search query and count the retry."""
    new_q = ask_llm(f"""Rewrite this student question as a better search query for UCSD
MS Data Science policy pages. Use official terms (units, core courses, electives,
progress to degree). Return only the query.
Question: {state['question']}""")
    return {"query": new_q, "retries": state.get("retries", 0) + 1}


def generate(state: AdvisorState) -> AdvisorState:
    """Answer agent: answer ONLY from retrieved context, with sources."""
    context = "\n\n".join(
        f"[{i + 1}] {d.page_content}" for i, d in enumerate(state["docs"]))
    answer = ask_llm(f"""You are a UCSD MS Data Science advising assistant.
Answer the question using ONLY the context below. If the context does not
contain the answer, say "I don't know based on the official pages."
Cite passages like [1], [2].

Context:
{context}

Question: {state['question']}
Answer:""")
    sources = sorted({d.metadata["source"] for d in state["docs"]})
    answer += "\n\nSources:\n" + "\n".join(f"- {s}" for s in sources)
    return {"answer": answer}


def check_grounding(state: AdvisorState) -> AdvisorState:
    """Hallucination checker: is every claim supported by the context?"""
    context = "\n\n".join(d.page_content for d in state["docs"])
    verdict = ask_llm(f"""Is every factual claim in the ANSWER supported by the CONTEXT?
CONTEXT:
{context}

ANSWER:
{state['answer']}
Answer yes or no.""")
    return {"grounded": yes(verdict)}


def out_of_scope(state: AdvisorState) -> AdvisorState:
    return {"answer": "I can only answer questions about the UCSD MS Data Science "
                      "program. For anything else, please contact an HDSI advisor."}


def refuse(state: AdvisorState) -> AdvisorState:
    return {"answer": "I couldn't find a reliable answer in the official UCSD pages. "
                      "Please check with an HDSI graduate advisor."}


# --------------------------------------------------------------------------
# 4. Conditional edges (routing functions read the state, return a node name)
# --------------------------------------------------------------------------
def route_by_intent(state: AdvisorState) -> str:
    return state["intent"] if state["intent"] in CATEGORIES else "out_of_scope"


def after_grading(state: AdvisorState) -> Literal["generate", "rewrite_query", "refuse"]:
    if state["docs"]:
        return "generate"
    return "rewrite_query" if state["retries"] < MAX_RETRIES else "refuse"


def after_grounding(state: AdvisorState) -> Literal["end", "rewrite_query", "refuse"]:
    if state["grounded"]:
        return "end"
    return "rewrite_query" if state["retries"] < MAX_RETRIES else "refuse"


# --------------------------------------------------------------------------
# 5. Build the graph
# --------------------------------------------------------------------------
def build_graph():
    g = StateGraph(AdvisorState)

    g.add_node("router", router)
    for c in CATEGORIES:
        g.add_node(c, make_specialist(c))
    g.add_node("grade_documents", grade_documents)
    g.add_node("rewrite_query", rewrite_query)
    g.add_node("generate", generate)
    g.add_node("check_grounding", check_grounding)
    g.add_node("out_of_scope", out_of_scope)
    g.add_node("refuse", refuse)

    g.add_edge(START, "router")
    g.add_conditional_edges(
        "router", route_by_intent,
        {**{c: c for c in CATEGORIES}, "out_of_scope": "out_of_scope"})
    for c in CATEGORIES:
        g.add_edge(c, "grade_documents")
    g.add_conditional_edges(
        "grade_documents", after_grading,
        {"generate": "generate", "rewrite_query": "rewrite_query", "refuse": "refuse"})
    g.add_edge("rewrite_query", "router")          # retry loop
    g.add_edge("generate", "check_grounding")
    g.add_conditional_edges(
        "check_grounding", after_grounding,
        {"end": END, "rewrite_query": "rewrite_query", "refuse": "refuse"})
    g.add_edge("out_of_scope", END)
    g.add_edge("refuse", END)

    return g.compile()


advisor_graph = build_graph()


def answer_question(question: str) -> dict:
    result = advisor_graph.invoke({"question": question, "retries": 0})
    return {"answer": result["answer"], "intent": result.get("intent"),
            "retries": result.get("retries", 0)}


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--build":
        build_index()
    else:
        q = " ".join(sys.argv[1:]) or "How many units do I need to graduate?"
        print(answer_question(q)["answer"])
