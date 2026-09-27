from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings


if not Path("chroma_index/ready.txt").exists():
    raise RuntimeError("Run rag.py once to build the PDF index first.")

app = FastAPI(title="NIST AI RMF RAG API")

embeddings = OllamaEmbeddings(model="nomic-embed-text")
database = Chroma(
    collection_name="nist_ai_rmf",
    persist_directory="chroma_index",
    embedding_function=embeddings,
)
model = ChatOllama(model="llama3.2")


class Question(BaseModel):
    question: str


class Answer(BaseModel):
    answer: str
    retrieved_pdf_pages: list[int]


@app.get("/")
def home():
    return FileResponse("index.html")


@app.post("/ask", response_model=Answer)
def ask(request: Question):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    matches = database.similarity_search(question, k=5)

    context = "\n\n".join(
        f"[PDF page {doc.metadata['page']}]\n{doc.page_content}"
        for doc in matches
    )

    response = model.invoke(
        "Use only the provided context. Answer the exact question asked "
        "in at most two sentences. If the answer is absent, say 'I don't know'.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )

    return Answer(
        answer=str(response.content),
        retrieved_pdf_pages=sorted(
            {doc.metadata["page"] for doc in matches}
        ),
    )