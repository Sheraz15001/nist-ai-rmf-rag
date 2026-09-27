import csv
from datetime import datetime
from pathlib import Path

from pypdf import PdfReader
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


PDF_PATH = next(Path("documents").glob("*.pdf"))
INDEX_DIR = Path("chroma_index")
READY_FILE = INDEX_DIR / "ready.txt"
EVALUATION_FILE = Path("evaluation.csv")

embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Load the saved index, or build it if it does not exist
if READY_FILE.exists():
    database = Chroma(
        collection_name="nist_ai_rmf",
        persist_directory=str(INDEX_DIR),
        embedding_function=embeddings,
    )
    print("Loaded saved Chroma index.")
else:
    reader = PdfReader(PDF_PATH)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": PDF_PATH.name,
                        "page": page_number,
                    },
                )
            )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(pages)

    print(f"Read {len(pages)} pages and created {len(chunks)} chunks.")
    print("Creating embeddings. This may take several minutes...")

    database = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="nist_ai_rmf",
        persist_directory=str(INDEX_DIR),
    )

    READY_FILE.write_text("Index complete", encoding="utf-8")
    print("Saved Chroma index.")

model = ChatOllama(model="llama3.2")

print("\nAsk questions about the PDF. Type 'exit' to stop.")

while True:
    question = input("\nYour question: ").strip()

    if question.lower() in {"exit", "quit"}:
        print("Goodbye!")
        break

    if not question:
        print("Please enter a question.")
        continue

    matches = database.similarity_search(question, k=5)
    retrieved_pages = sorted({doc.metadata["page"] for doc in matches})

    context = "\n\n".join(
        f"[PDF page {doc.metadata['page']}]\n{doc.page_content}"
        for doc in matches
    )

    response = model.invoke(
        "Use only the provided context. Answer the exact question asked "
        "in at most two sentences. If the question asks for names, list them. "
        "If the context does not contain the answer, say 'I don't know'.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )

    answer = str(response.content)

    print("\nAnswer:", answer)
    print("Retrieved PDF pages:", retrieved_pages)

    # Append this result to the evaluation CSV
    with EVALUATION_FILE.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                datetime.now().isoformat(timespec="seconds"),
                question,
                answer,
                "; ".join(map(str, retrieved_pages)),
                "",  # Grade manually: pass, partial, or fail
                "",  # Notes after checking the PDF
            ]
        )

    print("Saved to evaluation.csv")