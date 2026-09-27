# NIST AI RMF Question Answering

A local RAG application that answers questions using the 48-page NIST AI Risk Management Framework (AI RMF 1.0). It retrieves relevant PDF passages with Chroma and `nomic-embed-text`, then uses `llama3.2` to generate an answer. The project includes a command-line interface, a FastAPI endpoint, and a small web page.

## How it works

1. `rag.py` extracts text from the PDF, retaining PDF page numbers.
2. It splits the pages into overlapping chunks and embeds them with `nomic-embed-text`.
3. Chroma saves the searchable index in `chroma_index/`.
4. A question retrieves five relevant chunks.
5. `llama3.2` answers using those chunks. The app displays the retrieved PDF pages.

The displayed pages are retrieval results; generated claims should still be checked against the PDF.

## Run locally

Requires Python, [Ollama](https://ollama.com/download), and the two Ollama models.

On Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
ollama pull llama3.2
ollama pull nomic-embed-text