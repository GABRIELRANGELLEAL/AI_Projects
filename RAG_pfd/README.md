# RAG PDF Q&A System

A system that lets users upload PDF documents and ask questions about their contents. It extracts text, indexes it with embeddings, and uses an LLM to generate answers grounded in the documents.

## Tech Stack

- **Backend:** FastAPI + Python 3.12
- **Frontend:** React 19 + Tailwind CSS 4 + Vite 8
- **PDF Extraction:** PyMuPDF
- **Embeddings:** OpenAI `text-embedding-3-small`
- **Vector Store:** ChromaDB (cosine similarity)
- **Retrieval:** Hybrid search — semantic (ChromaDB) + keyword (BM25) fused via Reciprocal Rank Fusion
- **LLM:** OpenAI `gpt-4o-mini` (primary), Anthropic `claude-sonnet-4-6` (fallback)
- **Infrastructure:** Docker + Docker Compose

## Prerequisites

- Docker and Docker Compose
- An OpenAI API key

## Quick Start

1. Clone the repository and enter the project directory.

2. Create a `.env` file (optional — API keys are provided via the UI):
   ```
   LOG_LEVEL=INFO
   ```

3. Build and run:
   ```bash
   docker compose up --build
   ```

4. Open [http://localhost:8000](http://localhost:8000) in your browser.

5. Enter your OpenAI API key on the setup screen, upload a PDF, and start asking questions.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/documents` | Upload and index PDF files |
| `POST` | `/question` | Ask a question about indexed documents |
| `POST` | `/validate-keys` | Validate API keys |
| `GET`  | `/stats` | Document and chunk statistics |
| `GET`  | `/health` | Healthcheck |

API keys are passed via headers: `X-OpenAI-Key` (required), `X-Anthropic-Key` (optional).

Full API details and architecture documentation are in [ARCHITECTURE.md](ARCHITECTURE.md).

## How It Works

1. **Upload:** PDFs are extracted (PyMuPDF), split into semantic chunks with overlap, embedded, and stored in ChromaDB.
2. **Question:** The question is translated to both PT and EN for bilingual retrieval. Hybrid search (semantic + BM25) finds the most relevant chunks. The LLM answers using only those excerpts, citing its sources.

## Project Structure

```
backend/
  app/
    main.py              # FastAPI entry point
    config.py            # Settings via env vars
    api/routes/          # Endpoint handlers
    core/                # Extraction, chunking, embedding, retrieval, LLM
    store/               # ChromaDB wrapper
frontend/
  src/
    pages/SetupPage.jsx  # API key setup
    pages/MainPage.jsx   # Chat interface + document sidebar
Dockerfile               # Multi-stage build (Node + Python)
docker-compose.yml       # Single-command setup
```
