# RAG PDF Q&A System — Technical Specification v2

Accurate specification reflecting the current implementation as of 2026-06-24.

---

## Context

A system that allows users to upload PDF documents and ask questions about their contents. It extracts information from documents, stores it for efficient retrieval via embeddings, and uses an LLM to answer questions accurately.

This is a technical challenge for Tractian, evaluated on: **Functionality**, **Retrieval**, **LLM Use**, **Code Quality**, **API Design**, and **Developer UX**.

---

## Technology Stack

| Component | Technology | Version | Justification |
|---|---|---|---|
| Web Framework | FastAPI | 0.115.6 | Native async, Pydantic typing, automatic Swagger docs, supports multipart/form-data and JSON |
| PDF Extraction | PyMuPDF (fitz) | 1.25.5 | C-based (fast), preserves reading order, per-page metadata |
| Chunking | Custom implementation | — | Semantic chunking: respects paragraph/section boundaries, with fallback to size + overlap |
| Embeddings | OpenAI `text-embedding-3-small` | openai 1.59.6 | 1536-dim vectors, async client with exponential-backoff retry, batching |
| Vector Store | ChromaDB | 0.6.3 | In-process, persists to disk, stores metadata alongside vectors, cosine similarity |
| LLM (primary) | OpenAI `gpt-4o-mini` | — | Low-latency, cost-effective for RAG answers |
| LLM (fallback) | Anthropic `claude-sonnet-4-6` | anthropic 0.43.0 | Automatic fallback on primary failure |
| Frontend | React 19 + Tailwind CSS 4 + Vite 8 | — | Served as static files by FastAPI |
| Logging | structlog | 24.4.0 | Structured log output |
| Settings | pydantic-settings | 2.7.0 | Environment-based configuration |
| Infrastructure | Docker + Docker Compose | — | Single-command setup, multi-stage build |

---

## Architecture

```
project-root/
├── backend/
│   ├── app/
│   │   ├── main.py                  ← FastAPI entry point, mounts routes and serves static frontend
│   │   ├── config.py                ← Settings via pydantic-settings + env vars
│   │   ├── api/
│   │   │   ├── routes/
│   │   │   │   ├── documents.py     ← POST /documents (PDF upload and indexing)
│   │   │   │   ├── question.py      ← POST /question (question and answer)
│   │   │   │   ├── keys.py          ← POST /validate-keys (API key validation)
│   │   │   │   └── stats.py         ← GET /stats (aggregated metrics)
│   │   │   └── dependencies.py      ← Dependency injection (providers, store, API keys from headers)
│   │   ├── core/
│   │   │   ├── extraction.py        ← PDF text extraction + language detection + header/footer removal
│   │   │   ├── chunking.py          ← Semantic chunking with overlap
│   │   │   ├── embedding.py         ← OpenAI embedding provider with retry and batching
│   │   │   ├── retrieval.py         ← Hybrid retrieval: semantic (ChromaDB) + keyword (BM25) with RRF fusion
│   │   │   ├── translation.py       ← Question translation for bilingual retrieval (gpt-4o-mini)
│   │   │   └── llm.py               ← OpenAI + Anthropic providers, fallback logic, prompt construction, citation parsing
│   │   ├── store/
│   │   │   └── vector_store.py      ← ChromaDB wrapper (indexing, search, listing, stats, delete)
│   │   └── tests/
│   │       ├── test_extraction.ipynb
│   │       ├── test_chunking.ipynb
│   │       ├── test_embedding.ipynb
│   │       └── test_llm.ipynb
│   ├── static/                      ← Built frontend files (copied by Docker)
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx                  ← State-based routing between SetupPage and MainPage
│   │   ├── main.jsx                 ← React entry point
│   │   ├── index.css                ← Tailwind imports
│   │   └── pages/
│   │       ├── SetupPage.jsx        ← API key configuration screen (OpenAI only)
│   │       └── MainPage.jsx         ← Main screen: sidebar (upload + doc list) + chat area
│   ├── package.json
│   └── index.html
├── pdf_documents/                   ← Sample PDFs for testing (WEG motor manuals)
├── Dockerfile                       ← Multi-stage: node build → python runtime
├── docker-compose.yml
├── DOCKER.md                        ← Docker setup instructions
├── .dockerignore
└── .env                             ← Environment variables (not committed)
```

---

## Configuration (config.py)

All settings are loaded from environment variables via pydantic-settings, with these defaults:

| Setting | Default | Description |
|---|---|---|
| `chunk_size` | 512 | Target maximum tokens per chunk |
| `chunk_overlap` | 50 | Overlap tokens between consecutive chunks |
| `top_k` | 5 | Number of final results returned to LLM |
| `llm_model_openai` | `gpt-4o-mini` | OpenAI model for answer generation |
| `llm_model_anthropic` | `claude-sonnet-4-6` | Anthropic model for fallback |
| `embedding_model` | `text-embedding-3-small` | OpenAI embedding model |
| `chroma_persist_dir` | `./data/chromadb` | ChromaDB persistence directory |
| `log_level` | `INFO` | Logging level |

---

## API Specification

### POST /validate-keys

Validates API keys by making minimal test calls (list models for OpenAI, send a 1-token message for Anthropic).

**Request:**
```json
{
  "openai_key": "sk-...",
  "anthropic_key": "sk-ant-..."
}
```
Both fields are optional, but at least one must be provided.

**Response (success):**
```json
{
  "openai": { "valid": true, "message": "Conectado com sucesso" },
  "anthropic": { "valid": true, "message": "Conectado com sucesso" }
}
```

**Response (failure):**
```json
{
  "openai": { "valid": false, "message": "API key inválida ou sem permissão" },
  "anthropic": null
}
```

---

### POST /documents

Upload and index PDFs. Requires `X-OpenAI-Key` header for embedding generation.

**Request:**
- Content-Type: `multipart/form-data`
- Body: one or more PDFs under the field `files`
- Headers: `X-OpenAI-Key: sk-...`

**Response:**
```json
{
  "message": "Documents processed successfully",
  "documents_indexed": 2,
  "total_chunks": 128,
  "details": [
    {
      "filename": "motor_manual.pdf",
      "pages": 15,
      "chunks": 72,
      "processing_time_ms": 3400,
      "detected_language": "pt"
    }
  ]
}
```

**Error cases:**
- 400 if no files provided
- 400 if file is not a PDF
- 401 if `X-OpenAI-Key` header is missing

---

### POST /question

Ask a question about indexed documents. Uses bilingual retrieval (translates the question to the other language for broader recall).

**Request:**
```json
{
  "question": "What is the power consumption of the motor?"
}
```
- Headers: `X-OpenAI-Key: sk-...` (required), `X-Anthropic-Key: sk-ant-...` (optional, enables fallback)

**Response:**
```json
{
  "answer": "The motor's power consumption is 2.3 kW.",
  "references": [
    {
      "text": "the motor xxx has requires 2.3kw to operate at a 60hz line frequency",
      "document": "motor_manual.pdf",
      "page": 3,
      "similarity_score": 0.92
    }
  ],
  "metadata": {
    "provider_used": "openai",
    "model": "gpt-4o-mini",
    "retrieval_time_ms": 45,
    "llm_time_ms": 1230,
    "total_time_ms": 1320,
    "confidence": "high",
    "query_language_used": "bilingual"
  }
}
```

**Error cases:**
- 400 if no documents indexed yet
- 401 if no API key header provided

---

### GET /stats

Returns aggregated metrics from the vector store.

**Response:**
```json
{
  "documents_indexed": 5,
  "total_chunks": 312,
  "chunks_per_document": {
    "motor_manual.pdf": 72,
    "specs.pdf": 56
  },
  "documents": ["motor_manual.pdf", "specs.pdf"]
}
```

---

### GET /health

Healthcheck for Docker.

**Response:**
```json
{
  "status": "healthy"
}
```

---

## Module Implementation Details

### extraction.py

**Input:** Raw PDF bytes + filename.
**Output:** `ExtractionResult` with per-page text, metadata, and detected language.

**Pipeline:**
1. Open PDF with PyMuPDF (`fitz.open`) from raw bytes
2. Extract raw text from every page (`page.get_text("text")`)
3. If no extractable text across all pages → return result with `has_extractable_text=False` and a warning about scanned PDFs
4. **Header/footer detection:** count line occurrences across pages; lines appearing on >60% of pages (and >2 absolute occurrences) are flagged as headers/footers. Skipped for documents with <3 pages
5. **Text cleaning:** remove flagged header/footer lines, collapse runs of spaces/tabs, collapse 3+ consecutive newlines to paragraph breaks
6. **Language detection:** heuristic based on stop-word frequency (PT vs EN word lists) + diacritic boost (ã, õ, ç, etc.) from a 500-word sample. Returns `"pt"` or `"en"`

**Key data classes:**
- `PageContent(page_number, text, filename)` — 1-based page index
- `ExtractionResult(pages, filename, total_pages, has_extractable_text, warning, detected_language)`

---

### chunking.py

**Input:** List of `PageContent` objects from extraction.
**Output:** Flat list of `Chunk` objects with monotonically increasing `chunk_index`.

**Algorithm:**
1. **Recursive splitting** with separator hierarchy:
   - `\n\n` (paragraphs/sections)
   - `\n` (line breaks)
   - `. ` (sentences)
   - ` ` (words)
   - Character-level (last resort, splits at `chunk_size * 4` chars)
2. **Small segment merging:** segments below `MIN_CHUNK_TOKENS=50` are concatenated with the next segment
3. **Overlap assembly:** segments accumulate into chunks; when a chunk would exceed `CHUNK_SIZE=512` tokens, it's emitted and the tail segments fitting within `CHUNK_OVERLAP=50` tokens carry over to the next chunk
4. Each page is processed independently (preserves correct `page_number` per chunk)

**Token estimation:** `len(text) // 3.5` (heuristic optimized for mixed PT/EN text).

**Data class:** `Chunk(text, filename, page_number, chunk_index)`

---

### embedding.py

**Provider:** `OpenAIEmbeddingProvider` only (local sentence-transformers provider is commented out).

**Behavior:**
- Uses `text-embedding-3-small` → 1536-dimensional vectors
- Async client with 30s timeout
- Exponential-backoff retry (up to 3 attempts) on `RateLimitError`, `APITimeoutError`, `APIConnectionError`
- Input texts truncated to ~6150 words before embedding (headroom below the 8191-token hard limit)
- Empty/whitespace texts replaced with zero vectors in output

**Batching:** `embed_chunks_in_batches()` sends texts in batches of 100 to stay within per-request token limits.

**Factory:** `get_embedding_provider(openai_key)` → always returns `OpenAIEmbeddingProvider`.

---

### vector_store.py

ChromaDB wrapper with cosine similarity (`hnsw:space: cosine`).

**Methods:**
| Method | Description |
|---|---|
| `add_documents(chunks, embeddings)` | Upserts chunks with metadata (filename, page_number, chunk_index). IDs are `{filename}::{chunk_index}` |
| `search(query_embedding, top_k)` | Cosine similarity search. Returns `SearchResult` with `score = 1 - distance` clamped to [0, 1] |
| `get_all_chunks()` | Returns all stored chunks as dicts (used for BM25 indexing) |
| `list_documents()` | Returns sorted list of unique filenames |
| `get_stats()` | Returns `documents_indexed`, `total_chunks`, `chunks_per_document`, `documents` |
| `delete_document(filename)` | Deletes all chunks for a specific document |
| `reset()` | Drops and recreates the collection |

**Embedding model validation:** on init, checks that the collection's stored `embedding_model` metadata matches the requested model. Raises `ValueError` on mismatch to prevent dimension incompatibility.

**Persistence:** directory configurable, default `/app/data/chromadb`. Uses `PersistentClient` with telemetry disabled.

---

### retrieval.py — Hybrid Search with RRF

Combines two strategies fused via Reciprocal Rank Fusion:

**1. Semantic search:** generates a query embedding → queries ChromaDB for top `search_pool=50` candidates by cosine similarity.

**2. Keyword search (BM25):** in-memory BM25-Okapi index (`k1=1.5`, `b=0.75`) built from scratch. Bilingual stop-word removal (EN + PT). Tokenization via regex `[a-zA-Z0-9À-ÿ]+`, lowercased, words with length >1.

**3. RRF fusion:**
```
score(doc) = β / (k + rank_semantic) + (1 − β) / (k + rank_keyword)
```
Defaults: `β=0.70` (semantic weight), `k=60` (standard RRF smoothing constant).

**4. Confidence classification** (based on top cosine similarity, not RRF score):
- `high`: > 0.80
- `medium`: >= 0.60
- `low`: < 0.60

**5. Optional metadata filter:** restrict results to specific filenames.

**Bilingual retrieval** (`retrieve_bilingual`):
- Runs `retrieve()` for the original question
- If a translation is available, runs `retrieve()` again for the translated question
- Merges by chunk ID keeping the highest RRF score, re-sorts, trims to `top_k`

**Data classes:**
- `RetrievalResult(text, document, page, chunk_index, rrf_score, semantic_score, keyword_score, semantic_rank, keyword_rank)`
- `RetrievalMetrics(embedding_time_ms, semantic_search_time_ms, keyword_search_time_ms, fusion_time_ms, total_time_ms, top_rrf_scores, confidence, semantic_pool_size, keyword_pool_size, query_language_used)`
- `RetrievalResponse(results, metrics)`

---

### translation.py

Translates questions between PT ↔ EN using `gpt-4o-mini` (temperature 0, max 500 tokens, 10s timeout). Returns the original question on failure (graceful degradation). Called by the `/question` endpoint to enable bilingual retrieval.

---

### llm.py

**Providers:**
- `OpenAILLMProvider`: async ChatCompletion API, `gpt-4o-mini`, temperature 0.1
- `AnthropicLLMProvider`: async Messages API, `claude-sonnet-4-6`, temperature 0.1
- `FallbackLLMProvider`: wraps primary + fallback; on `LLMError` from primary, retries with fallback and logs the switch

Both providers return `tuple[str, int | None]` — raw answer text + total tokens.

**Factory:** `get_llm_provider(model, openai_key, anthropic_key)` infers provider from model name prefix (`gpt` → OpenAI, `claude` → Anthropic).

**Dependency injection** (`dependencies.py`): if both keys are provided → `FallbackLLMProvider(OpenAI primary, Anthropic fallback)`. If only one key → single provider. No keys → 401.

**System prompt:**
```
You are an assistant that answers questions based exclusively on the document excerpts provided.

Rules:
- Answer ONLY with information present in the provided excerpts
- At the end of your response, list the numbers of the excerpts you used in the format: [USED: 1, 3, 5]
- If the information is not in the excerpts, explicitly state that you did not find sufficient information
- Be direct and precise
- Respond in the same language as the question
```

**Citation parsing:** extracts `[USED: 1, 3, 5]` from the end of the LLM response via regex, converts 1-based indices to 0-based.

**Orchestration (`answer_question`):** builds prompt → calls provider → parses citations → returns `LLMResponse(answer, referenced_chunk_indices, provider_used, model, tokens_used, latency_ms)`.

---

## Frontend

### Screen 1: SetupPage

Centered card with:
- Title "RAG PDF Q&A" + subtitle
- Single OpenAI API key input (password field) with "Test" button
- Visual feedback: loading indicator, green "Valid key" text, or red error message
- "Get Started" button disabled until key is validated
- Key stored in React state only (never persisted)

Note: Anthropic key input is not present in the current UI. The Anthropic key is not collected from the user. Fallback to Anthropic only works if `X-Anthropic-Key` header is provided externally.

### Screen 2: MainPage

**Left sidebar (w-72):**
- "Change API key" link back to setup
- File upload area: click-to-select (accepts `.pdf`, multiple files)
- Upload progress: "Processing..." label while uploading
- Document list: each document shows filename, page count, chunk count

**Main area (chat):**
- Empty states: "Upload a PDF to get started" / "Ask a question about your documents"
- User messages: right-aligned blue bubbles
- Assistant messages: white card with:
  - Answer text (pre-wrap)
  - Metadata row: confidence (color-coded green/yellow/red), model name, response time
  - Collapsible references section with text excerpt, document name, page number, similarity percentage
- Error messages: red-bordered cards
- Loading state: pulse animation skeleton
- Input form: text field + "Send" button at bottom, auto-scrolls to latest message

**API key passing:** headers are built per-request from React state (`X-OpenAI-Key`, `X-Anthropic-Key`).

---

## Complete Request Flow

### Document Upload
```
User drops PDF → Frontend POST /documents (multipart + X-OpenAI-Key header)
  → extract_pdf() — PyMuPDF text extraction, language detection, header/footer removal
  → chunk_pages() — semantic splitting with overlap
  → embed_chunks_in_batches() — OpenAI API in batches of 100
  → vector_store.add_documents() — upsert to ChromaDB
  ← Response with per-file stats (pages, chunks, time, language)
```

### Question Answering
```
User types question → Frontend POST /question (JSON + API key headers)
  → Detect question language (PT/EN heuristic)
  → Translate question to the other language (gpt-4o-mini)
  → retrieve_bilingual():
      → retrieve() on original question:
          → Embed question → ChromaDB cosine search (top 50)
          → BM25 keyword search over all chunks (top 50)
          → RRF fusion → top 5
      → retrieve() on translated question (same pipeline)
      → Merge both result sets by chunk ID, keep best RRF scores → top 5
  → answer_question():
      → Build prompt with numbered excerpts + question
      → Call LLM (OpenAI primary, Anthropic fallback)
      → Parse [USED: ...] citations
  → Build references from cited chunks
  ← Response with answer, references, metadata (confidence, provider, timings, language)
```

---

## Infrastructure

### Dockerfile (multi-stage)
1. **Stage 1 (frontend):** `node:20-slim` → `npm install` → `npm run build` → produces `dist/`
2. **Stage 2 (runtime):** `python:3.12-slim` → installs curl + pip deps → copies backend + built frontend to `static/` → creates non-root `appuser` → exposes 8000 → runs uvicorn

### docker-compose.yml
- Single service `api` with port 8000
- Named volume `chroma_data` at `/app/data` for persistence
- Healthcheck: `curl -f http://localhost:8000/health` every 30s
- Loads `.env` file

---

## Logging

Structured logging via `structlog` with `ConsoleRenderer`. Each pipeline stage logs with timing:

```
action=extract     | file=manual.pdf | pages=15 | extracted_pages=14 | language=pt
action=chunk       | file=manual.pdf | chunks=72
action=embed_batch | batch=1/1 | chunks=72
action=add_documents | chunks=72 | collection=documents
action=translate   | from=en | to=pt | original="What is..." | translated="Qual é..."
action=retrieve    | question="..." | strategy=hybrid_rrf | beta=0.70 | ...
action=llm_call    | provider=openai | model=gpt-4o-mini | tokens=580 | latency_ms=1230
```

---

## Testing

Current tests are Jupyter notebooks (not pytest):
- `test_extraction.ipynb` — PDF extraction validation
- `test_chunking.ipynb` — Chunking logic validation
- `test_embedding.ipynb` — Embedding generation tests
- `test_llm.ipynb` — LLM call tests

pytest and httpx are listed in `requirements.txt` but commented out.