# Research Content Agent

![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/-FastAPI-009688?logo=fastapi&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/-PostgreSQL-4169E1?logo=postgresql&logoColor=white)
![OpenAI](https://img.shields.io/badge/-OpenAI-412991?logo=openai&logoColor=white)
![LLM](https://img.shields.io/badge/-LLM%20Agents-6366F1)
![arXiv](https://img.shields.io/badge/-arXiv-B31B1B?logo=arxiv&logoColor=white)


> **AI-powered research pipeline** that automatically searches, analyzes, and ranks scientific papers using LLM agents.

---

<div align="center">

**AI Research Pipeline • FastAPI Backend • Streaming UI • LLM Ranking**

</div>

## 🧠 How It Works

```
User Query
    │
    ▼
┌─────────────────────┐
│   Keyword Agent     │  Extracts academic search phrases (GPT-4o-mini)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   arXiv Search      │  Fetches papers via arXiv Atom API
└────────┬────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│              Research Ranking Agent             │
│                                                 │
│  score_similarity  — token overlap with query   │
│  score_recency     — publication date decay     │
│  score_quality     — metadata completeness      │
│  score_subjective  — LLM relevance scoring      │
│                                                 │
│  rank_score = weighted average of all four      │
└─────────────────────────────────────────────────┘
         │
         ▼
   Ranked JSON results → Frontend
```

---

## 🚀 Features

- **Natural language input** — just describe your research topic in plain English
- **Keyword extraction agent** — uses GPT-4o-mini to generate optimal arXiv search phrases
- **arXiv integration** — queries the official Atom API with retry logic and rate-limit handling
- **Multi-criteria ranking** — combines similarity, recency, quality, and LLM-based subjective relevance into a single `rank_score`
- **PDF text extraction** — optionally downloads and reads the first pages of each paper (PyMuPDF + pdfminer fallback)
- **Real-time streaming** — `/research/stream` endpoint uses SSE to push step-by-step progress to the frontend
- **Dockerized** — single container setup, ready to run anywhere

---

## 📁 Project Structure

```
.
├── main.py                  # FastAPI app — GET /, POST /research, POST /research/stream
├── src/
│   ├── agents.py            # key_word() and research_agent() — the AI pipeline
│   └── research_tools.py    # arXiv, Tavily, and Wikipedia search tools + PDF utilities
├── templates/
│   └── index_v3.html        # Frontend (Jinja2 template)
├── static/                  # Static assets
├── requirements.txt
├── Dockerfile
└── docker/
    └── entrypoint.sh        # Container startup script
```

---

## ⚙️ Setup

### Prerequisites

- Docker (recommended), or Python 3.11+
- An OpenAI API key

### Environment Variables

Create a `.env` file at the project root:

```env
OPENAI_API_KEY=sk-...

# Optional — only needed if using Tavily web search
TAVILY_API_KEY=tvly-...
```

### Running with Docker

```bash
docker build -t research-agent .
docker run -p 8000:8000 --env-file .env research-agent
```

### Running locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Then open [http://localhost:8000](http://localhost:8000).

---

## 📡 API Reference

### `POST /research`
Synchronous endpoint. Returns the full ranked result in a single response.

**Request body:**
```json
{
  "sentence": "transformer models for time series forecasting",
  "max_results": 10,
  "fetch_pdf": false,
  "add_subjective": true
}
```

**Response:**
```json
{
  "sentence": "...",
  "keywords": { "search_phrases": [...], "url_to_query": "..." },
  "papers_raw": [...],
  "papers_ranked": [
    {
      "title": "...",
      "authors": [...],
      "published": "2024-01-15",
      "url": "https://arxiv.org/abs/...",
      "summary": "...",
      "score_similarity": 8.5,
      "score_recency": 9.1,
      "score_quality": 9.0,
      "score_subjective": 7.8,
      "rank_score": 8.6,
      "rationale": "..."
    }
  ]
}
```

### `POST /research/stream`
Streaming endpoint (SSE). Emits progress events as each pipeline step completes.

**SSE Events:**

| Event | Payload |
|-------|---------|
| `task` | `{ task_id, steps }` |
| `progress` | `{ task_id, step_index, step_name, message }` |
| `result` | Full result payload (same as `/research`) |
| `error` | `{ task_id, message }` |

---

## 🧩 Agent Details

### `key_word(prompt)`
Calls GPT-4o-mini with `temperature=0` to extract concise academic search phrases from the user's natural language query. Returns a list of phrases and a pre-built arXiv query string.

### `research_agent(query, papers)`
Scores each paper across four dimensions:

| Score | Method | Weight |
|-------|--------|--------|
| `score_similarity` | Token overlap between query and title+abstract | 2× |
| `score_recency` | Years since publication (linear decay, max 10y) | 1× |
| `score_quality` | Metadata completeness heuristic | 1× |
| `score_subjective` | GPT-4o-mini reads abstract + PDF pages and scores relevance | 2× |

Final `rank_score` = weighted mean, clamped to [0, 10].

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI + Uvicorn |
| AI | OpenAI GPT-4o-mini via `aisuite` |
| Data source | arXiv Atom API |
| PDF parsing | PyMuPDF (primary) + pdfminer.six (fallback) |
| Frontend | HTML + Jinja2 (SSE-powered) |
| Container | Docker |
