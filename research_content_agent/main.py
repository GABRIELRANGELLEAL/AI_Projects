"""
main.py — Fixed workflow research agent (FastAPI + Postgres)

Key differences vs your current planner-based version:
- NO planner_agent() / executor_agent_step() dynamic planning.
- Uses a FIXED pipeline:
    1) extracting informations (validate + store config)
    2) research agent (collect research text)
    3) writer agent (draft report)
    4) editor agent (final polish)
- UI controls:
    - sources: checkboxes (tavily / wikipedia / arxiv)
    - n_articles: dropdown (5 / 10 / 15)
    - prompt: textarea

Important notes:
- This keeps Postgres storage for tasks, and stores pipeline state in Task.result JSON.
- Steps are persisted to DB (so you can refresh the page and still see state).
- Background processing runs in a thread (dev-friendly, not a production queue).
"""

import os
import uuid
import json
import threading
from datetime import datetime
from typing import List, Literal

# ---------------------------
# FastAPI (web framework)
# ---------------------------
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------
# Pydantic (request validation)
# ---------------------------
from pydantic import BaseModel, Field

# ---------------------------
# SQLAlchemy (Postgres ORM)
# ---------------------------
from sqlalchemy import create_engine, Column, Text, DateTime, String
from sqlalchemy.orm import sessionmaker, declarative_base

# ---------------------------
# .env loader
# ---------------------------
from dotenv import load_dotenv

# ---------------------------
# HTML escaping (avoid UI injection)
# ---------------------------
import html

# ---------------------------
# Your existing agents (already in your repo)
# - research_agent: calls tools internally (tavily/arxiv/wikipedia)
# - writer_agent: writes the draft
# - editor_agent: polishes it
# ---------------------------
from src.agents import writer_agent, editor_agent, research_agent  # :contentReference[oaicite:1]{index=1}


# ============================================================
# 1) Environment / DB configuration
# ============================================================

# Load environment variables from .env into process environment
load_dotenv()

# DATABASE_URL must exist; your entrypoint.sh sets it by default
# Example: postgresql://app:local@127.0.0.1:5432/appdb
DATABASE_URL = os.getenv("DATABASE_URL")

# Heroku-style "postgres://" fix (SQLAlchemy prefers "postgresql://")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL not set")

# Base class for ORM models
Base = declarative_base()

# Create DB engine (connection pool + dialect)
engine = create_engine(DATABASE_URL, echo=False, future=True)

# Factory to open DB sessions
SessionLocal = sessionmaker(bind=engine)


# ============================================================
# 2) ORM Model
# ============================================================

class Task(Base):
    """
    Each run creates a Task row.

    - id: UUID string
    - prompt: user prompt used to drive the workflow
    - status: 'running' | 'done' | 'error'
    - result: JSON blob (stored as text) with steps + outputs
    - sources: JSON list of selected sources, stored as text
    - n_articles: selected article count as string (5/10/15)
    """
    __tablename__ = "tasks"

    id = Column(String, primary_key=True, index=True)
    prompt = Column(Text)
    status = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    result = Column(Text)

    # Store UI configuration for traceability / debugging
    sources = Column(Text)      # JSON list: ["tavily", "arxiv"]
    n_articles = Column(String) # "5" | "10" | "15"


# ============================================================
# 3) Create/reset database schema (DEV ONLY)
# ============================================================
# Your existing main.py drops tables on startup. :contentReference[oaicite:2]{index=2}
# That’s convenient for dev (clean state), but will delete history on each restart.
# Keep it for now if you want; later you can guard by env flag.
try:
    Base.metadata.drop_all(bind=engine)
except Exception as e:
    print(f"❌ DB drop failed: {e}")

try:
    Base.metadata.create_all(bind=engine)
except Exception as e:
    print(f"❌ DB creation failed: {e}")


# ============================================================
# 4) FastAPI app initialization
# ============================================================

app = FastAPI()

# CORS: allow your UI (even if served from a different origin) to call the API.
# In production you'd lock this down.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static assets (optional): /static
app.mount("/static", StaticFiles(directory="static"), name="static")

# Jinja templates directory: /templates
templates = Jinja2Templates(directory="templates")


# ============================================================
# 5) Pydantic request models (what the UI POSTs)
# ============================================================

# Only allow these source values from the front-end.
AllowedSource = Literal["tavily", "wikipedia", "arxiv"]

# Only allow these article counts.
AllowedN = Literal[5, 10, 15]


class ResearchRequest(BaseModel):
    """
    /generate_report expects this JSON body:
    {
      "prompt": "...",
      "sources": ["tavily","arxiv"],
      "n_articles": 10
    }
    """
    prompt: str = Field(..., min_length=3)
    sources: List[AllowedSource] = Field(..., min_items=1)
    n_articles: AllowedN


class ChatRequest(BaseModel):
    """
    Used for post-run edits: user asks changes after final article is ready.
    """
    message: str = Field(..., min_length=1)


# ============================================================
# 6) Simple endpoints
# ============================================================

@app.get("/api", response_class=JSONResponse)
def health_check(request: Request):
    """Health check endpoint: easy test that API is alive."""
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    """
    Serves the UI.
    NOTE: we point to index_v2.html (new UI with sources + article count).
    """
    return templates.TemplateResponse("index_v2.html", {"request": request})


# ============================================================
# 7) Core endpoint: start workflow
# ============================================================

@app.post("/generate_report")
def generate_report(req: ResearchRequest):
    """
    Starts a new workflow run.

    1) Creates a Task row in DB
    2) Initializes a Task.result JSON object that will hold:
        - steps: list of pipeline steps
        - research_text, draft_markdown, article_markdown
        - chat: post-run editing chat
    3) Spawns a background thread to run the fixed pipeline.
    """
    task_id = str(uuid.uuid4())

    # The "result" JSON we store in DB:
    # - steps: used by UI to render the workflow progress
    # - article_markdown: final output
    initial_result = {
        "steps": [
            {"name": "extracting informations", "status": "pending", "detail": ""},
            {"name": "research agent", "status": "pending", "detail": ""},
            {"name": "write agent", "status": "pending", "detail": ""},
            {"name": "editor agent", "status": "pending", "detail": ""},
        ],
        "research_text": None,
        "draft_markdown": None,
        "article_markdown": None,
        "chat": [],
    }

    # Persist the Task row
    db = SessionLocal()
    db.add(
        Task(
            id=task_id,
            prompt=req.prompt,
            status="running",
            sources=json.dumps(req.sources),
            n_articles=str(req.n_articles),
            result=json.dumps(initial_result, ensure_ascii=False),
        )
    )
    db.commit()
    db.close()

    # Spawn background pipeline (threading is OK for dev / single container)
    thread = threading.Thread(
        target=run_pipeline,
        args=(task_id, req.prompt, req.sources, req.n_articles),
        daemon=True,
    )
    thread.start()

    return {"task_id": task_id}


# ============================================================
# 8) Read task state (UI polls this)
# ============================================================

@app.get("/tasks/{task_id}")
def get_task(task_id: str):
    """
    Returns the full task state.
    This is the endpoint the UI should poll (instead of memory-only progress dict).

    Response includes:
      - status
      - config: sources + n_articles
      - result: steps + outputs (research_text/draft/article)
    """
    db = SessionLocal()
    task = db.query(Task).filter(Task.id == task_id).first()
    db.close()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    result = json.loads(task.result) if task.result else {}

    return {
        "id": task.id,
        "status": task.status,
        "prompt": task.prompt,
        "config": {
            "sources": json.loads(task.sources) if task.sources else [],
            "n_articles": int(task.n_articles) if task.n_articles else None,
        },
        "result": result,
    }


# ============================================================
# 9) Optional: post-run chat endpoint to request edits
# ============================================================

@app.post("/tasks/{task_id}/chat")
def chat_edit(task_id: str, req: ChatRequest):
    """
    Allows user to request edits AFTER the pipeline is done.

    Behavior:
    - Load Task.result.article_markdown
    - Call editor_agent with: (article + user request)
    - Save updated article + chat history back to Task.result

    Why this exists:
    - Your user might want: "add a section", "shorten", "focus on X", etc.
    - No need to re-run research; just apply edits.
    """
    db = SessionLocal()
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        db.close()
        raise HTTPException(status_code=404, detail="Task not found")

    if task.status != "done":
        db.close()
        raise HTTPException(
            status_code=400,
            detail="Task is not completed yet; wait for the article to be ready.",
        )

    result = json.loads(task.result) if task.result else {}
    article_md = result.get("article_markdown")
    if not article_md:
        db.close()
        raise HTTPException(
            status_code=400,
            detail="No article available to edit for this task.",
        )

    chat_history = result.get("chat", [])

    # Editor prompt: include the whole article + instructions
    edit_prompt = f"""
        You are an academic editor. The user will request modifications to the article below.

        ARTICLE (Markdown):
        {article_md}

        USER REQUEST:
        {req.message}

        Apply the requested changes and return the FULL updated article in Markdown.
    """.strip()

    try:
        updated_article_md, _ = editor_agent(prompt=edit_prompt)
    except Exception as e:
        db.close()
        raise HTTPException(status_code=500, detail=f"Editor agent failed: {e}")

    # Store chat history for UI
    now_iso = datetime.utcnow().isoformat()
    chat_history.append({"role": "user", "message": req.message, "created_at": now_iso})
    chat_history.append(
        {
            "role": "assistant",
            "message": "Article updated based on your request.",
            "created_at": now_iso,
        }
    )

    # Persist the updated article + chat history
    result["article_markdown"] = updated_article_md
    result["chat"] = chat_history

    task.result = json.dumps(result, ensure_ascii=False)
    task.updated_at = datetime.utcnow()
    db.commit()
    db.close()

    return {"article_markdown": updated_article_md, "chat": chat_history}


# ============================================================
# 10) Helper functions for updating task state in DB
# ============================================================

def _esc(s: str) -> str:
    """
    Escapes HTML so you can safely show raw text inside the UI
    without it being interpreted as HTML/JS.
    """
    return html.escape(s or "")


def _update_step_in_db(task_id: str, step_name: str, status: str, detail: str = ""):
    """
    Loads Task.result (JSON), updates one step, and persists.

    Why do this:
    - You want step status even if the UI refreshes.
    - Avoids relying on in-memory dicts that vanish on restart.

    Step structure:
      result.steps = [
        { name: "...", status: "pending|running|done|error", detail: "...", updated_at: "..." }
      ]
    """
    db = SessionLocal()
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        db.close()
        return

    result = json.loads(task.result) if task.result else {}
    steps = result.get("steps", [])

    for step in steps:
        if step.get("name") == step_name:
            step["status"] = status
            if detail:
                step["detail"] = detail
            step["updated_at"] = datetime.utcnow().isoformat()
            break

    result["steps"] = steps
    task.result = json.dumps(result, ensure_ascii=False)
    task.updated_at = datetime.utcnow()

    db.commit()
    db.close()


def _update_result_field(task_id: str, **fields):
    """
    Patch top-level keys in Task.result, e.g.:
      _update_result_field(task_id, research_text="...", draft_markdown="...")

    Why:
    - keep writes small and consistent
    - avoid repeating the load/merge/persist logic everywhere
    """
    db = SessionLocal()
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        db.close()
        return

    result = json.loads(task.result) if task.result else {}
    result.update(fields)

    task.result = json.dumps(result, ensure_ascii=False)
    task.updated_at = datetime.utcnow()

    db.commit()
    db.close()


# ============================================================
# 11) Fixed pipeline runner (the "heart" of your new design)
# ============================================================

def run_pipeline(task_id: str, prompt: str, sources: List[str], n_articles: int):
    """
    Fixed 4-step pipeline:
      1) extracting informations
      2) research agent
      3) write agent
      4) editor agent

    Notes:
    - This is run in a background thread.
    - Each step updates Task.result.steps for UI to show progress.
    - Final status is written to Task.status.

    Where sources / n_articles actually take effect:
    - We pass them as explicit constraints to the research agent prompt.
    - The current research_agent in src/agents.py decides which tools to call.
      (It may call multiple tools by itself.) :contentReference[oaicite:3]{index=3}
    """
    try:
        # ---------------------------
        # STEP 1 — extracting informations
        # ---------------------------
        _update_step_in_db(
            task_id,
            "extracting informations",
            "running",
            "Validating request and extracting configuration from UI.",
        )

        # Here extraction is basically: we already have validated params via Pydantic.
        extraction_detail = {
            "prompt": prompt,
            "sources": sources,
            "n_articles": n_articles,
        }

        # Store a pretty JSON in step detail (escaped)
        _update_step_in_db(
            task_id,
            "extracting informations",
            "done",
            _esc(json.dumps(extraction_detail, indent=2, ensure_ascii=False)),
        )

        # ---------------------------
        # STEP 2 — research agent
        # ---------------------------
        _update_step_in_db(
            task_id,
            "research agent",
            "running",
            f"Running research agent with sources={sources} and n_articles={n_articles}.",
        )

        # IMPORTANT:
        # Your research_agent prompt should *force* it to respect sources + top N.
        # The agent internally has the tools available:
        # tavily_search_tool, arxiv_search_tool, wikipedia_search_tool :contentReference[oaicite:4]{index=4}
        # Tool implementations live in src/research_tools.py :contentReference[oaicite:5]{index=5}
        research_prompt = f"""
            You are a research agent. You MUST follow the user's configuration.

            USER PROMPT:
            {prompt}

            CONFIG:
            - Allowed sources (only these): {sources}
            - Select up to {n_articles} items total to support the writing draft.

            INSTRUCTIONS:
            1) Collect candidate items using ONLY the allowed sources.
            2) Return a structured output containing:
            - item_id (1..N)
            - title
            - authors (if available)
            - year/date (if available)
            - url
            - short note (1-3 lines why it matters)
            3) After listing the items, include a section "Synthesis" that extracts 5-10 key takeaways
            that will be used by the writer.

            Do NOT invent citations or URLs.
        """.strip()

        research_text, _ = research_agent(prompt=research_prompt)

        # Save the raw research output for traceability (helps debugging drafts)
        _update_result_field(task_id, research_text=research_text)

        _update_step_in_db(
            task_id,
            "research agent",
            "done",
            "Research agent completed. Findings stored in result.research_text.",
        )

        # ---------------------------
        # STEP 3 — writer agent
        # ---------------------------
        _update_step_in_db(
            task_id,
            "write agent",
            "running",
            "Drafting full academic report from research_text.",
        )

        # Writer gets the research_text; if you want stricter control,
        # you can wrap it in a writer-specific instruction template.
        draft_md, _ = writer_agent(prompt=research_text)

        _update_result_field(task_id, draft_markdown=draft_md)

        _update_step_in_db(
            task_id,
            "write agent",
            "done",
            "Writer agent completed. Draft stored in result.draft_markdown.",
        )

        # ---------------------------
        # STEP 4 — editor agent
        # ---------------------------
        _update_step_in_db(
            task_id,
            "editor agent",
            "running",
            "Refining and polishing the draft into the final article.",
        )

        final_md, _ = editor_agent(prompt=draft_md)

        _update_result_field(task_id, article_markdown=final_md)

        _update_step_in_db(
            task_id,
            "editor agent",
            "done",
            "Final article ready. Stored in result.article_markdown.",
        )

        # Mark task as done
        db = SessionLocal()
        task = db.query(Task).filter(Task.id == task_id).first()
        if task:
            task.status = "done"
            task.updated_at = datetime.utcnow()
            db.commit()
        db.close()

    except Exception as e:
        # Any exception means we mark task as error
        print(f"Workflow error for task {task_id}: {e}")

        # Best-effort: flag editor step as error (you can improve by tracking "current step")
        _update_step_in_db(
            task_id,
            "editor agent",
            "error",
            f"Error during pipeline execution: {str(e)}",
        )

        db = SessionLocal()
        task = db.query(Task).filter(Task.id == task_id).first()
        if task:
            task.status = "error"
            task.updated_at = datetime.utcnow()
            db.commit()
        db.close()