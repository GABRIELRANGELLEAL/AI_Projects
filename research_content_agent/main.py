import os
import uuid
import json
import threading
from datetime import datetime
from typing import List, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from sqlalchemy import create_engine, Column, Text, DateTime, String
from sqlalchemy.orm import sessionmaker, declarative_base

from dotenv import load_dotenv

import html

# We will reuse your existing agents
from src.agents import writer_agent, editor_agent, research_agent


# ============================================================
# Env / DB
# ============================================================
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL not set")

Base = declarative_base()
engine = create_engine(DATABASE_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine)


class Task(Base):
    __tablename__ = "tasks"

    id = Column(String, primary_key=True, index=True)
    prompt = Column(Text)
    status = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    result = Column(Text)

    # new: store configuration
    sources = Column(Text)     # JSON list of strings
    n_articles = Column(String)


# Note: Your previous main.py drops tables on startup (dev only). Keep or remove as you prefer.
try:
    Base.metadata.drop_all(bind=engine)
except Exception as e:
    print(f"❌ DB drop failed: {e}")

try:
    Base.metadata.create_all(bind=engine)
except Exception as e:
    print(f"❌ DB creation failed: {e}")


# ============================================================
# FastAPI app
# ============================================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# ============================================================
# Request model (includes sources + article count)
# ============================================================
AllowedSource = Literal["tavily", "wikipedia", "arxiv"]
AllowedN = Literal[5, 10, 15]


class ResearchRequest(BaseModel):
    prompt: str = Field(..., min_length=3)
    sources: List[AllowedSource] = Field(..., min_items=1)
    n_articles: AllowedN


@app.get("/api", response_class=JSONResponse)
def health_check(request: Request):
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index_v2.html", {"request": request})


@app.post("/generate_report")
def generate_report(req: ResearchRequest):
    """
    Entry point for a new end-to-end workflow.
    Creates a root Task row and kicks off the background pipeline:
    1) extracting informations
    2) research agent
    3) write agent
    4) editor agent

    All step progress + final article are stored in Task.result (JSON).
    """
    task_id = str(uuid.uuid4())

    # Initial result structure persisted in DB and updated as we go
    initial_result = {
        "steps": [
            {"name": "extracting informations", "status": "pending", "detail": ""},
            {"name": "research agent", "status": "pending", "detail": ""},
            {"name": "write agent", "status": "pending", "detail": ""},
            {"name": "editor agent", "status": "pending", "detail": ""},
        ],
        "article_markdown": None,
        "chat": [],
    }

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

    # Run the pipeline in the background
    thread = threading.Thread(
        target=run_pipeline,
        args=(task_id, req.prompt, req.sources, req.n_articles),
        daemon=True,
    )
    thread.start()

    return {"task_id": task_id}


@app.get("/tasks/{task_id}")
def get_task(task_id: str):
    """
    Returns the full state of a given task, including:
    - status
    - prompt and config (sources, n_articles)
    - current steps metadata
    - latest article markdown (if any)
    - chat history
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


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)


@app.post("/tasks/{task_id}/chat")
def chat_edit(task_id: str, req: ChatRequest):
    """
    Simple chat endpoint to request edits on the final article.
    It:
    - loads the latest article from Task.result.article_markdown
    - calls the editor_agent with the user's instructions
    - stores the updated article + chat history back into Task.result
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

    # Update chat history (very simple: store user messages + acks)
    now_iso = datetime.utcnow().isoformat()
    chat_history.append({"role": "user", "message": req.message, "created_at": now_iso})
    chat_history.append(
        {
            "role": "assistant",
            "message": "Artigo atualizado conforme sua solicitação.",
            "created_at": now_iso,
        }
    )

    result["article_markdown"] = updated_article_md
    task.result = json.dumps(result, ensure_ascii=False)
    task.updated_at = datetime.utcnow()
    db.commit()
    db.close()

    return {"article_markdown": updated_article_md, "chat": chat_history}


# ============================================================
# Workflow helpers
# ============================================================
def _esc(s: str) -> str:
    return html.escape(s or "")


def _update_step_in_db(task_id: str, step_name: str, status: str, detail: str = ""):
    """
    Load Task.result JSON, update the matching step, and persist.
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
    Convenience helper to patch top-level fields in Task.result.
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


def run_pipeline(task_id: str, prompt: str, sources: List[str], n_articles: int):
    """
    Fixed, but explicitly named, 4-step pipeline:
    1) extracting informations
    2) research agent
    3) write agent
    4) editor agent

    Each step updates Task.result and Task.status as it progresses.
    """
    try:
        # STEP 0 — extracting informations
        _update_step_in_db(
            task_id,
            "extracting informations",
            "running",
            "Validating request and extracting configuration.",
        )
        # Here the 'extraction' is essentially the validated ResearchRequest.
        extraction_detail = {
            "prompt": prompt,
            "sources": sources,
            "n_articles": n_articles,
        }
        _update_step_in_db(
            task_id,
            "extracting informations",
            "done",
            _esc(json.dumps(extraction_detail, indent=2, ensure_ascii=False)),
        )

        # STEP 1 — research agent
        _update_step_in_db(
            task_id,
            "research agent",
            "running",
            f"Running research agent with sources={sources} and n_articles={n_articles}.",
        )

        research_text, _ = research_agent(prompt=prompt)

        _update_result_field(task_id, research_text=research_text)
        
        _update_step_in_db(
            task_id,
            "research agent",
            "done",
            "Research agent completed. Findings stored in result.research_text.",
        )

        # STEP 2 — write agent
        _update_step_in_db(
            task_id,
            "write agent",
            "running",
            "Drafting full academic report from research_text.",
        )


        draft_md, _ = writer_agent(prompt=research_text)

        _update_result_field(task_id, draft_markdown=draft_md)

        _update_step_in_db(
            task_id,
            "write agent",
            "done",
            "Writer agent completed. Draft stored in result.draft_markdown.",
        )

        # STEP 3 — editor agent
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
        print(f"Workflow error for task {task_id}: {e}")

        # Mark last step as error (best effort)
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
