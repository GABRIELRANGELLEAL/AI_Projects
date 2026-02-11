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

# We will reuse your existing writer/editor agents
from src.agents import writer_agent, editor_agent

# And reuse your existing tool implementations
from src.research_tools import tavily_search_tool, wikipedia_search_tool, arxiv_search_tool


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

task_progress = {}


# ============================================================
# Request model (now includes sources + article count)
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
    return templates.TemplateResponse("index_2.html", {"request": request})


@app.post("/generate_report")
def generate_report(req: ResearchRequest):
    task_id = str(uuid.uuid4())

    # Persist task
    db = SessionLocal()
    db.add(
        Task(
            id=task_id,
            prompt=req.prompt,
            status="running",
            sources=json.dumps(req.sources),
            n_articles=str(req.n_articles),
        )
    )
    db.commit()
    db.close()

    # Fixed steps (no planner)
    task_progress[task_id] = {
        "steps": [
            {
                "title": "Research: collect candidates from selected sources",
                "status": "pending",
                "description": "Waiting to start",
                "substeps": [],
            },
            {
                "title": "Writer: create draft using top N items",
                "status": "pending",
                "description": "Waiting for research results",
                "substeps": [],
            },
            {
                "title": "Editor: refine and finalize",
                "status": "pending",
                "description": "Waiting for draft",
                "substeps": [],
            },
        ]
    }

    thread = threading.Thread(
        target=run_fixed_workflow,
        args=(task_id, req.prompt, req.sources, req.n_articles),
        daemon=True,
    )
    thread.start()

    return {"task_id": task_id}


@app.get("/task_progress/{task_id}")
def get_task_progress(task_id: str):
    return task_progress.get(task_id, {"steps": []})


@app.get("/task_status/{task_id}")
def get_task_status(task_id: str):
    db = SessionLocal()
    task = db.query(Task).filter(Task.id == task_id).first()
    db.close()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    return {
        "status": task.status,
        "result": json.loads(task.result) if task.result else None,
        "config": {
            "sources": json.loads(task.sources) if task.sources else [],
            "n_articles": int(task.n_articles) if task.n_articles else None,
        },
    }


# ============================================================
# Workflow helpers
# ============================================================
def _esc(s: str) -> str:
    return html.escape(s or "")


def _update_step(task_id: str, idx: int, status: str, description: str = "", substep=None):
    steps = task_progress.get(task_id, {}).get("steps", [])
    if idx >= len(steps):
        return
    steps[idx]["status"] = status
    if description:
        steps[idx]["description"] = description
    if substep:
        steps[idx]["substeps"].append(substep)
    steps[idx]["updated_at"] = datetime.utcnow().isoformat()


def _run_selected_sources(prompt: str, sources: List[str], n_articles: int):
    """
    Runs the selected tools and returns a normalized list of items.
    We intentionally retrieve ~n_articles per selected source,
    then we trim to n_articles overall (writer will use top N).
    """
    per_source = max(2, n_articles)  # simple rule: fetch up to N per source if possible
    items = []

    if "tavily" in sources:
        tav = tavily_search_tool(query=prompt, max_results=per_source, include_images=False)
        for r in tav:
            if "error" in r:
                continue
            items.append(
                {
                    "source": "tavily",
                    "title": r.get("title", ""),
                    "authors": [],
                    "year": "",
                    "url": r.get("url", ""),
                    "snippet": r.get("content", ""),
                }
            )

    if "wikipedia" in sources:
        wiki = wikipedia_search_tool(query=prompt, sentences=6)
        for r in wiki:
            if "error" in r:
                continue
            items.append(
                {
                    "source": "wikipedia",
                    "title": r.get("title", ""),
                    "authors": [],
                    "year": "",
                    "url": r.get("url", ""),
                    "snippet": r.get("summary", ""),
                }
            )

    if "arxiv" in sources:
        ax = arxiv_search_tool(query=prompt, max_results=per_source)
        for r in ax:
            if "error" in r:
                continue
            items.append(
                {
                    "source": "arxiv",
                    "title": r.get("title", ""),
                    "authors": r.get("authors", []) or [],
                    "year": (r.get("published", "") or "")[:4],
                    "url": r.get("url", ""),
                    "snippet": r.get("summary", ""),
                    "pdf": r.get("link_pdf", ""),
                }
            )

    # naive trimming (you can add ranking/dedup later)
    # for now: keep first n_articles items
    return items[:n_articles]


def run_fixed_workflow(task_id: str, prompt: str, sources: List[str], n_articles: int):
    try:
        # STEP 1 — research
        _update_step(task_id, 0, "running", f"Searching sources: {', '.join(sources)}")
        items = _run_selected_sources(prompt, sources, n_articles)

        _update_step(
            task_id,
            0,
            "done",
            f"Collected {len(items)} items (target: {n_articles})",
            substep={
                "title": "Collected items (normalized)",
                "content": _esc(json.dumps(items, indent=2, ensure_ascii=False)),
            },
        )

        # STEP 2 — writer
        _update_step(task_id, 1, "running", f"Drafting with top {n_articles} items")

        writer_prompt = f"""
Create a detailed academic-style Markdown report based on the user prompt and the research items below.

USER PROMPT:
{prompt}

CONSTRAINTS:
- Use ONLY the items provided below as sources.
- Use numeric inline citations [1], [2], etc.
- References section must include links for every cited item.
- The user requested exactly {n_articles} items max; do not invent additional sources.

RESEARCH ITEMS (JSON):
{json.dumps(items, indent=2, ensure_ascii=False)}
""".strip()

        draft_md, _ = writer_agent(prompt=writer_prompt)

        _update_step(
            task_id,
            1,
            "done",
            "Draft created",
            substep={"title": "Writer output (draft)", "content": _esc(draft_md)},
        )

        # STEP 3 — editor
        _update_step(task_id, 2, "running", "Refining final report")

        editor_prompt = f"""
Refine the following Markdown report:
- Improve structure and clarity
- Keep citations [n] consistent
- Keep References complete and clickable

REPORT:
{draft_md}
""".strip()

        final_md, _ = editor_agent(prompt=editor_prompt)

        _update_step(
            task_id,
            2,
            "done",
            "Final report ready",
            substep={"title": "Editor output (final)", "content": _esc(final_md)},
        )

        result = {
            "html_report": final_md,  # frontend already renders markdown to html
            "config": {"sources": sources, "n_articles": n_articles},
        }

        db = SessionLocal()
        task = db.query(Task).filter(Task.id == task_id).first()
        task.status = "done"
        task.result = json.dumps(result, ensure_ascii=False)
        task.updated_at = datetime.utcnow()
        db.commit()
        db.close()

    except Exception as e:
        print(f"Workflow error for task {task_id}: {e}")
        _update_step(task_id, 2, "error", f"Error: {str(e)}", substep={"title": "Error", "content": _esc(str(e))})

        db = SessionLocal()
        task = db.query(Task).filter(Task.id == task_id).first()
        if task:
            task.status = "error"
            task.updated_at = datetime.utcnow()
            db.commit()
        db.close()
