
import os
import uuid
import json
import threading
from datetime import datetime
from typing import Optional, Literal

# FastAPI core components
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

# Pydantic → validates request bodies
from pydantic import BaseModel

# SQLAlchemy → ORM for interacting with Postgres
from sqlalchemy import create_engine, Column, Text, DateTime, String
from sqlalchemy.orm import sessionmaker, declarative_base

# Loads variables from a .env file into environment variables
from dotenv import load_dotenv

# Internal agent functions (planner + executor)
from src.planning_agent import planner_agent, executor_agent_step

import html, textwrap

# ============================================================
# === Load environment variables (from .env)
# ============================================================
load_dotenv()

# Get DB connection URL from environment
DATABASE_URL = os.getenv("DATABASE_URL")

# Heroku sometimes uses "postgres://" but SQLAlchemy requires "postgresql://"
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# If DATABASE_URL is missing, stop the application
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL not set")

# ============================================================
# === SQLAlchemy / Database Setup
# ============================================================

# Base class all ORM models inherit from
Base = declarative_base()

# Engine → opens connections to the database
engine = create_engine(DATABASE_URL, echo=False, future=True)

# SessionLocal → factory used to open DB sessions (transactions)
SessionLocal = sessionmaker(bind=engine)

# ============================================================
# === ORM Model: Task
# ============================================================
# Represents a single research task stored in the database.
# Each workflow execution corresponds to one Task row.
class Task(Base):
    __tablename__ = "tasks"

    id = Column(String, primary_key=True, index=True)  # UUID of the task
    prompt = Column(Text)                              # User-provided input
    status = Column(String)                            # running / done / error
    created_at = Column(DateTime, default=datetime.utcnow)  # creation timestamp
    updated_at = Column(DateTime, default=datetime.utcnow)  # last modified
    result = Column(Text)                              # final report (JSON blob)

# ============================================================
# === Create (or reset) the database schema
# ============================================================

# Drop existing tables (dev only)
try:
    Base.metadata.drop_all(bind=engine)
except Exception as e:
    print(f"❌ DB drop failed: {e}")

# Create tables
try:
    Base.metadata.create_all(bind=engine)
except Exception as e:
    print(f"❌ DB creation failed: {e}")

# ============================================================
# === FastAPI Initialization
# ============================================================

# Create the FastAPI app
app = FastAPI()

# Enable CORS so any frontend can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # allow any domain
    allow_methods=["*"],    # allow any HTTP method
    allow_headers=["*"],    # allow any header
)

# Serve static files (CSS/JS/images) from /static
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize Jinja2 template rendering (for index.html)
templates = Jinja2Templates(directory="templates")

# ============================================================
# === In-Memory progress tracking (not stored in DB)
# ============================================================
# Stores current progress of each task_id:
# { task_id: { "steps": [ ... ] } }
task_progress = {}

# ============================================================
# === Pydantic model for API request validation
# ============================================================
# Used by /generate_report to enforce JSON schema
class PromptRequest(BaseModel):
    prompt: str

# ============================================================
# === Health Check Endpoint
# ============================================================
# Simple ping endpoint to confirm the API is running
@app.get("/api", response_class=JSONResponse)
def health_check(request: Request):
    return {"status": "ok"}
    
# ============================================================
# === Root Endpoint (UI)
# ============================================================
@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    """Serve the main UI page."""
    return templates.TemplateResponse("index.html", {"request": request})

# ============================================================
# === Generate Report Endpoint
# ============================================================
@app.post("/generate_report")
def generate_report(req: PromptRequest):
    """
    Entry point to start a new research/report generation workflow.

    This endpoint:
    1. Creates a new task record in the database
    2. Generates a high-level execution plan using the planner agent
    3. Initializes in-memory progress tracking
    4. Starts the agent workflow asynchronously in a background thread
    5. Immediately returns a task_id to the client
    """

    # 1. Create a unique identifier for this task execution
    task_id = str(uuid.uuid4())

    # 2. Persist the task metadata in the database
    #    (status starts as "running")
    db = SessionLocal()
    db.add(
        Task(
            id=task_id,
            prompt=req.prompt,
            status="running",
        )
    )
    db.commit()
    db.close()

    # 3. Initialize in-memory progress tracking
    #    This is used by /task_progress/{task_id}
    task_progress[task_id] = {"steps": []}
    

    # 4. Ask the planner agent to break the prompt into steps
    #    Returns something like:
    #    [
    #      "Research agent: Use Tavily ...",
    #      "Research agent: Search arXiv ...",
    #      "Writer agent: Draft report ...",
    #      ...
    #    ]
    initial_plan_steps = planner_agent(req.prompt)

    # 5. Pre-populate progress structure with "pending" steps
    #    These will be updated live as the workflow runs
    for step_title in initial_plan_steps:
        task_progress[task_id]["steps"].append(
            {
                "title": step_title,               # Human-readable step name
                "status": "pending",               # pending | running | done | error
                "description": "Awaiting execution",
                "substeps": [],                     # Filled with agent calls / outputs
            }
        )

    # 6. Start the agent workflow asynchronously
    #    - run_agent_workflow executes each step sequentially
    #    - threading is used so the HTTP request can return immediately
    thread = threading.Thread(
        target=run_agent_workflow,
        args=(task_id, req.prompt, initial_plan_steps),
    )
    thread.start()

    # 7. Return immediately with the task_id, the frontend will poll /task_progress and /task_status
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
    }


def format_history(history):
    return "\n\n".join(
        f"🔹 {title}\n{desc}\n\n📝 Output:\n{output}" for title, desc, output in history
    )


def run_agent_workflow(task_id: str, prompt: str, initial_plan_steps: list):
    steps_data = task_progress[task_id]["steps"]
    execution_history = []

    def update_step_status(index, status, description="", substep=None):
        if index < len(steps_data):
            steps_data[index]["status"] = status
            if description:
                steps_data[index]["description"] = description
            if substep:
                steps_data[index]["substeps"].append(substep)
            steps_data[index]["updated_at"] = datetime.utcnow().isoformat()

    try:
        for i, plan_step_title in enumerate(initial_plan_steps):
            update_step_status(i, "running", f"Executing: {plan_step_title}")

            actual_step_description, agent_name, output = executor_agent_step(
                plan_step_title, execution_history, prompt
            )

            execution_history.append([plan_step_title, actual_step_description, output])

            def esc(s: str) -> str:
                return html.escape(s or "")

            def nl2br(s: str) -> str:
                return esc(s).replace("\n", "<br>")

            # ...
            update_step_status(
                i,
                "done",
                f"Completed: {plan_step_title}",
                {
                    "title": f"Called {agent_name}",
                    "content": f"""
                        <div style='border:1px solid #ccc; border-radius:8px; padding:10px; margin:8px 0; background:#fff;'>
                        <div style='font-weight:bold; color:#2563eb;'>📘 User Prompt</div>
                        <div style='white-space:pre-wrap;'>{prompt}</div>

                        <div style='font-weight:bold; color:#16a34a; margin-top:8px;'>📜 Previous Step</div>
                        <pre style='white-space:pre-wrap; background:#f9fafb; padding:6px; border-radius:6px; margin:0;'>
                        {format_history(execution_history[-2:-1])}
                        </pre>

                        <div style='font-weight:bold; color:#f59e0b; margin-top:8px;'>🧹 Your next task</div>
                        <div style='white-space:pre-wrap;'>{actual_step_description}</div>

                        <div style='font-weight:bold; color:#10b981; margin-top:8px;'>✅ Output</div>
                        <!-- ⚠️ NO <pre> AQUÍ -->
                        <div style='white-space:pre-wrap;'>
                        {output}
                        </div>
                        </div>
                    """.strip(),
                },
            )

        final_report_markdown = (
            execution_history[-1][-1] if execution_history else "No report generated."
        )

        result = {"html_report": final_report_markdown, "history": steps_data}

        db = SessionLocal()
        task = db.query(Task).filter(Task.id == task_id).first()
        task.status = "done"
        task.result = json.dumps(result)
        task.updated_at = datetime.utcnow()
        db.commit()
        db.close()

    except Exception as e:
        print(f"Workflow error for task {task_id}: {e}")
        if steps_data:
            error_step_index = next(
                (i for i, s in enumerate(steps_data) if s["status"] == "running"),
                len(steps_data) - 1,
            )
            if error_step_index >= 0:
                update_step_status(
                    error_step_index,
                    "error",
                    f"Error during execution: {e}",
                    {"title": "Error", "content": str(e)},
                )

        db = SessionLocal()
        task = db.query(Task).filter(Task.id == task_id).first()
        task.status = "error"
        task.updated_at = datetime.utcnow()
        db.commit()
        db.close()
