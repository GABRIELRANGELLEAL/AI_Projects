"""
main.py — Fixed workflow research agent (FastAPI + Postgres)
Correção: Adicionado "wikipedia" ao AllowedSource para evitar erro 422.
"""

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

# Importação dos agentes
from src.agents import writer_agent, editor_agent, research_agent


# ============================================================
# 1) Configuração do Banco de Dados
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


# ============================================================
# 2) Modelo ORM
# ============================================================

class Task(Base):
    __tablename__ = "tasks"
    id = Column(String, primary_key=True, index=True)
    prompt = Column(Text)
    status = Column(String) 
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    result = Column(Text)
    sources = Column(Text)      


# ============================================================
# 3) Setup DB
# ============================================================
try:
    Base.metadata.create_all(bind=engine)
except Exception as e:
    print(f"❌ DB setup failed: {e}")


# ============================================================
# 4) Inicialização do FastAPI
# ============================================================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Certifique-se de que as pastas static e templates existam
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# ============================================================
# 5) Modelos Pydantic (CORREÇÃO AQUI)
# ============================================================

# Adicionado "wikipedia" para sincronizar com o index_v3.html
AllowedSource = Literal["arxiv", "scielo", "wikipedia"]

class ResearchRequest(BaseModel):
    prompt: str = Field(..., min_length=3)
    sources: List[AllowedSource] = Field(..., min_items=1)

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)


# ============================================================
# 6) Endpoints
# ============================================================

@app.get("/api")
def health_check():
    return {"status": "ok"}

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index_v3.html", {"request": request})

@app.post("/generate_report")
def generate_report(req: ResearchRequest):
    task_id = str(uuid.uuid4())
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

    db = SessionLocal()
    db.add(Task(
        id=task_id,
        prompt=req.prompt,
        status="running",
        sources=json.dumps(req.sources),
        result=json.dumps(initial_result, ensure_ascii=False),
    ))
    db.commit()
    db.close()

    thread = threading.Thread(
        target=run_pipeline,
        args=(task_id, req.prompt, req.sources),
        daemon=True,
    )
    thread.start()

    return {"task_id": task_id}

@app.get("/tasks/{task_id}")
def get_task(task_id: str):
    db = SessionLocal()
    task = db.query(Task).filter(Task.id == task_id).first()
    db.close()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return {
        "id": task.id,
        "status": task.status,
        "prompt": task.prompt,
        "result": json.loads(task.result) if task.result else {}
    }

@app.post("/tasks/{task_id}/cancel")
def cancel_task(task_id: str):
    db = SessionLocal()
    task = db.query(Task).filter(Task.id == task_id).first()
    if task and task.status == "running":
        task.status = "cancelled"
        db.commit()
    db.close()
    return {"status": "cancelled"}


# ============================================================
# 7) Helpers e Workflow
# ============================================================

def _update_step_in_db(task_id: str, step_name: str, status: str, detail: str = ""):
    db = SessionLocal()
    task = db.query(Task).filter(Task.id == task_id).first()
    if task:
        result = json.loads(task.result)
        for step in result["steps"]:
            if step["name"] == step_name:
                step["status"] = status
                if detail: step["detail"] = detail
                break
        task.result = json.dumps(result, ensure_ascii=False)
        db.commit()
    db.close()

def _update_result_field(task_id: str, **fields):
    db = SessionLocal()
    task = db.query(Task).filter(Task.id == task_id).first()
    if task:
        result = json.loads(task.result)
        result.update(fields)
        task.result = json.dumps(result, ensure_ascii=False)
        db.commit()
    db.close()

def _is_cancelled(task_id: str) -> bool:
    db = SessionLocal()
    task = db.query(Task).filter(Task.id == task_id).first()
    cancelled = task and task.status == "cancelled"
    db.close()
    return cancelled

def run_pipeline(task_id: str, prompt: str, sources: List[str]):
    try:
        if _is_cancelled(task_id): return
        
        # Etapa 1
        _update_step_in_db(task_id, "extracting informations", "running")
        _update_step_in_db(task_id, "extracting informations", "done")

        if _is_cancelled(task_id): return

        # Etapa 2 - Research (CORREÇÃO: Usando o prompt estruturado)
        _update_step_in_db(task_id, "research agent", "running")
        
        # Criamos uma instrução clara para o agente sobre as fontes
        structured_research_prompt = f"Topic: {prompt}, Sources: {', '.join(sources)}"
        
        research_text, _ = research_agent(prompt=structured_research_prompt)
        _update_result_field(task_id, research_text=research_text)
        _update_step_in_db(task_id, "research agent", "done")

        if _is_cancelled(task_id): return

        # Etapa 3 - Writer
        _update_step_in_db(task_id, "write agent", "running")
        draft_md, _ = writer_agent(prompt=research_text)
        _update_result_field(task_id, draft_markdown=draft_md)
        _update_step_in_db(task_id, "write agent", "done")

        if _is_cancelled(task_id): return

        # Etapa 4 - Editor
        _update_step_in_db(task_id, "editor agent", "running")
        final_md, _ = editor_agent(prompt=draft_md)
        _update_result_field(task_id, article_markdown=final_md)
        _update_step_in_db(task_id, "editor agent", "done")

        # Conclusão
        db = SessionLocal()
        task = db.query(Task).filter(Task.id == task_id).first()
        if task:
            task.status = "done"
            db.commit()
        db.close()

    except Exception as e:
        print(f"Error: {e}")
        _update_step_in_db(task_id, "editor agent", "error", str(e))
        db = SessionLocal()
        task = db.query(Task).filter(Task.id == task_id).first()
        if task:
            task.status = "error"
            db.commit()
        db.close()