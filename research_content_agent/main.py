"""
main.py — Research Content Agent (FastAPI)

Fluxo atual (1-shot):
- frontend envia uma sentença
- `key_word` extrai frases de busca
- `arxiv_search_tool` busca artigos no arXiv
- `research_agent` ranqueia/atribui scores aos artigos
- backend retorna o JSON final para o frontend renderizar
"""

# =========================
# Imports padrão do Python
# =========================
from typing import Any
import json
import time
import uuid
import os
# =========================
# Imports do FastAPI
# =========================
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# =========================
# Validação de payloads
# =========================
from pydantic import BaseModel, Field

# =========================
# env key
# =========================

from dotenv import load_dotenv

load_dotenv()

from src.agents import key_word, research_agent
from src.research_tools import arxiv_search_tool
# =========================
# Importação dos agentes de IA
# =========================
# key_word: extrai palavras-chave do prompt
# research_agent: rankeia/atribui score aos papers retornados
# =========================
from src.agents import key_word, research_agent
from src.research_tools import arxiv_search_tool

# ============================================================
# Modelos (Pydantic) — fluxo simples
# ============================================================

class SimpleResearchRequest(BaseModel):
    """
    Payload esperado para o fluxo simples (1-shot):
    - recebe uma sentença do frontend
    - extrai keywords
    - busca arXiv
    - rankeia com research_agent
    - retorna JSON final para o frontend renderizar
    """
    sentence: str = Field(..., min_length=3)
    max_results: int = Field(10, ge=1, le=200)
    fetch_pdf: bool = False
    add_subjective: bool = True
# ============================================================
# Inicialização do app FastAPI and routes
# ============================================================

app = FastAPI()

# Libera CORS para qualquer origem
# Isso facilita no desenvolvimento, mas em produção o ideal é restringir
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Expõe arquivos estáticos em /static
app.mount("/static", StaticFiles(directory="static"), name="static")

# Diretório de templates HTML
templates = Jinja2Templates(directory="templates")

# Creating endpoints

@app.get("/")
def read_root(request: Request):
    """
    Endpoint raiz.
    Renderiza o frontend principal.
    """
    return templates.TemplateResponse(request=request, name="index_v3.html", context={"request": request})

@app.post("/research")
def research(req: SimpleResearchRequest) -> dict[str, Any]:
    """
    Fluxo simples solicitado:
    sentence -> key_word -> arxiv_search_tool -> research_agent -> output

    Retorna um JSON com:
    - keywords: saída do key_word (inclui url_to_query)
    - papers_raw: retorno da busca no arXiv
    - papers_ranked: retorno do research_agent (scores + rank_score)
    """
    kw = key_word(prompt=req.sentence) or {}
    url_to_query = (kw.get("url_to_query") or "").strip()

    if not url_to_query:
        raise HTTPException(status_code=400, detail="key_word não retornou url_to_query.")

    papers = arxiv_search_tool(url_to_query, max_results=req.max_results, fetch_pdf=req.fetch_pdf)

    ranked = research_agent(
        query=req.sentence,
        papers=papers,
        add_subjective=req.add_subjective,
    )

    if ranked:
        ranked = sorted(ranked, key=lambda x: float(x.get("rank_score", 0)), reverse=True)

    return {
        "sentence": req.sentence,
        "keywords": kw,
        "papers_raw": papers,
        "papers_ranked": ranked,
    }


@app.post("/research/stream")
def research_stream(req: SimpleResearchRequest):
    """
    Mesmo fluxo do `/research`, mas emitindo eventos (SSE) para o frontend mostrar
    progresso sequencial (etapas) em tempo real.

    Eventos:
    - task: { task_id }
    - progress: { step_index, step_name, message }
    - result: payload final (mesma estrutura do /research)
    - error: { message }
    """

    task_id = f"run_{uuid.uuid4().hex}"
    steps = [
        "Extração de Palavras-chave",
        "Busca no Diretório (arXiv)",
        "Avaliação e Ranqueamento (IA)",
    ]

    def _sse(event: str, data: dict[str, Any]) -> str:
        return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    def _gen():
        t0 = time.time()
        print(f"[{task_id}] START /research/stream sentence_len={len(req.sentence)} max_results={req.max_results} fetch_pdf={req.fetch_pdf} add_subjective={req.add_subjective}", flush=True)
        yield _sse("task", {"task_id": task_id, "steps": steps})

        try:
            print(f"[{task_id}] STEP 1/3 {steps[0]}", flush=True)
            yield _sse("progress", {"task_id": task_id, "step_index": 0, "step_name": steps[0], "message": "Extraindo termos de busca..."})
            kw = key_word(prompt=req.sentence) or {}
            url_to_query = (kw.get("url_to_query") or "").strip()
            if not url_to_query:
                raise ValueError("key_word não retornou url_to_query.")
            yield _sse("progress", {"task_id": task_id, "step_index": 0, "step_name": steps[0], "message": f"OK. {len(kw.get('search_phrases') or [])} frase(s) gerada(s)."})

            
            print(f"[{task_id}] STEP 2/3 {steps[1]}", flush=True)
            yield _sse("progress", {"task_id": task_id, "step_index": 1, "step_name": steps[1], "message": "Consultando arXiv..."})
            papers = arxiv_search_tool(url_to_query, max_results=req.max_results, fetch_pdf=req.fetch_pdf)
            yield _sse("progress", {"task_id": task_id, "step_index": 1, "step_name": steps[1], "message": f"OK. {len(papers or [])} resultado(s)."})

           
            print(f"[{task_id}] STEP 3/3 {steps[2]}", flush=True)
            yield _sse("progress", {"task_id": task_id, "step_index": 2, "step_name": steps[2], "message": "Ranqueando artigos com IA..."})
            ranked = research_agent(
                query=req.sentence,
                papers=papers,
                add_subjective=req.add_subjective,
            )
            if ranked:
                ranked = sorted(ranked, key=lambda x: float(x.get("rank_score", 0)), reverse=True)

            yield _sse("progress", {"task_id": task_id, "step_index": 2, "step_name": steps[2], "message": f"OK. {len(ranked or [])} artigo(s) ranqueado(s)."})

            out = {
                "task_id": task_id,
                "sentence": req.sentence,
                "keywords": kw,
                "papers_raw": papers,
                "papers_ranked": ranked,
            }
            
            print(f"[{task_id}] DONE elapsed_s={time.time()-t0:.2f}", flush=True)
            yield _sse("result", out)
            
        except Exception as e:
            msg = str(e)
            print(f"[{task_id}] ERROR {msg}", flush=True)
            yield _sse("error", {"task_id": task_id, "message": msg})

    return StreamingResponse(_gen(), media_type="text/event-stream")
