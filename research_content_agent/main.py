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

# =========================
# Imports do FastAPI
# =========================
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

# =========================
# Validação de payloads
# =========================
from pydantic import BaseModel, Field

# =========================
# Carrega variáveis do .env
# =========================
from dotenv import load_dotenv

# =========================
# Importação dos agentes de IA
# =========================
# key_word: extrai palavras-chave do prompt
# research_agent: rankeia/atribui score aos papers retornados
# =========================
from src.agents import key_word, research_agent
from src.research_tools import arxiv_search_tool

load_dotenv()

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

    return {
        "sentence": req.sentence,
        "keywords": kw,
        "papers_raw": papers,
        "papers_ranked": ranked,
    }

