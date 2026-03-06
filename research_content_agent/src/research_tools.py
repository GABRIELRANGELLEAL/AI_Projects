"""
Ferramentas de Pesquisa para Agentes LLM.
Este módulo contém funções para buscar informações no arXiv (com extração de PDF),
Tavily (pesquisa web geral) e Wikipedia, além de definições de ferramentas (schemas) 
para uso com modelos de linguagem.
"""

from typing import List, Dict, Optional
import os
import re
import time
import tempfile
import xml.etree.ElementTree as ET
from io import BytesIO
import wikipedia

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ----- Configuração de Sessão com Retentativas (Retries) & Headers -----

def _build_session(
    user_agent: str = "LF-ADP-Agent/1.0 (mailto:your.email@example.com)",
) -> requests.Session:
    """
    Cria e configura uma sessão do requests com uma política de retentativas (retry)
    robusta para lidar com falhas temporárias de rede ou limites de taxa (rate limits).
    """
    s = requests.Session()
    # Define cabeçalhos padrão para simular um navegador/agente real e evitar bloqueios
    s.headers.update(
        {
            "User-Agent": user_agent,
            "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        }
    )
    # Configura a estratégia de repetição: tenta até 5 vezes em caso de erros 429, 50x, etc.
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=0.6, # Tempo de espera exponencial entre as tentativas
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD"]),
        raise_on_redirect=False,
        raise_on_status=False,
    )
    # Monta o adaptador HTTP na sessão para conexões HTTP e HTTPS
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=20)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s

# Inicializa uma sessão global para ser reutilizada pelas funções
session = _build_session()


# ----- Funções Utilitárias para Tratamento de PDFs e Textos -----
def ensure_pdf_url(abs_or_pdf_url: str) -> str:
    """
    Garante que a URL do arXiv aponte diretamente para o arquivo PDF 
    em vez da página de resumo (abstract).
    """
    url = abs_or_pdf_url.strip().replace("http://", "https://")
    if "/pdf/" in url and url.endswith(".pdf"):
        return url
    # Substitui a rota '/abs/' (abstract) por '/pdf/'
    url = url.replace("/abs/", "/pdf/")
    if not url.endswith(".pdf"):
        url += ".pdf"
    return url

def _safe_filename(name: str) -> str:
    """
    Remove caracteres especiais de uma string para torná-la um nome de arquivo seguro.
    """
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    return name

def clean_text(s: str) -> str:
    """
    Limpa e normaliza o texto extraído de PDFs (remove quebras de palavras, 
    espaços extras e múltiplas quebras de linha).
    """
    s = re.sub(r"-\n", "", s)         # Une palavras separadas por hífen no final da linha
    s = re.sub(r"\r\n|\r", "\n", s)   # Normaliza os saltos de linha para o padrão Unix
    s = re.sub(r"[ \t]+", " ", s)     # Colapsa múltiplos espaços em um único espaço
    s = re.sub(r"\n{3,}", "\n\n", s)  # Evita mais de uma linha em branco seguida
    return s.strip()

def fetch_pdf_bytes(pdf_url: str, timeout: int = 90) -> bytes:
    """
    Faz o download do PDF em formato de bytes usando a sessão configurada.
    """
    r = session.get(pdf_url, timeout=timeout, allow_redirects=True)
    r.raise_for_status() # Levanta exceção se o status HTTP indicar erro
    return r.content

def pdf_bytes_to_text(pdf_bytes: bytes, max_pages: Optional[int] = None) -> str:
    """
    Extrai texto de um PDF em memória (bytes). 
    Tenta usar PyMuPDF (fitz) primeiro, e se falhar, tenta pdfminer.six.
    """
    # 1ª Tentativa: PyMuPDF (geralmente mais rápido e preciso)
    try:
        import fitz  # PyMuPDF
        out = []
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            n = len(doc)
            limit = n if max_pages is None else min(max_pages, n)
            for i in range(limit):
                out.append(doc.load_page(i).get_text("text"))
        return "\n".join(out)
    except Exception:
        pass # Se falhar ou não estiver instalado, passa para o fallback

    # 2ª Tentativa: pdfminer.six (fallback)
    try:
        from pdfminer.high_level import extract_text_to_fp
        buf_in = BytesIO(pdf_bytes)
        buf_out = BytesIO()
        extract_text_to_fp(buf_in, buf_out)
        return buf_out.getvalue().decode("utf-8", errors="ignore")
    except Exception as e:
        raise RuntimeError(f"PDF text extraction failed: {e}")

def maybe_save_pdf(pdf_bytes: bytes, dest_dir: str, filename: str) -> str:
    """
    Salva os bytes do PDF fisicamente no disco em um diretório especificado.
    """
    os.makedirs(dest_dir, exist_ok=True)
    path = os.path.join(dest_dir, _safe_filename(filename))
    with open(path, "wb") as f:
        f.write(pdf_bytes)
    return path


# ----- Ferramenta de Pesquisa: arXiv -----
def arxiv_search_tool(
    query: str,
    max_results: int = 3,
) -> List[Dict]:
    """
    Busca artigos no arXiv. Retorna uma lista de dicionários onde a chave `summary`
    é sobrescrita para conter o texto extraído diretamente do PDF do artigo.
    """
    # ===== FLAGS INTERNOS (Configurações da ferramenta) =====
    _INCLUDE_PDF = True       # Controla se o PDF deve ser processado
    _EXTRACT_TEXT = True      # Controla se o texto deve ser extraído do PDF
    _MAX_PAGES = 6            # Limita a extração às primeiras N páginas (economiza tempo/tokens)
    _TEXT_CHARS = 5000        # Limita o número de caracteres retornados do texto
    _SAVE_FULL_TEXT = False   # Se True, ignora _TEXT_CHARS e salva o texto completo
    _SLEEP_SECONDS = 1.0      # Pausa entre requisições para evitar rate limit do arXiv
    # ==========================

    # Constrói a URL da API do arXiv
    api_url = (
        "https://export.arxiv.org/api/query"
        f"?search_query=all:{requests.utils.quote(query)}&start=0&max_results={max_results}"
    )

    out: List[Dict] = []
    
    # Faz a requisição para a API
    try:
        resp = session.get(api_url, timeout=60)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        return [{"error": f"arXiv API request failed: {e}"}]

    # Analisa o XML retornado pelo arXiv
    try:
        root = ET.fromstring(resp.content)
        ns = {"atom": "http://www.w3.org/2005/Atom"} # Namespace do feed Atom

        for entry in root.findall("atom:entry", ns):
            # Extrai metadados básicos
            title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
            published = (entry.findtext("atom:published", default="", namespaces=ns) or "")[:10]
            url_abs = entry.findtext("atom:id", default="", namespaces=ns) or ""
            abstract_summary = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()

            # Extrai autores
            authors = []
            for a in entry.findall("atom:author", ns):
                nm = a.findtext("atom:name", default="", namespaces=ns)
                if nm:
                    authors.append(nm)

            # Encontra o link direto para o PDF
            link_pdf = None
            for link in entry.findall("atom:link", ns):
                if link.attrib.get("title") == "pdf":
                    link_pdf = link.attrib.get("href")
                    break
            if not link_pdf and url_abs:
                link_pdf = ensure_pdf_url(url_abs)

            # Monta o dicionário de resultado para este artigo
            item = {
                "title": title,
                "authors": authors,
                "published": published,
                "url": url_abs,
                "summary": abstract_summary, # Começa com o abstract, pode ser sobrescrito abaixo
                "link_pdf": link_pdf,
            }

            pdf_bytes = None
            # Baixa o PDF se as flags permitirem e o link existir
            if (_INCLUDE_PDF or _EXTRACT_TEXT) and link_pdf:
                try:
                    pdf_bytes = fetch_pdf_bytes(link_pdf, timeout=90)
                    time.sleep(_SLEEP_SECONDS) # Pausa educada entre downloads
                except Exception as e:
                    item["pdf_error"] = f"PDF fetch failed: {e}"

            # Extrai o texto do PDF baixado
            if _EXTRACT_TEXT and pdf_bytes:
                try:
                    text = pdf_bytes_to_text(pdf_bytes, max_pages=_MAX_PAGES)
                    text = clean_text(text) if text else ""
                    if text:
                        # Substitui o `summary` (abstract) pelo texto real do PDF
                        if _SAVE_FULL_TEXT:
                            item["summary"] = text
                        else:
                            item["summary"] = text[:_TEXT_CHARS]
                except Exception as e:
                    item["text_error"] = f"Text extraction failed: {e}"

            out.append(item)
        return out
        
    except ET.ParseError as e:
        return [{"error": f"arXiv API XML parse failed: {e}"}]
    except Exception as e:
        return [{"error": f"Unexpected error: {e}"}]

# Definição do schema da ferramenta arXiv para consumo do LLM
arxiv_tool_def = {
    "type": "function",
    "function": {
        "name": "arxiv_search_tool",
        "description": "Searches arXiv and (internally) fetches PDFs to memory and extracts text.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search keywords."},
                "max_results": {"type": "integer", "default": 3},
            },
            "required": ["query"],
        },
    },
}


# ----- Ferramenta de Pesquisa: Tavily -----
from dotenv import load_dotenv
from tavily import TavilyClient
load_dotenv()  # Carrega as variáveis de ambiente de um arquivo .env

def tavily_search_tool(
    query: str, max_results: int = 5, include_images: bool = False
) -> list[dict]:
    """
    Realiza uma pesquisa na web utilizando a API do Tavily (focada em IA).
    
    Args:
        query (str): A consulta de pesquisa.
        max_results (int): Número máximo de resultados.
        include_images (bool): Se True, inclui links para imagens relevantes.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY not found in environment variables.")

    # Inicializa o cliente Tavily (suporta URL base customizada via DLAI_TAVILY_BASE_URL)
    client = TavilyClient(api_key, api_base_url=os.getenv("DLAI_TAVILY_BASE_URL"))

    try:
        # Executa a pesquisa
        response = client.search(
            query=query, max_results=max_results, include_images=include_images
        )

        results = []
        # Formata os resultados retornados
        for r in response.get("results", []):
            results.append(
                {
                    "title": r.get("title", ""),
                    "content": r.get("content", ""),
                    "url": r.get("url", ""),
                }
            )

        # Adiciona imagens aos resultados se solicitado
        if include_images:
            for img_url in response.get("images", []):
                results.append({"image_url": img_url})

        return results

    except Exception as e:
        return [{"error": str(e)}]  # Retorna o erro em formato amigável para o LLM

# Definição do schema da ferramenta Tavily
tavily_tool_def = {
    "type": "function",
    "function": {
        "name": "tavily_search_tool",
        "description": "Performs a general-purpose web search using the Tavily API.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search keywords for retrieving information from the web.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return.",
                    "default": 5,
                },
                "include_images": {
                    "type": "boolean",
                    "description": "Whether to include image results.",
                    "default": False,
                },
            },
            "required": ["query"],
        },
    },
}


# ----- Ferramenta de Pesquisa: Wikipedia -----
def wikipedia_search_tool(query: str, sentences: int = 5) -> List[Dict]:
    """
    Busca na Wikipedia e retorna um resumo do artigo correspondente.

    Args:
        query (str): Termo de pesquisa para a Wikipedia.
        sentences (int): Número de frases para incluir no resumo retornado.
    """
    try:
        # Busca o título da página que melhor corresponde à query
        page_title = wikipedia.search(query)[0]
        # Carrega o objeto da página
        page = wikipedia.page(page_title)
        # Extrai um resumo com o número limitado de frases
        summary = wikipedia.summary(page_title, sentences=sentences)

        return [{"title": page.title, "summary": summary, "url": page.url}]
    except Exception as e:
        return [{"error": str(e)}]

# Definição do schema da ferramenta Wikipedia
wikipedia_tool_def = {
    "type": "function",
    "function": {
        "name": "wikipedia_search_tool",
        "description": "Searches for a Wikipedia article summary by query string.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search keywords for the Wikipedia article.",
                },
                "sentences": {
                    "type": "integer",
                    "description": "Number of sentences in the summary.",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
}

# ----- Mapeamento de Ferramentas (Tool Registry) -----

# Dicionário útil para conectar o nome da função (string) vindo do LLM
# à função real em Python que deve ser executada.
tool_mapping = {
    "tavily_search_tool": tavily_search_tool,
    "arxiv_search_tool": arxiv_search_tool,
    "wikipedia_search_tool": wikipedia_search_tool,
}