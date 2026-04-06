from datetime import datetime
from urllib import response
import json
import os
from aisuite import Client
from sqlalchemy.sql import true
from src.research_tools import (
    arxiv_search_tool,
    tavily_search_tool,
    wikipedia_search_tool,
)

def get_client():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not found in environment variables")
    return Client()

def key_word(
    prompt: str,
    model: str = "openai:gpt-4o-mini"
    ):
    """
    Extracts academic search keywords and short search phrases from a research prompt.

    Returns
    -------
    dict
        {
            "keywords": [...],
            "search_phrases": [...]
        }
    """
    client = get_client()

    # Prints a simple header in the console so you can easily identify
    # when this function starts running
    print("==================================")
    print(" Keyword Extraction Agent ")
    print("==================================")

    # Build the instruction that will be sent to the model.
    # The prompt is very explicit to increase the chance that the model
    # returns exactly the JSON format we need.
    full_prompt = f"""
        You are a research assistant.

        The user will provide a research topic.
        Your task is to generate:
        A list of short academic search phrases

        Rules:
        - Return only valid JSON
        - Use this exact structure:
        {{
            "search_phrases": ["..."]
        }}
        - Do not explain anything
        - Prefer English terms for academic databases
        - Keep search phrases concise and useful

        User research topic:
        {prompt}
    """.strip()

    # Create the messages payload in chat format.
    # Here we send only one message with role "user".
    messages = [{"role": "user", "content": full_prompt}]

    try:
        # Call the model using the provided client.
        # temperature=0.0 is used to make the output more deterministic,
        # which is useful when you expect structured JSON.
        print('')
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0
        )

        # Extract the text content returned by the model.
        # If content is None, fallback to an empty string.
        content = resp.choices[0].message.content or ""

        # Print the raw model output for debugging.
        # This is useful to inspect whether the model really returned valid JSON.
        print("Raw output:")
        print(content)

        # Convert the JSON string returned by the model into a Python dictionary.
        result = json.loads(content)

        # Get the "keywords" list from the parsed JSON.
        # If the key does not exist, use an empty list as default.
        # keywords = result.get("keywords", [])

        # Get the "search_phrases" list from the parsed JSON.
        # If the key does not exist, use an empty list as default.
        search_phrases = result.get("search_phrases", [])

        # Return a cleaned version of both lists:
        # - convert every item to string
        # - remove leading/trailing spaces
        # - ignore empty values
        search_phrases = [str(s).strip() for s in search_phrases if str(s).strip()]
        return {
            # "keywords": [str(k).strip() for k in keywords if str(k).strip()],
            "search_phrases": search_phrases,
            "url_to_query": " or ".join(search_phrases)
        }
    except Exception as e:
        print("❌ Error:", e)
        return {
            "search_phrases": [],
            "url_to_query": ""
        }

# === Research Agent ===
def research_agent(
    query: str,
    papers: list[dict],
    *,
    add_subjective: bool = True,
    subjective_model: str = "openai:gpt-4o-mini",
    subjective_max_pages: int = 2,
    subjective_max_chars: int = 6000,
    ):
    """
    Rank arXiv papers.

    This agent receives the output of `arxiv_search_tool` (a list of dicts) and returns the same
    list with three score categories added for each paper (0-10), plus an overall `rank_score`
    to make sorting easy.

    Scoring categories:
    - score_similarity: token overlap similarity to `query`
    - score_recency: newer papers score higher, based on `published` date
    - score_quality: metadata/abstract quality & completeness heuristic
    """
    client = get_client()

    print("==================================")
    print("Paper Ranking Agent (research_agent)")
    print("==================================")

    def _clamp_0_10(x: float) -> float:
        return float(max(0.0, min(10.0, x)))

    def _safe_str(v) -> str:
        return str(v).strip() if v is not None else ""

    def _published_to_recency_score(published: str) -> float:
        # Map age in years to a 0-10 score with a soft decay:
        # 0y->10, 1y->9, 2y->8, ... 8y->2, 10y+->0 (clamped)
        try:
            dt = datetime.strptime(published, "%Y-%m-%d")
            years = (datetime.now() - dt).days / 365.25
            return _clamp_0_10(10.0 - years)
        except Exception:
            return 0.0

    def _tokenize(text: str) -> set[str]:
        import re

        return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) >= 3}

    
    def _similarity_score(title: str, summary: str) -> float:
        # Pure token overlap similarity (no heuristics/keywords fallback).
        q = _tokenize(query or "")
        if not q:
            return 0.0
        doc = _tokenize(f"{title}\n{summary}")
        overlap = len(q & doc)
        return _clamp_0_10(10.0 * (overlap / max(1, len(q))))

    
    def _quality_score(paper: dict) -> float:
        title = _safe_str(paper.get("title"))
        summary = _safe_str(paper.get("summary"))
        url = _safe_str(paper.get("url"))
        authors = paper.get("authors") or []

        s = 0.0
        if len(title) >= 10:
            s += 2.0
        if len(summary) >= 400:
            s += 4.0
        elif len(summary) >= 150:
            s += 2.5
        if isinstance(authors, list) and len(authors) >= 1:
            s += 1.0
        if isinstance(authors, list) and len(authors) >= 3:
            s += 0.5
        if "arxiv.org/abs/" in url or "arxiv.org/pdf/" in url:
            s += 1.5
        return _clamp_0_10(s)

    
    def _llm_subjective_relevance_score(
        paper: dict,
        *,
        model: str = "openai:gpt-4o-mini",
        max_pages: int = 2,
        max_chars: int = 6000,
    ) -> dict:
        """
        Uses an LLM to provide a subjective relevance score (0-10) for a single paper
        given the research query, the paper summary, and the first pages of the PDF.

        Returns a dict:
          {
            "score_subjective": float,
            "rationale": str,
            "pdf_pages_used": int,
          }
        """
        if not isinstance(paper, dict):
            return {
                "score_subjective": 0.0,
                "rationale": "Invalid paper: expected dict",
                "pdf_pages_used": 0,
            }

        if "error" in paper:
            return {
                "score_subjective": 0.0,
                "rationale": f"Paper has error: {paper.get('error')}",
                "pdf_pages_used": 0,
            }

        title = str(paper.get("title") or "").strip()
        summary = str(paper.get("summary") or "").strip()
        pdf_url = (paper.get("link_pdf") or paper.get("pdf_url") or paper.get("url") or "").strip()

        pdf_text = ""
        pages_used = 0
        try:
            from src.research_tools import (
                ensure_pdf_url,
                fetch_pdf_bytes,
                pdf_bytes_to_text,
                clean_text,
            )

            if pdf_url:
                pdf_url = ensure_pdf_url(pdf_url)
                pdf_bytes = fetch_pdf_bytes(pdf_url, timeout=60)
                pdf_text = pdf_bytes_to_text(pdf_bytes, max_pages=max_pages) or ""
                pdf_text = clean_text(pdf_text) if pdf_text else ""
                pages_used = max_pages
        except Exception as e:
            pdf_text = f"[PDF extraction failed: {e}]"
            pages_used = 0

        if len(pdf_text) > max_chars:
            pdf_text = pdf_text[:max_chars]

        prompt = f"""
            You are scoring how relevant an academic paper is to a research query.

            Return ONLY valid JSON with this exact schema:
            {{
            "score_subjective": number,   // 0 to 10 (can be decimal)
            "rationale": string           // 1-3 short sentences
            }}

            Rules:
            - Base the score on the query fit, not writing quality.
            - Use the paper title, summary, and the first pages text provided.
            - If information is insufficient, score conservatively (<=5) and say why.

            RESEARCH QUERY:
            {query}

            PAPER TITLE:
            {title}

            PAPER SUMMARY:
            {summary}

            FIRST PAGES (TEXT EXTRACT):
            {pdf_text}
        """.strip()

        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        content = resp.choices[0].message.content or ""
        try:
            data = json.loads(content)
            score = float(data.get("score_subjective", 0.0))
            rationale = str(data.get("rationale", "")).strip()
            score = float(max(0.0, min(10.0, score)))
            return {
                "score_subjective": round(score, 2),
                "rationale": rationale,
                "pdf_pages_used": pages_used,
            }
        except Exception:
            return {
                "score_subjective": 0.0,
                "rationale": f"Invalid LLM JSON output: {content[:400]}",
                "pdf_pages_used": pages_used,
            }
    print('creating the final list')
    ranked: list[dict] = []
    for idx, paper in enumerate(papers or []):
        if not isinstance(paper, dict):
            ranked.append(
                {
                    "error": "Invalid paper entry: expected dict",
                    "original_index": idx,
                    "score_similarity": 0.0,
                    "score_recency": 0.0,
                    "score_quality": 0.0,
                    "score_subjective": 0.0,
                    "rationale": "",
                    "rank_score": 0.0,
                }
            )
            continue

        if "error" in paper:
            out = dict(paper)
            out["original_index"] = idx
            out["score_similarity"] = 0.0
            out["score_recency"] = 0.0
            out["score_quality"] = 0.0
            out["score_subjective"] = 0.0
            out["rationale"] = ""
            out["rank_score"] = 0.0
            ranked.append(out)
            continue

        model_output_subjective_relevance = _llm_subjective_relevance_score(
            paper,
            model=subjective_model,
            max_pages=subjective_max_pages,
            max_chars=subjective_max_chars,
        )

        title = _safe_str(paper.get("title"))
        summary = _safe_str(paper.get("summary"))
        published = _safe_str(paper.get("published"))

        score_similarity = _similarity_score(title, summary)
        score_recency = _published_to_recency_score(published)
        score_quality = _quality_score(paper)
        score_subj = model_output_subjective_relevance.get("score_subjective", 0.0)
        rank_score = _clamp_0_10(((score_similarity*2) + score_recency + score_quality+(score_subj*2)) / 4.0)

        out = dict(paper)
        out["original_index"] = idx
        out["score_similarity"] = round(score_similarity, 2)
        out["score_recency"] = round(score_recency, 2)
        out["score_quality"] = round(score_quality, 2)
        out["score_subjective"] = round(float(score_subj), 2)
        out["rationale"] = str(model_output_subjective_relevance.get("rationale", "")).strip()
        out["rank_score"] = round(rank_score, 2)

        ranked.append(out)

    ranked_sorted = sorted(ranked, key=lambda x: x["rank_score"], reverse=True)
    print(f"Ranked {len(ranked)} papers")
    return ranked_sorted