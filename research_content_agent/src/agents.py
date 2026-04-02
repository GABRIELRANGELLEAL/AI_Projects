from datetime import datetime
from urllib import response
import json
from aisuite import Client
from sqlalchemy.sql import true
from src.research_tools import (
    arxiv_search_tool,
    tavily_search_tool,
    wikipedia_search_tool,
)

client = Client()

def key_word(
    prompt: str,
    model: str = "openai:gpt-4.1-mini"
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
        # If any error happens (API error, invalid JSON, etc.),
        # print the error and return empty lists so the function
        # does not break the rest of your pipeline.
        print("❌ Error:", e)

# === Research Agent ===
def research_agent(
    query: str,
    papers: list[dict],
    *,
    add_subjective: bool = true,
    subjective_model: str = "openai:gpt-4.1-mini",
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
        model: str = "openai:gpt-4.1-mini",
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

        title = _safe_str(paper.get("title"))
        summary = _safe_str(paper.get("summary"))
        published = _safe_str(paper.get("published"))

        score_similarity = _similarity_score(title, summary)
        score_recency = _published_to_recency_score(published)
        score_quality = _quality_score(paper)
        rank_score = _clamp_0_10((score_similarity + score_recency + score_quality) / 3.0)

        out = dict(paper)
        out["original_index"] = idx
        out["score_similarity"] = round(score_similarity, 2)
        out["score_recency"] = round(score_recency, 2)
        out["score_quality"] = round(score_quality, 2)
        out["score_subjective"] = 0.0
        out["rationale"] = ""
        out["rank_score"] = round(rank_score, 2)

        if add_subjective:
            subj = _llm_subjective_relevance_score(
                out,
                model=subjective_model,
                max_pages=subjective_max_pages,
                max_chars=subjective_max_chars,
            )
            out["score_subjective"] = float(subj.get("score_subjective", 0.0))
            out["rationale"] = str(subj.get("rationale", "")).strip()

        ranked.append(out)

    print(f"Ranked {len(ranked)} papers")
    return ranked

def writer_agent(
    prompt: str,
    model: str = "openai:gpt-4.1-mini",
    min_words_total: int = 2400,
    min_words_per_section: int = 400,
    max_tokens: int = 15000,
    retries: int = 1,
):
    print("==================================")
    print("Writer Agent")
    print("==================================")

    system_message = """
        You are an expert academic writer with a PhD-level understanding of scholarly communication. 
        Your task is to synthesize research materials into a comprehensive, well-structured academic report.

        ## REPORT REQUIREMENTS:
        - Produce a COMPLETE, POLISHED, and PUBLICATION-READY academic report in Markdown format
        - Create original content that thoroughly analyzes the provided research materials
        - DO NOT merely summarize the sources; develop a cohesive narrative with critical analysis
        - Length should be appropriate to thoroughly cover the topic (typically 1500-3000 words)

        ## MANDATORY STRUCTURE:
        1. **Title**: Clear, concise, and descriptive of the content
        2. **Abstract**: Brief summary (100-150 words) of the report's purpose, methods, and key findings
        3. **Introduction**: Present the topic, research question/problem, significance, and outline of the report
        4. **Background/Literature Review**: Contextualize the topic within existing scholarship
        5. **Methodology**: If applicable, describe research methods, data collection, and analytical approaches
        6. **Key Findings/Results**: Present the primary outcomes and evidence
        7. **Discussion**: Interpret findings, address implications, limitations, and connections to broader field
        8. **Conclusion**: Synthesize main points and suggest directions for future research
        9. **References**: Complete list of all cited works

        ## ACADEMIC WRITING GUIDELINES:
        - Maintain formal, precise, and objective language throughout
        - Use discipline-appropriate terminology and concepts
        - Support all claims with evidence and reasoning
        - Develop logical flow between ideas, paragraphs, and sections
        - Include relevant examples, case studies, data, or equations to strengthen arguments
        - Address potential counterarguments and limitations

        ## CITATION AND REFERENCE RULES:
        - Use numeric inline citations [1], [2], etc. for all borrowed ideas and information
        - Every claim based on external sources MUST have a citation
        - Each inline citation must correspond to a complete entry in the References section
        - Every reference listed must be cited at least once in the text
        - Preserve ALL original URLs, DOIs, and bibliographic information from source materials
        - Format references consistently according to academic standards

        ## FORMATTING GUIDELINES:
        - Use Markdown syntax for all formatting (headings, emphasis, lists, etc.)
        - Include appropriate section headings and subheadings to organize content
        - Format any equations, tables, or figures according to academic conventions
        - Use bullet points or numbered lists when appropriate for clarity
        - Use html syntax to handle all links with target="_blank", so user can always open link in new tab on both html and markdown format

        Output the complete report in Markdown format only. Do not include meta-commentary about the writing process.

        INTERNAL CHECKLIST (DO NOT INCLUDE IN OUTPUT):
        - [ ] Incorporated all provided research materials
        - [ ] Developed original analysis beyond mere summarization
        - [ ] Included all mandatory sections with appropriate content
        - [ ] Used proper inline citations for all borrowed content
        - [ ] Created complete References section with all cited sources
        - [ ] Maintained academic tone and language throughout
        - [ ] Ensured logical flow and coherent structure
        - [ ] Preserved all source URLs and bibliographic information
    """.strip()

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]

    def _call(messages_):
        resp = client.chat.completions.create(
            model=model,
            messages=messages_,
            temperature=0,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    def _word_count(md_text: str) -> int:
        import re

        words = re.findall(r"\b\w+\b", md_text)
        return len(words)

    content = _call(messages)

    print("Output:\n", content)
    return content, messages


def editor_agent(
    prompt: str,
    model: str = "openai:gpt-4.1-mini",
    target_min_words: int = 2400,
):
    print("==================================")
    print("Editor Agent")
    print("==================================")

    system_message = """
        You are a professional academic editor with expertise in improving scholarly writing across disciplines. 
        Your task is to refine and elevate the quality of the academic text provided.

        ## Your Editing Process:
        1. Analyze the overall structure, argument flow, and coherence of the text
        2. Ensure logical progression of ideas with clear topic sentences and transitions between paragraphs
        3. Improve clarity, precision, and conciseness of language while maintaining academic tone
        4. Verify technical accuracy (to the extent possible based on context)
        5. Enhance readability through appropriate formatting and organization

        ## Specific Elements to Address:
        - Strengthen thesis statements and main arguments
        - Clarify complex concepts with additional explanations or examples where needed
        - Add relevant equations, diagrams, or illustrations (described in markdown) when they would enhance understanding
        - Ensure proper integration of evidence and maintain academic rigor
        - Standardize terminology and eliminate redundancies
        - Improve sentence variety and paragraph structure
        - Preserve all citations [1], [2], etc., and maintain the integrity of the References section

        ## Formatting Guidelines:
        - Use markdown formatting consistently for headings, emphasis, lists, etc.
        - Structure content with appropriate section headings and subheadings
        - Format equations, tables, and figures according to academic standards

        Return only the revised, polished text in Markdown format without explanatory comments about your edits.
    """.strip()

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]

    response = client.chat.completions.create(
        model=model, messages=messages, temperature=0
    )

    content = response.choices[0].message.content
    print("Output:\n", content)
    return content, messages