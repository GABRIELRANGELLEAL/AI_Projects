"""
Test script for the planner_agent function.
This allows you to test the planner agent independently without running the full FastAPI application.
"""

import os
from dotenv import load_dotenv
from aisuite import Client
from typing import List
import ast

# Load environment variables (if you have a .env file)
load_dotenv()

# Explicitly get and pass the API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

client = Client()

def planner_agent(topic: str, model: str = "openai:o4-mini") -> List[str]:
    """
    Generates a step-by-step research plan for a given topic.
    
    This function uses an AI model to break down a research task into actionable steps
    that will be executed by different agents (Research, Writer, Editor).
    
    Args:
        topic: The research topic/prompt provided by the user
        model: The AI model to use for planning (default: "openai:o4-mini")
    
    Returns:
        A list of strings, where each string is a step description to be executed
    """
    
    # Construct the prompt that instructs the AI to create a research plan
    prompt = f"""
        You are a planning agent responsible for organizing a research workflow using multiple intelligent agents.

        Available agents:
        - Research agent: MUST begin with a broad **web search using Tavily** to identify only **relevant** and **authoritative** items (e.g., high-impact venues, seminal works, surveys, or recent comprehensive sources). The output of this step MUST capture for each candidate: title, authors, year, venue/source, URL, and (if available) DOI.
        - Research agent: AFTER the Tavily step, perform a **targeted arXiv search** ONLY for the candidates discovered in the web step (match by title/author/DOI). If an arXiv preprint/version exists, record its arXiv URL and version info. Do NOT run a generic arXiv search detached from the Tavily results.
        - Writer agent: drafts based on research findings.
        - Editor agent: reviews, reflects on, and improves drafts.

        Produce a clear step-by-step research plan **as a valid Python list of strings** (no markdown, no explanations). 
        Each step must be atomic, actionable, and assigned to one of the agents.
        Maximum of 7 steps.

        DO NOT include steps like "create CSV", "set up repo", "install packages".
            - Focus on meaningful research tasks (search, extract, rank, draft, revise).
            - The FIRST step MUST be exactly: "Research agent: Use Tavily to perform a broad web search and collect top relevant items (title, authors, year, venue/source, URL, DOI if available)."
            - The SECOND step MUST be exactly: "Research agent: For each collected item, search on arXiv to find matching preprints/versions and record arXiv URLs (if they exist)."

        The FINAL step MUST instruct the writer agent to generate a comprehensive Markdown report that:
        - Uses all findings and outputs from previous steps
        - Includes inline citations (e.g., [1], (Wikipedia/arXiv))
        - Includes a References section with clickable links for all citations
        - Preserves earlier sources
        - Is detailed and self-contained

        Topic: "{topic}"
    """

    # Call the AI model to generate the plan
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=1,  # High temperature for more creative/diverse planning
    )

    # Extract the raw response text from the AI
    raw = response.choices[0].message.content.strip()

print("🔄 Generating plan...")
steps = planner_agent(topic)
print(f"\n✅ Plan generated successfully!")


if __name__ == "__main__":
    # Test with a sample topic
    # You can modify this topic or pass it as a command-line argument
    
    # Example topics to test:
    test_topics = [
        "Recent advances in large language models"
    ]
    
    # Test with the first topic (or modify to test others)
    topic = test_topics[0]
    
    # Or uncomment to use command-line argument:
    # import sys
    # if len(sys.argv) > 1:
    #     topic = " ".join(sys.argv[1:])
    # else:
    #     topic = test_topics[0]
    
    test_planner_agent(topic)

