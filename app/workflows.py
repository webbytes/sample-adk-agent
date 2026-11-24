# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ADK 2.0 Workflow graph for parallel Trail Memory + Google Search Q&A."""

from __future__ import annotations

import json
from typing import Any

from google.adk import Agent, Context, Workflow
from google.adk.models.base_llm import BaseLlm
from google.adk.tools import google_search
from google.adk.workflow import JoinNode
from google.genai import types

from app.observability import emit_structured_metric, tracer
from app.schemas import JudgedAnswer, KnowledgeQueryInput
from app.tools import PROFILE_STATE_KEY, TRAIL_STATE_KEY, rank_trail_entries


async def retrieve_from_trail_and_memory(
    ctx: Context, node_input: KnowledgeQueryInput
) -> dict[str, Any]:
    """Branch 1: Search the user's saved trail library and cross-session memory."""
    entries = list(ctx.state.get(TRAIL_STATE_KEY) or [])
    matches = rank_trail_entries(
        entries=entries,
        query=node_input.question,
        category_filter=node_input.category_filter,
        limit=5,
    )
    profile = ctx.state.get(PROFILE_STATE_KEY) or {}

    memory_snippets: list[str] = []
    try:
        mem_response = await ctx.search_memory(node_input.question)
        for mem in getattr(mem_response, "memories", []) or []:
            content = getattr(mem, "content", None)
            if content and getattr(content, "parts", None):
                text = " ".join(p.text for p in content.parts if p.text)
                if text.strip():
                    memory_snippets.append(text.strip()[:400])
    except Exception:
        # MemoryService is optional in standalone unit runners; degrade cleanly.
        memory_snippets = []

    compact_matches = [
        {
            "title": m.get("title"),
            "url": m.get("url"),
            "category": m.get("category"),
            "source_type": m.get("source_type"),
            "url_was_truncated": m.get("url_was_truncated", False),
            "summary": m.get("summary"),
            "key_takeaways": m.get("key_takeaways", []),
            "excerpt": str(m.get("content_excerpt", ""))[:900],
        }
        for m in matches
    ]

    with tracer.start_as_current_span(
        "onboarding_trail.workflow.trail_memory_branch"
    ) as span:
        span.set_attribute("onboarding_trail.trail_matches", len(compact_matches))
        span.set_attribute("onboarding_trail.memory_hits", len(memory_snippets))

    return {
        "question": node_input.question,
        "matched_count": len(compact_matches),
        "matches": compact_matches,
        "memory_snippets": memory_snippets[:5],
        "onboarding_profile": profile,
    }


def prepare_web_search_prompt(node_input: KnowledgeQueryInput) -> str:
    """Format the question for the isolated Google Search grounding sub-agent."""
    if node_input.category_filter.strip():
        return (
            f"Question (topic: {node_input.category_filter.strip()}): "
            f"{node_input.question}\n"
            "Provide an accurate technical answer with official public URLs."
        )
    return (
        f"Question: {node_input.question}\n"
        "Provide an accurate technical answer with official public URLs."
    )


def format_candidates_for_judge(node_input: dict[str, Any]) -> str:
    """Combine parallel Trail Memory and Google Search outputs for the Judge."""
    trail_payload = node_input.get("retrieve_from_trail_and_memory") or {}
    web_answer = node_input.get("web_grounding_agent") or ""
    return (
        "Evaluate the two parallel retrieval branches below and produce the "
        "best grounded answer for the user.\n\n"
        f"### User Question\n{trail_payload.get('question', '')}\n\n"
        "### Branch A: Personal Onboarding Trail & Cross-Session Memory\n"
        f"{json.dumps(trail_payload, indent=2, default=str)}\n\n"
        "### Branch B: Live Google Search Grounding\n"
        f"{web_answer}\n"
    )


def record_judge_telemetry(ctx: Context, node_input: JudgedAnswer) -> dict[str, Any]:
    """Record observability metrics for the judge's source selection."""
    result = node_input.model_dump()
    ctx.state["temp:last_judge_winner"] = node_input.winning_source
    with tracer.start_as_current_span(
        "onboarding_trail.workflow.judge_decision"
    ) as span:
        span.set_attribute("onboarding_trail.winning_source", node_input.winning_source)
        span.set_attribute("onboarding_trail.confidence", node_input.confidence)
        span.set_attribute("onboarding_trail.citation_count", len(node_input.citations))

    emit_structured_metric(
        "parallel_judge_completed",
        winning_source=node_input.winning_source,
        confidence=node_input.confidence,
        citation_count=len(node_input.citations),
    )
    return result


def build_parallel_knowledge_workflow(
    *,
    web_model: str | BaseLlm,
    judge_model: str | BaseLlm,
    include_google_search_tool: bool = True,
) -> Workflow:
    """Build the ADK 2.0 fan-out / JoinNode / judge Workflow for Q&A.

    Graph topology:
        START -> (retrieve_from_trail_and_memory, prepare_web_search_prompt)
        prepare_web_search_prompt -> web_grounding_agent
        (retrieve_from_trail_and_memory, web_grounding_agent) -> join_knowledge_sources
        join_knowledge_sources -> format_candidates_for_judge -> answer_judge_agent -> record_judge_telemetry
    """
    # Isolated search agent prevents mixing google_search with function tools
    # on root_agent (which triggers Gemini 400 INVALID_ARGUMENT).
    web_grounding_agent = Agent(
        name="web_grounding_agent",
        model=web_model,
        mode="single_turn",
        description="Answers technical questions grounded in live Google Search.",
        instruction=(
            "You are a technical documentation researcher. Use Google Search to "
            "find authoritative public documentation, GitHub repositories, and "
            "official guides that answer the question. Include concrete URLs "
            "supporting every claim."
        ),
        tools=[google_search] if include_google_search_tool else [],
    )

    answer_judge_agent = Agent(
        name="answer_judge_agent",
        model=judge_model,
        mode="single_turn",
        output_schema=JudgedAnswer,
        generate_content_config=types.GenerateContentConfig(temperature=0.0),
        description="Compares Trail Memory vs. Google Search and selects the best answer.",
        instruction=(
            "You are an impartial technical answer judge for an engineer's "
            "Personal Onboarding Trail Assistant.\n"
            "1. Inspect Branch A (Personal Trail & Memory) and Branch B (Google Search).\n"
            "2. If Branch A contains saved links or screenshots that directly "
            "answer the question, prefer 'trail_memory' (or 'combined' when "
            "Branch B adds essential missing detail) and cite the exact saved URLs "
            "(noting if a screenshot URL was marked truncated).\n"
            "3. If Branch A has zero relevant matches, select 'google_search' and "
            "cite the authoritative public URLs from Branch B.\n"
            "4. Tailor explanations to the user's onboarding_profile when present."
        ),
    )

    join_sources = JoinNode(name="join_knowledge_sources")

    return Workflow(
        name="parallel_knowledge_lookup",
        description=(
            "Answers technical or onboarding questions by querying the user's "
            "saved trail memory and live Google Search in parallel, then using "
            "a judge agent to pick or combine the stronger cited answer."
        ),
        input_schema=KnowledgeQueryInput,
        edges=[
            ("START", (retrieve_from_trail_and_memory, prepare_web_search_prompt)),
            (prepare_web_search_prompt, web_grounding_agent),
            ((retrieve_from_trail_and_memory, web_grounding_agent), join_sources),
            (
                join_sources,
                format_candidates_for_judge,
                answer_judge_agent,
                record_judge_telemetry,
            ),
        ],
    )
