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

"""ADK 2.0 Workflow graph with parallel retrieval and strategic model routing."""

from __future__ import annotations

import json
import re
from typing import Any

from google.adk import Agent, Context, Workflow
from google.adk.events import Event, EventActions
from google.adk.models.base_llm import BaseLlm
from google.adk.tools import google_search
from google.adk.workflow import JoinNode
from google.genai import types

from app.observability import emit_structured_metric, tracer
from app.schemas import JudgedAnswer, KnowledgeQueryInput
from app.tools import PROFILE_STATE_KEY, TRAIL_STATE_KEY, rank_trail_entries

_COMPLEX_QUERY_RE = re.compile(
    r"\b(compare|comparison|versus|vs\.?|trade-?offs?|architect\w*|design|"
    r"difference|differ|why|debug\w*|orchestrat\w*|multi-?agent|security)\b",
    re.IGNORECASE,
)

_JUDGE_INSTRUCTION = (
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
)


def classify_query_complexity(
    question: str,
    *,
    trail_match_count: int,
    memory_hit_count: int,
    has_web_candidate: bool,
) -> tuple[str, str]:
    """Select 'pro_judge' or 'fast_judge' based on query & evidence complexity.

    Strategic model routing rules:
      - Route to 'pro_judge' (Pro reasoning model) when reconciling multiple
        substantive knowledge sources (personal trail/memory + web search) or
        when the question requires architectural/comparative reasoning.
      - Route to 'fast_judge' (Flash low-latency model) for concise, single-source
        factual lookups to minimize latency and token cost.
    """
    has_personal_context = (trail_match_count + memory_hit_count) > 0
    if has_personal_context and has_web_candidate:
        return "pro_judge", "multi_source_reconciliation"

    word_count = len(question.split())
    if word_count >= 18 or bool(_COMPLEX_QUERY_RE.search(question)):
        return "pro_judge", "complex_or_architectural_query"

    return "fast_judge", "single_source_factual_lookup"


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


def route_judge_by_complexity(ctx: Context, node_input: dict[str, Any]) -> Event:
    """Format candidate answers and dynamically route between Flash and Pro judges."""
    trail_payload = node_input.get("retrieve_from_trail_and_memory") or {}
    web_answer = str(node_input.get("web_grounding_agent") or "")
    question = str(trail_payload.get("question", ""))
    matched_count = int(trail_payload.get("matched_count", 0) or 0)
    memory_hits = len(trail_payload.get("memory_snippets") or [])

    route, reason = classify_query_complexity(
        question,
        trail_match_count=matched_count,
        memory_hit_count=memory_hits,
        has_web_candidate=bool(web_answer.strip()),
    )

    with tracer.start_as_current_span(
        "onboarding_trail.workflow.model_router"
    ) as span:
        span.set_attribute("onboarding_trail.judge_route", route)
        span.set_attribute("onboarding_trail.routing_reason", reason)
        span.set_attribute("onboarding_trail.trail_matches", matched_count)

    emit_structured_metric(
        "model_route_selected",
        judge_route=route,
        routing_reason=reason,
        trail_matches=matched_count,
        memory_hits=memory_hits,
    )

    prompt = (
        "Evaluate the two parallel retrieval branches below and produce the "
        "best grounded answer for the user.\n\n"
        f"### User Question\n{question}\n\n"
        "### Branch A: Personal Onboarding Trail & Cross-Session Memory\n"
        f"{json.dumps(trail_payload, indent=2, default=str)}\n\n"
        "### Branch B: Live Google Search Grounding\n"
        f"{web_answer}\n"
    )
    return Event(
        output=prompt,
        actions=EventActions(
            route=route,
            state_delta={
                "temp:selected_judge_tier": route,
                "temp:routing_reason": reason,
            },
        ),
    )


def record_judge_telemetry(ctx: Context, node_input: JudgedAnswer) -> dict[str, Any]:
    """Record observability metrics for the judge's source and model-tier selection."""
    selected_tier = str(ctx.state.get("temp:selected_judge_tier") or "fast_judge")
    result = node_input.model_dump()
    result["judge_tier"] = selected_tier
    ctx.state["temp:last_judge_winner"] = node_input.winning_source

    with tracer.start_as_current_span(
        "onboarding_trail.workflow.judge_decision"
    ) as span:
        span.set_attribute("onboarding_trail.winning_source", node_input.winning_source)
        span.set_attribute("onboarding_trail.judge_tier", selected_tier)
        span.set_attribute("onboarding_trail.confidence", node_input.confidence)
        span.set_attribute("onboarding_trail.citation_count", len(node_input.citations))

    emit_structured_metric(
        "parallel_judge_completed",
        winning_source=node_input.winning_source,
        judge_tier=selected_tier,
        confidence=node_input.confidence,
        citation_count=len(node_input.citations),
    )
    return result


def build_parallel_knowledge_workflow(
    *,
    web_model: str | BaseLlm,
    fast_judge_model: str | BaseLlm,
    pro_judge_model: str | BaseLlm,
    include_google_search_tool: bool = True,
) -> Workflow:
    """Build the ADK 2.0 fan-out / JoinNode / strategic-model-routed Judge Workflow.

    Graph topology:
        START -> (retrieve_from_trail_and_memory, prepare_web_search_prompt)
        prepare_web_search_prompt -> web_grounding_agent [fast retrieval tier]
        (retrieve_from_trail_and_memory, web_grounding_agent) -> join_knowledge_sources
        join_knowledge_sources -> route_judge_by_complexity
        route_judge_by_complexity -> {
            'fast_judge': fast_answer_judge_agent [Flash tier],
            'pro_judge': pro_answer_judge_agent [Pro reasoning tier],
        }
        fast_answer_judge_agent -> record_judge_telemetry
        pro_answer_judge_agent -> record_judge_telemetry
    """
    # Tier 1 (Fast Retrieval): lightweight Flash model isolated with google_search
    # so search never collides with Python function tools on root_agent.
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

    # Tier 2 (Fast Judge): Flash model for low-latency single-source factual lookups.
    fast_answer_judge_agent = Agent(
        name="fast_answer_judge_agent",
        model=fast_judge_model,
        mode="single_turn",
        output_schema=JudgedAnswer,
        generate_content_config=types.GenerateContentConfig(temperature=0.0),
        description="Low-latency Flash judge for straightforward single-source answers.",
        instruction=_JUDGE_INSTRUCTION,
    )

    # Tier 3 (Pro Reasoning Judge): Pro model for multi-source synthesis and
    # complex architectural or comparative questions.
    pro_answer_judge_agent = Agent(
        name="pro_answer_judge_agent",
        model=pro_judge_model,
        mode="single_turn",
        output_schema=JudgedAnswer,
        generate_content_config=types.GenerateContentConfig(temperature=0.0),
        description="Deep-reasoning Pro judge for multi-source reconciliation and complex queries.",
        instruction=_JUDGE_INSTRUCTION,
    )

    join_sources = JoinNode(name="join_knowledge_sources")

    return Workflow(
        name="parallel_knowledge_lookup",
        description=(
            "Answers technical or onboarding questions by querying the user's "
            "saved trail memory and live Google Search in parallel, then routing "
            "by complexity to either a low-latency Flash judge or a deep-reasoning "
            "Pro judge to return the strongest cited answer."
        ),
        input_schema=KnowledgeQueryInput,
        edges=[
            ("START", (retrieve_from_trail_and_memory, prepare_web_search_prompt)),
            (prepare_web_search_prompt, web_grounding_agent),
            ((retrieve_from_trail_and_memory, web_grounding_agent), join_sources),
            (join_sources, route_judge_by_complexity),
            (
                route_judge_by_complexity,
                {
                    "fast_judge": fast_answer_judge_agent,
                    "pro_judge": pro_answer_judge_agent,
                },
            ),
            (fast_answer_judge_agent, record_judge_telemetry),
            (pro_answer_judge_agent, record_judge_telemetry),
        ],
    )
