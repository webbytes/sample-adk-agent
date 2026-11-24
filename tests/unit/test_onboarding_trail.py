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

"""Unit and workflow tests for the Personal Onboarding Trail Agent."""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest
from google.adk import Agent
from google.adk.apps import App
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.genai import types

import app.tools as trail_tools
from app.agent import MODEL, REASONING_MODEL, RETRIEVAL_MODEL, root_agent
from app.observability import (
    TrailObservabilityPlugin,
    classify_message_modality,
    classify_user_intent,
    redact_pii_data,
    redact_pii_text,
    sanitize_url_pii,
)
from app.schemas import DEFAULT_CATEGORIES, JudgedAnswer
from app.tools import (
    CATEGORIES_STATE_KEY,
    PROFILE_STATE_KEY,
    TRAIL_STATE_KEY,
    confirm_category_proposal,
    list_trail_status,
    save_link_to_trail,
    save_screenshot_to_trail,
    search_trail_library,
    update_onboarding_profile,
)
from app.workflows import build_parallel_knowledge_workflow, classify_query_complexity


class ScriptedLlm(BaseLlm):
    """Deterministic fake LLM for offline unit testing of ADK agents and workflows."""

    def __init__(self, steps: list[types.Content]) -> None:
        super().__init__(model="scripted-fake-llm")
        self._steps = list(steps)

    async def generate_content_async(self, llm_request: Any, stream: bool = False):
        content = self._steps.pop(0)
        yield LlmResponse(
            content=content,
            usage_metadata=types.GenerateContentResponseUsageMetadata(
                prompt_token_count=42,
                candidates_token_count=18,
                total_token_count=60,
            ),
        )


def test_root_agent_wiring() -> None:
    """Verify root_agent keeps manifest contract, model tiers, and trail tools."""
    assert root_agent.name == "sample_adk_agent"
    assert RETRIEVAL_MODEL == "gemini-3.5-flash"
    assert MODEL == "gemini-3.8-flash"
    assert REASONING_MODEL == "gemini-3.1-pro-preview"
    assert len({RETRIEVAL_MODEL, MODEL, REASONING_MODEL}) == 3
    tool_names = { getattr(t, "name", getattr(t, "__name__", "")) for t in root_agent.tools }
    expected = {
        "save_link_to_trail",
        "save_screenshot_to_trail",
        "confirm_category_proposal",
        "search_trail_library",
        "update_onboarding_profile",
        "list_trail_status",
        "parallel_knowledge_lookup",
        "get_user_choice",
        "load_memory",
        "load_artifacts",
    }
    assert expected.issubset(tool_names)


def test_classify_message_modality() -> None:
    """Verify modality classifier distinguishes URLs, screenshots, and questions."""
    url_msg = types.Content(
        role="user",
        parts=[types.Part.from_text(text="  https://google.github.io/adk-docs/  ")],
    )
    assert classify_message_modality(url_msg) == "url"

    img_msg = types.Content(
        role="user",
        parts=[
            types.Part(
                inline_data=types.Blob(mime_type="image/png", data=b"\x89PNG\r\n")
            )
        ],
    )
    assert classify_message_modality(img_msg) == "screenshot"

    q_msg = types.Content(
        role="user",
        parts=[types.Part.from_text(text="How does VertexAiMemoryBankService work?")],
    )
    assert classify_message_modality(q_msg) == "text"


@pytest.mark.asyncio
async def test_link_capture_and_search(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify link capture stores user-scoped trail entries and ranks searches."""
    monkeypatch.setattr(
        trail_tools,
        "load_web_page",
        lambda url: (
            "Agent Development Kit State Scopes\n"
            "ADK supports four state prefixes: session, user:, app:, and temp: "
            "for cross-session state persistence."
        ),
    )

    llm = ScriptedLlm(
        [
            types.Content(
                role="model",
                parts=[
                    types.Part.from_function_call(
                        name="save_link_to_trail",
                        args={
                            "url": "https://google.github.io/adk-docs/sessions/state/",
                            "category": "adk-sdk",
                            "tags": ["state", "sessions"],
                        },
                    )
                ],
            ),
            types.Content(
                role="model",
                parts=[types.Part.from_text(text="Saved ADK State Scopes to adk-sdk.")],
            ),
            types.Content(
                role="model",
                parts=[
                    types.Part.from_function_call(
                        name="search_trail_library",
                        args={"query": "state prefixes persistence", "category": "adk-sdk"},
                    )
                ],
            ),
            types.Content(
                role="model",
                parts=[types.Part.from_text(text="Found 1 matching entry.")],
            ),
        ]
    )

    agent = Agent(
        name="test_trail_agent",
        model=llm,
        instruction="Save and search links.",
        tools=[save_link_to_trail, search_trail_library],
    )
    test_app = App(name="test_link_app", root_agent=agent, plugins=[TrailObservabilityPlugin()])
    runner = InMemoryRunner(app=test_app)
    session = await runner.session_service.create_session(
        app_name="test_link_app", user_id="u_alice"
    )

    _ = [
        e
        async for e in runner.run_async(
            user_id="u_alice",
            session_id=session.id,
            new_message=types.Content(
                role="user",
                parts=[types.Part.from_text(text="https://google.github.io/adk-docs/sessions/state/")],
            ),
        )
    ]

    after_save = await runner.session_service.get_session(
        app_name="test_link_app", user_id="u_alice", session_id=session.id
    )
    assert after_save is not None
    saved_items = after_save.state[TRAIL_STATE_KEY]
    assert len(saved_items) == 1
    assert saved_items[0]["category"] == "adk-sdk"
    assert saved_items[0]["source_type"] == "url"

    # Verify cross-session persistence via user: prefix in a brand-new session.
    session_2 = await runner.session_service.create_session(
        app_name="test_link_app", user_id="u_alice"
    )
    evs_2 = [
        e
        async for e in runner.run_async(
            user_id="u_alice",
            session_id=session_2.id,
            new_message=types.Content(
                role="user", parts=[types.Part.from_text(text="Find state prefixes")]
            ),
        )
    ]
    tool_responses = [
        p.function_response.response
        for e in evs_2
        if e.content
        for p in (e.content.parts or [])
        if p.function_response and p.function_response.name == "search_trail_library"
    ]
    assert tool_responses
    first_tool_resp = tool_responses[0]
    assert first_tool_resp is not None
    assert first_tool_resp["matched_count"] == 1
    assert (
        first_tool_resp["matches"][0]["url"]
        == "https://google.github.io/adk-docs/sessions/state/"
    )


@pytest.mark.asyncio
async def test_fetch_failure_triggers_screenshot_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify failed URL fetch returns deterministic screenshot fallback guidance."""
    monkeypatch.setattr(
        trail_tools,
        "load_web_page",
        lambda url: f"Failed to fetch url: {url}",
    )

    llm = ScriptedLlm(
        [
            types.Content(
                role="model",
                parts=[
                    types.Part.from_function_call(
                        name="save_link_to_trail",
                        args={"url": "https://example.com/spa-page", "category": "tutorials"},
                    )
                ],
            ),
            types.Content(
                role="model",
                parts=[
                    types.Part.from_text(
                        text="Fetch failed; please upload a full-window screenshot."
                    )
                ],
            ),
        ]
    )
    agent = Agent(
        name="fallback_agent",
        model=llm,
        instruction="Save links.",
        tools=[save_link_to_trail],
    )
    runner = InMemoryRunner(agent=agent, app_name="fallback_app")
    session = await runner.session_service.create_session(
        app_name="fallback_app", user_id="u1"
    )
    evs = [
        e
        async for e in runner.run_async(
            user_id="u1",
            session_id=session.id,
            new_message=types.Content(
                role="user", parts=[types.Part.from_text(text="https://example.com/spa-page")]
            ),
        )
    ]
    responses = [
        p.function_response.response
        for e in evs
        if e.content
        for p in (e.content.parts or [])
        if p.function_response
    ]
    assert responses[0]["status"] == "fetch_failed"
    assert "screenshot" in responses[0]["fallback_instruction"].lower()


@pytest.mark.asyncio
async def test_category_governance_and_screenshot_artifact() -> None:
    """Verify unapproved categories pause for approval and screenshots archive artifacts."""
    llm = ScriptedLlm(
        [
            # Attempt 1: tries to save under an unapproved category 'observability'
            types.Content(
                role="model",
                parts=[
                    types.Part.from_function_call(
                        name="save_screenshot_to_trail",
                        args={
                            "extracted_url": "https://github.com/google/adk-python/tree/main/plugins",
                            "page_title": "ADK Plugins & Tracing",
                            "content_summary": "Shows BasePlugin hooks and OpenTelemetry setup.",
                            "category": "observability",
                            "key_takeaways": ["Use BasePlugin for app-wide hooks"],
                            "url_was_truncated": True,
                        },
                    )
                ],
            ),
            types.Content(
                role="model",
                parts=[types.Part.from_text(text="Approve adding category 'observability'?")],
            ),
            # Turn 2: user approves, agent confirms category and saves screenshot
            types.Content(
                role="model",
                parts=[
                    types.Part.from_function_call(
                        name="confirm_category_proposal",
                        args={"category_name": "observability", "approved": True},
                    )
                ],
            ),
            types.Content(
                role="model",
                parts=[
                    types.Part.from_function_call(
                        name="save_screenshot_to_trail",
                        args={
                            "extracted_url": "https://github.com/google/adk-python/tree/main/plugins",
                            "page_title": "ADK Plugins & Tracing",
                            "content_summary": "Shows BasePlugin hooks and OpenTelemetry setup.",
                            "category": "observability",
                            "key_takeaways": ["Use BasePlugin for app-wide hooks"],
                            "url_was_truncated": True,
                        },
                    )
                ],
            ),
            types.Content(
                role="model",
                parts=[
                    types.Part.from_function_call(
                        name="update_onboarding_profile",
                        args={
                            "role": "Forward Deployed Engineer",
                            "focus_topics": ["ADK 2.0", "OpenTelemetry"],
                            "learning_goals": ["Deploy to Agent Runtime"],
                        },
                    )
                ],
            ),
            types.Content(
                role="model",
                parts=[
                    types.Part.from_function_call(
                        name="list_trail_status",
                        args={},
                    )
                ],
            ),
            types.Content(
                role="model",
                parts=[types.Part.from_text(text="Approved category and saved screenshot.")],
            ),
        ]
    )

    agent = Agent(
        name="gov_agent",
        model=llm,
        instruction="Capture screenshots with category governance.",
        tools=[
            save_screenshot_to_trail,
            confirm_category_proposal,
            update_onboarding_profile,
            list_trail_status,
        ],
    )
    test_app = App(
        name="gov_app", root_agent=agent, plugins=[TrailObservabilityPlugin()]
    )
    runner = InMemoryRunner(app=test_app)
    session = await runner.session_service.create_session(
        app_name="gov_app", user_id="u_bob"
    )

    png_msg = types.Content(
        role="user",
        parts=[
            types.Part(
                inline_data=types.Blob(
                    mime_type="image/png",
                    display_name="browser_window.png",
                    data=b"\x89PNG\r\n\x1a\nfakeimagebytes",
                )
            )
        ],
    )
    evs_turn_1 = [
        e
        async for e in runner.run_async(
            user_id="u_bob", session_id=session.id, new_message=png_msg
        )
    ]
    first_resp = next(
        p.function_response.response
        for e in evs_turn_1
        if e.content
        for p in (e.content.parts or [])
        if p.function_response
    )
    assert first_resp is not None
    assert first_resp["status"] == "category_approval_required"
    assert first_resp["proposed_category"] == "observability"

    # Artifact was archived in artifact_service by TrailObservabilityPlugin
    assert runner.artifact_service is not None
    artifact_keys = await runner.artifact_service.list_artifact_keys(
        app_name="gov_app", user_id="u_bob", session_id=session.id
    )
    assert any("browser_window.png" in k for k in artifact_keys)

    # User approves in turn 2
    _ = [
        e
        async for e in runner.run_async(
            user_id="u_bob",
            session_id=session.id,
            new_message=types.Content(
                role="user", parts=[types.Part.from_text(text="Yes, approve observability")]
            ),
        )
    ]
    final_session = await runner.session_service.get_session(
        app_name="gov_app", user_id="u_bob", session_id=session.id
    )
    assert final_session is not None
    assert "observability" in final_session.state[CATEGORIES_STATE_KEY]
    assert set(DEFAULT_CATEGORIES).issubset(set(final_session.state[CATEGORIES_STATE_KEY]))
    saved = final_session.state[TRAIL_STATE_KEY]
    assert len(saved) == 1
    assert saved[0]["source_type"] == "screenshot"
    assert saved[0]["url_was_truncated"] is True
    assert saved[0]["artifact_name"] is not None
    assert final_session.state[PROFILE_STATE_KEY]["role"] == "Forward Deployed Engineer"


def test_strategic_complexity_classifier() -> None:
    """Verify strategic router chooses fast Flash vs. deep Pro judge appropriately."""
    route_fast, reason_fast = classify_query_complexity(
        "What default port does adk web bind?",
        trail_match_count=0,
        memory_hit_count=0,
        has_web_candidate=True,
    )
    assert route_fast == "fast_judge"
    assert reason_fast == "single_source_factual_lookup"

    route_multi, reason_multi = classify_query_complexity(
        "What port does adk web bind?",
        trail_match_count=1,
        memory_hit_count=0,
        has_web_candidate=True,
    )
    assert route_multi == "pro_judge"
    assert reason_multi == "multi_source_reconciliation"

    route_arch, reason_arch = classify_query_complexity(
        "Compare the architecture trade-offs of Workflow vs sub_agents.",
        trail_match_count=0,
        memory_hit_count=0,
        has_web_candidate=True,
    )
    assert route_arch == "pro_judge"
    assert reason_arch == "complex_or_architectural_query"


@pytest.mark.asyncio
async def test_parallel_knowledge_workflow_and_judge() -> None:
    """Verify ADK 2.0 fan-out + JoinNode + strategic Pro/Flash judge routing."""
    web_fake = ScriptedLlm(
        [
            types.Content(
                role="model",
                parts=[
                    types.Part.from_text(
                        text="Public docs state JoinNode merges parallel fan-out branches."
                    )
                ],
            ),
            types.Content(
                role="model",
                parts=[
                    types.Part.from_text(
                        text="adk web listens on port 8000 by default."
                    )
                ],
            ),
        ]
    )
    pro_judged = JudgedAnswer(
        winning_source="trail_memory",
        confidence="high",
        rationale="The user's saved trail note directly covers ADK 2.0 JoinNode.",
        answer_markdown="Use `JoinNode` after a tuple fan-out edge to merge outputs.",
        citations=["https://google.github.io/adk-docs/workflows/parallel/"],
    )
    fast_judged = JudgedAnswer(
        winning_source="google_search",
        confidence="high",
        rationale="No saved trail entry matched; web search gives the default port.",
        answer_markdown="`adk web` binds to port `8000` by default.",
        citations=["https://google.github.io/adk-docs/get-started/"],
    )
    pro_judge_fake = ScriptedLlm(
        [
            types.Content(
                role="model",
                parts=[types.Part.from_text(text=json.dumps(pro_judged.model_dump()))],
            )
        ]
    )
    fast_judge_fake = ScriptedLlm(
        [
            types.Content(
                role="model",
                parts=[types.Part.from_text(text=json.dumps(fast_judged.model_dump()))],
            )
        ]
    )

    wf = build_parallel_knowledge_workflow(
        web_model=web_fake,
        fast_judge_model=fast_judge_fake,
        pro_judge_model=pro_judge_fake,
        include_google_search_tool=False,
    )

    coordinator_fake = ScriptedLlm(
        [
            types.Content(
                role="model",
                parts=[
                    types.Part.from_function_call(
                        name="parallel_knowledge_lookup",
                        args={"question": "How do I merge parallel branches in ADK 2.0?"},
                    )
                ],
            ),
            types.Content(
                role="model",
                parts=[
                    types.Part.from_text(
                        text=(
                            "Source: trail_memory (pro_judge). Use JoinNode "
                            "(https://google.github.io/adk-docs/workflows/parallel/)."
                        )
                    )
                ],
            ),
            types.Content(
                role="model",
                parts=[
                    types.Part.from_function_call(
                        name="parallel_knowledge_lookup",
                        args={"question": "What port does uvicorn bind?"},
                    )
                ],
            ),
            types.Content(
                role="model",
                parts=[types.Part.from_text(text="Port 8000 (fast_judge).")],
            ),
        ]
    )

    coordinator = Agent(
        name="coord",
        model=coordinator_fake,
        instruction="Use parallel_knowledge_lookup.",
        tools=[wf],
    )
    test_app = App(
        name="wf_app", root_agent=coordinator, plugins=[TrailObservabilityPlugin()]
    )
    runner = InMemoryRunner(app=test_app)
    session = await runner.session_service.create_session(
        app_name="wf_app",
        user_id="u_carol",
        state={
            TRAIL_STATE_KEY: [
                {
                    "entry_id": "abc123def456",
                    "url": "https://google.github.io/adk-docs/workflows/parallel/",
                    "title": "Parallel Fan-Out and JoinNode",
                    "summary": "JoinNode waits for all parallel predecessors.",
                    "content_excerpt": "JoinNode merges outputs keyed by predecessor name.",
                    "key_takeaways": ["Follow fan-out tuples with JoinNode"],
                    "category": "adk-sdk",
                    "tags": ["workflow", "joinnode"],
                    "source_type": "url",
                    "saved_at": "2026-11-24T00:00:00+00:00",
                }
            ]
        },
    )

    # Query 1: Matches both Trail Memory and Web -> escalates to 'pro_judge'
    events_1 = [
        e
        async for e in runner.run_async(
            user_id="u_carol",
            session_id=session.id,
            new_message=types.Content(
                role="user",
                parts=[types.Part.from_text(text="How do I merge parallel branches in ADK 2.0?")],
            ),
        )
    ]

    wf_responses_1 = [
        p.function_response.response
        for e in events_1
        if e.content
        for p in (e.content.parts or [])
        if p.function_response and p.function_response.name == "parallel_knowledge_lookup"
    ]
    assert wf_responses_1
    first_wf = wf_responses_1[0]
    assert first_wf is not None
    payload_1 = first_wf.get("result", first_wf)
    assert payload_1["winning_source"] == "trail_memory"
    assert payload_1["judge_tier"] == "pro_judge"
    assert "https://google.github.io/adk-docs/workflows/parallel/" in payload_1["citations"]

    # Query 2: Unmatched in Trail Memory + short factual query -> routes to 'fast_judge'
    events_2 = [
        e
        async for e in runner.run_async(
            user_id="u_carol",
            session_id=session.id,
            new_message=types.Content(
                role="user",
                parts=[types.Part.from_text(text="What port does uvicorn bind?")],
            ),
        )
    ]
    wf_responses_2 = [
        p.function_response.response
        for e in events_2
        if e.content
        for p in (e.content.parts or [])
        if p.function_response and p.function_response.name == "parallel_knowledge_lookup"
    ]
    assert wf_responses_2
    second_wf = wf_responses_2[0]
    assert second_wf is not None
    payload_2 = second_wf.get("result", second_wf)
    assert payload_2["winning_source"] == "google_search"
    assert payload_2["judge_tier"] == "fast_judge"


def test_pii_redaction_and_intent_classification() -> None:
    """Verify deterministic PII redaction across text, URLs, dicts, and intents."""
    raw = (
        "Contact jane.doe@example.com or 415-555-0199; SSN 123-45-6789; "
        "auth Bearer secretTokenValue123456."
    )
    scrubbed = redact_pii_text(raw)
    assert "jane.doe@example.com" not in scrubbed
    assert "415-555-0199" not in scrubbed
    assert "123-45-6789" not in scrubbed
    assert "secretTokenValue123456" not in scrubbed
    assert "[REDACTED_EMAIL]" in scrubbed
    assert "[REDACTED_PHONE]" in scrubbed
    assert "[REDACTED_SSN]" in scrubbed
    assert "[REDACTED_SECRET]" in scrubbed

    dirty_url = "https://user:pass@docs.example.com/guide?api_key=abc123&topic=adk"
    clean_url = sanitize_url_pii(dirty_url)
    assert "user:pass" not in clean_url
    assert "abc123" not in clean_url
    assert "topic=adk" in clean_url

    nested = redact_pii_data(
        {"authorization": "Bearer topsecret12345", "note": "Email bob@example.org"}
    )
    assert nested["authorization"] == "[REDACTED_SECRET]"
    assert "[REDACTED_EMAIL]" in nested["note"]

    url_turn = types.Content(
        role="user", parts=[types.Part.from_text(text="https://adk.dev")]
    )
    assert classify_user_intent(url_turn) == "capture_url_to_trail"


@pytest.mark.asyncio
async def test_pre_execution_intent_logs_and_pii_pipeline(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Verify pre-execution intent events precede outcomes and scrub PII end-to-end."""
    monkeypatch.setattr(
        trail_tools,
        "load_web_page",
        lambda url: (
            "ADK Auth Guide\n"
            "Maintainer email: maintainer@example.com and SSN 123-45-6789."
        ),
    )

    llm = ScriptedLlm(
        [
            types.Content(
                role="model",
                parts=[
                    types.Part.from_function_call(
                        name="save_link_to_trail",
                        args={
                            "url": "https://docs.example.com/auth?token=supersecret999",
                            "category": "security",
                        },
                    )
                ],
            ),
            types.Content(
                role="model",
                parts=[types.Part.from_text(text="Saved sanitized auth guide.")],
            ),
        ]
    )

    agent = Agent(
        name="pii_test_agent",
        model=llm,
        instruction="Save links.",
        tools=[save_link_to_trail],
    )
    test_app = App(
        name="pii_app", root_agent=agent, plugins=[TrailObservabilityPlugin()]
    )
    runner = InMemoryRunner(app=test_app)
    session = await runner.session_service.create_session(
        app_name="pii_app", user_id="u_pii"
    )

    with caplog.at_level(logging.INFO, logger="onboarding_trail.telemetry"):
        _ = [
            e
            async for e in runner.run_async(
                user_id="u_pii",
                session_id=session.id,
                new_message=types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(
                            text="Save https://docs.example.com/auth?token=supersecret999 for alice@example.com"
                        )
                    ],
                ),
            )
        ]

    records = [json.loads(r.message) for r in caplog.records]
    event_types = [r["event_type"] for r in records]
    # Intent events are logged BEFORE execution outcomes
    assert "user_intent_recorded" in event_types
    assert "agent_intent_recorded" in event_types
    assert "model_request_intent" in event_types
    assert "tool_intent_recorded" in event_types
    assert event_types.index("user_intent_recorded") < event_types.index(
        "tool_intent_recorded"
    )
    assert event_types.index("tool_intent_recorded") < event_types.index(
        "tool_completed"
    )

    # Zero raw PII or URL tokens appear anywhere in telemetry logs or stored state
    raw_log_blob = "\n".join(r.message for r in caplog.records)
    assert "alice@example.com" not in raw_log_blob
    assert "maintainer@example.com" not in raw_log_blob
    assert "supersecret999" not in raw_log_blob
    assert "123-45-6789" not in raw_log_blob

    stored = await runner.session_service.get_session(
        app_name="pii_app", user_id="u_pii", session_id=session.id
    )
    assert stored is not None
    entry = stored.state[TRAIL_STATE_KEY][0]
    assert "supersecret999" not in entry["url"]
    assert "maintainer@example.com" not in entry["content_excerpt"]
    assert "[REDACTED_EMAIL]" in entry["content_excerpt"]
