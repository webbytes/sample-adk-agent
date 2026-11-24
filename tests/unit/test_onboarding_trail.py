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
from typing import Any

import pytest
from google.adk import Agent
from google.adk.apps import App
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.genai import types

import app.tools as trail_tools
from app.agent import root_agent
from app.observability import TrailObservabilityPlugin, classify_message_modality
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
from app.workflows import build_parallel_knowledge_workflow


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
    """Verify root_agent keeps manifest contract and exposes all trail tools."""
    assert root_agent.name == "sample_adk_agent"
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


@pytest.mark.asyncio
async def test_parallel_knowledge_workflow_and_judge() -> None:
    """Verify ADK 2.0 fan-out + JoinNode + Judge workflow executes both branches."""
    web_fake = ScriptedLlm(
        [
            types.Content(
                role="model",
                parts=[
                    types.Part.from_text(
                        text="Public docs state JoinNode merges parallel fan-out branches."
                    )
                ],
            )
        ]
    )
    judged = JudgedAnswer(
        winning_source="trail_memory",
        confidence="high",
        rationale="The user's saved trail note directly covers ADK 2.0 JoinNode.",
        answer_markdown="Use `JoinNode` after a tuple fan-out edge to merge outputs.",
        citations=["https://google.github.io/adk-docs/workflows/parallel/"],
    )
    judge_fake = ScriptedLlm(
        [
            types.Content(
                role="model",
                parts=[types.Part.from_text(text=json.dumps(judged.model_dump()))],
            )
        ]
    )

    wf = build_parallel_knowledge_workflow(
        web_model=web_fake,
        judge_model=judge_fake,
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
                            "Source: trail_memory. Use JoinNode "
                            "(https://google.github.io/adk-docs/workflows/parallel/)."
                        )
                    )
                ],
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

    events = [
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

    wf_responses = [
        p.function_response.response
        for e in events
        if e.content
        for p in (e.content.parts or [])
        if p.function_response and p.function_response.name == "parallel_knowledge_lookup"
    ]
    assert wf_responses
    first_wf = wf_responses[0]
    assert first_wf is not None
    payload = first_wf.get("result", first_wf)
    assert payload["winning_source"] == "trail_memory"
    assert "https://google.github.io/adk-docs/workflows/parallel/" in payload["citations"]
