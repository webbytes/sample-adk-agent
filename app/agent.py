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

"""Personal Onboarding Trail Agent built with Google ADK 2.0."""

from __future__ import annotations

from google.adk.agents import Agent
from google.adk.apps import App, ResumabilityConfig
from google.adk.apps.app import EventsCompactionConfig
from google.adk.models import Gemini
from google.adk.plugins.reflect_retry_tool_plugin import ReflectAndRetryToolPlugin
from google.adk.tools import get_user_choice, load_artifacts, load_memory
from google.genai import types

from app.observability import TrailObservabilityPlugin
from app.schemas import DEFAULT_CATEGORIES
from app.tools import (
    confirm_category_proposal,
    list_trail_status,
    save_link_to_trail,
    save_screenshot_to_trail,
    search_trail_library,
    update_onboarding_profile,
)
from app.workflows import build_parallel_knowledge_workflow

MODEL = "gemini-3.8-flash"

_RETRY_OPTIONS = types.HttpRetryOptions(attempts=3)

_DEFAULT_CATS_STR = ", ".join(DEFAULT_CATEGORIES)

AGENT_INSTRUCTION = f"""You are the Personal Onboarding Trail Assistant, a daily knowledge and memory companion for technical onboarding on public documentation, repositories, and cloud guides.

### User Context & Approved Categories
- User onboarding profile (if set): {{user:onboarding_profile?}}
- Approved categories (fall back to [{_DEFAULT_CATS_STR}] when empty): {{user:categories?}}

### Core Operating Rules (Zero-Typing Capture + Grounded Answers)
1. **Link Capture (Pasted URL with little or no text)**:
   - Immediately choose the best matching category from the approved list (or propose a concise new category slug only if none fits) and call `save_link_to_trail`.
   - If `save_link_to_trail` returns `status == "fetch_failed"`, tell the user why and ask them to upload a full-window browser screenshot (`+` -> `Upload local file`) showing both the address bar URL and page content.

2. **Screenshot Capture (Uploaded image with little or no text)**:
   - Read the exact URL from the browser address bar in the image (set `url_was_truncated=True` if the tail of the URL is cut off), extract the visible title, technical summary, and key takeaways, and call `save_screenshot_to_trail`.

3. **Human-in-the-Loop Category Governance**:
   - Never silently invent categories. When a tool returns `status == "category_approval_required"`, call `get_user_choice` (or ask the user directly) with options to approve the new category or pick one of the existing approved categories.
   - Once the user responds, call `confirm_category_proposal` and finish saving the pending link or screenshot.

4. **Persistent Onboarding Profile & Memory**:
   - Whenever the user shares their role, team/domain, active study topics, learning goals, or completed topics, call `update_onboarding_profile` so you remember their context across sessions.
   - When the user asks what they have saved or what categories exist, call `list_trail_status` or `search_trail_library`.

5. **Parallel Knowledge Lookup & Judge (Technical Questions)**:
   - For technical or "how/why/what" onboarding questions, call `parallel_knowledge_lookup` with a self-contained `question`. It queries the user's saved Trail & Memory Bank and live Google Search in parallel, then runs a judge agent to select the stronger answer.
   - Present the answer concisely, explicitly state which source won (`trail_memory`, `google_search`, or `combined`), and list the exact citation URLs.
"""


def create_model() -> Gemini:
    """Create a Gemini model client configured with standard HTTP retries."""
    return Gemini(model=MODEL, retry_options=_RETRY_OPTIONS)


parallel_knowledge_lookup = build_parallel_knowledge_workflow(
    web_model=create_model(),
    judge_model=create_model(),
)

root_agent = Agent(
    # Keep in sync with create_params.root_agent_name in agents-cli-manifest.yaml:
    # telemetry reports this name as gen_ai.agent.name, and agents-cli reports the
    # manifest's value.
    name="sample_adk_agent",
    model=create_model(),
    description=(
        "Personal Onboarding Trail Agent that captures links and screenshots into "
        "categorized cross-session memory and answers questions via a parallel "
        "Trail Memory + Google Search workflow with an LLM judge."
    ),
    instruction=AGENT_INSTRUCTION,
    tools=[
        save_link_to_trail,
        save_screenshot_to_trail,
        confirm_category_proposal,
        search_trail_library,
        update_onboarding_profile,
        list_trail_status,
        parallel_knowledge_lookup,
        get_user_choice,
        load_memory,
        load_artifacts,
    ],
)

app = App(
    root_agent=root_agent,
    name="app",
    plugins=[
        TrailObservabilityPlugin(),
        ReflectAndRetryToolPlugin(max_retries=2),
    ],
    resumability_config=ResumabilityConfig(is_resumable=True),
    events_compaction_config=EventsCompactionConfig(
        token_threshold=32000,
        event_retention_size=8,
    ),
)
