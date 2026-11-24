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

"""OpenTelemetry tracing and ADK BasePlugin observability for Onboarding Trail."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import TYPE_CHECKING, Any

from google.adk.plugins.base_plugin import BasePlugin
from google.genai import types
from opentelemetry import trace

if TYPE_CHECKING:
    from google.adk.agents.base_agent import BaseAgent
    from google.adk.agents.callback_context import CallbackContext
    from google.adk.agents.invocation_context import InvocationContext
    from google.adk.models.llm_response import LlmResponse
    from google.adk.tools.base_tool import BaseTool
    from google.adk.tools.tool_context import ToolContext

telemetry_logger = logging.getLogger("onboarding_trail.telemetry")
tracer = trace.get_tracer("onboarding_trail.agent", "0.1.0")

_URL_ONLY_RE = re.compile(r"^\s*https?://\S+\s*$", re.IGNORECASE)


def classify_message_modality(user_message: types.Content | None) -> str:
    """Classify an incoming user message as 'screenshot', 'url', or 'text'."""
    if not user_message or not user_message.parts:
        return "empty"
    has_image = any(
        part.inline_data is not None
        and (part.inline_data.mime_type or "").startswith("image/")
        for part in user_message.parts
    )
    if has_image:
        return "screenshot"
    texts = [part.text.strip() for part in user_message.parts if part.text]
    combined = " ".join(t for t in texts if t)
    if combined and _URL_ONLY_RE.match(combined):
        return "url"
    return "text"


def emit_structured_metric(event_type: str, **fields: Any) -> None:
    """Emit a single-line JSON structured telemetry record for Cloud Logging."""
    payload = {"event_type": event_type, **fields}
    telemetry_logger.info(json.dumps(payload, default=str, sort_keys=True))


class TrailObservabilityPlugin(BasePlugin):
    """App-wide observability, screenshot artifact archival, and Memory Bank sync.

    Designed as a stateless singleton: never mutates ``self.*`` with per-session
    data; all per-turn state lives in ``temp:`` keys on the ADK context.
    """

    def __init__(self, name: str = "trail_observability_plugin") -> None:
        super().__init__(name=name)

    async def on_user_message_callback(
        self,
        *,
        invocation_context: InvocationContext,
        user_message: types.Content,
    ) -> types.Content | None:
        """Detect input modality and archive uploaded screenshots as artifacts.

        Unlike SaveFilesAsArtifactsPlugin, this preserves inline_data on the
        current user_message so Gemini can inspect the screenshot in a single
        turn while also persisting a durable artifact copy.
        """
        modality = classify_message_modality(user_message)
        archived_artifact: str | None = None

        if modality == "screenshot" and invocation_context.artifact_service:
            for idx, part in enumerate(user_message.parts or []):
                blob = part.inline_data
                if blob is None or not (blob.mime_type or "").startswith("image/"):
                    continue
                raw_name = blob.display_name or f"screenshot_{idx}.png"
                safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", raw_name)
                artifact_key = (
                    f"user:screenshot_{invocation_context.invocation_id}_{safe_name}"
                )
                try:
                    await invocation_context.artifact_service.save_artifact(
                        app_name=invocation_context.app_name,
                        user_id=invocation_context.user_id,
                        session_id=invocation_context.session.id,
                        filename=artifact_key,
                        artifact=types.Part(inline_data=blob),
                    )
                    archived_artifact = artifact_key
                except Exception as exc:  # pragma: no cover - defensive telemetry
                    telemetry_logger.warning(
                        "Failed to archive screenshot artifact: %s", exc
                    )

        with tracer.start_as_current_span("onboarding_trail.user_message") as span:
            span.set_attribute("onboarding_trail.input_modality", modality)
            span.set_attribute(
                "onboarding_trail.invocation_id", invocation_context.invocation_id
            )
            if archived_artifact:
                span.set_attribute(
                    "onboarding_trail.archived_artifact", archived_artifact
                )

        # Stage metadata on session state; before_agent_callback persists the
        # latest uploaded screenshot key into session-scoped state_delta so it
        # survives multi-turn category approval.
        invocation_context.session.state["temp:input_modality"] = modality
        if archived_artifact:
            invocation_context.session.state["temp:staged_artifact"] = archived_artifact

        emit_structured_metric(
            "user_message_received",
            invocation_id=invocation_context.invocation_id,
            modality=modality,
            archived_artifact=archived_artifact,
        )
        return None

    async def before_agent_callback(
        self,
        *,
        agent: BaseAgent,
        callback_context: CallbackContext,
    ) -> types.Content | None:
        callback_context.state[f"temp:agent_t0:{agent.name}"] = time.perf_counter()
        if staged := callback_context.state.get("temp:staged_artifact"):
            callback_context.state["last_uploaded_artifact"] = staged
        return None

    async def after_agent_callback(
        self,
        *,
        agent: BaseAgent,
        callback_context: CallbackContext,
    ) -> types.Content | None:
        t0 = callback_context.state.get(f"temp:agent_t0:{agent.name}")
        duration_ms = round((time.perf_counter() - t0) * 1000, 2) if t0 else 0.0
        with tracer.start_as_current_span(f"onboarding_trail.agent.{agent.name}") as sp:
            sp.set_attribute("gen_ai.agent.name", agent.name)
            sp.set_attribute("onboarding_trail.duration_ms", duration_ms)
        emit_structured_metric(
            "agent_completed",
            agent_name=agent.name,
            duration_ms=duration_ms,
        )
        return None

    async def before_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
    ) -> dict[str, Any] | None:
        tool_context.state[f"temp:tool_t0:{tool.name}"] = time.perf_counter()
        return None

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        result: dict[str, Any],
    ) -> dict[str, Any] | None:
        t0 = tool_context.state.get(f"temp:tool_t0:{tool.name}")
        duration_ms = round((time.perf_counter() - t0) * 1000, 2) if t0 else 0.0
        status = (
            result.get("status", "ok")
            if isinstance(result, dict)
            else "ok"
        )
        metrics = list(tool_context.state.get("temp:tool_metrics") or [])
        metrics.append(
            {
                "tool": tool.name,
                "duration_ms": duration_ms,
                "status": status,
            }
        )
        tool_context.state["temp:tool_metrics"] = metrics

        with tracer.start_as_current_span(f"onboarding_trail.tool.{tool.name}") as span:
            span.set_attribute("gen_ai.tool.name", tool.name)
            span.set_attribute("onboarding_trail.duration_ms", duration_ms)
            span.set_attribute("onboarding_trail.tool_status", str(status))

        emit_structured_metric(
            "tool_completed",
            tool_name=tool.name,
            duration_ms=duration_ms,
            status=status,
            arg_keys=sorted(tool_args.keys()),
        )
        return None

    async def on_tool_error_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        error: Exception,
    ) -> dict[str, Any] | None:
        t0 = tool_context.state.get(f"temp:tool_t0:{tool.name}")
        duration_ms = round((time.perf_counter() - t0) * 1000, 2) if t0 else 0.0
        with tracer.start_as_current_span(
            f"onboarding_trail.tool_error.{tool.name}"
        ) as span:
            span.set_attribute("gen_ai.tool.name", tool.name)
            span.set_attribute("onboarding_trail.duration_ms", duration_ms)
            span.set_attribute("error.type", type(error).__name__)
            span.record_exception(error)

        emit_structured_metric(
            "tool_error",
            tool_name=tool.name,
            duration_ms=duration_ms,
            error_type=type(error).__name__,
        )
        return {
            "status": "error",
            "tool": tool.name,
            "error_type": type(error).__name__,
            "message": f"Tool '{tool.name}' encountered an error: {error}",
        }

    async def after_model_callback(
        self,
        *,
        callback_context: CallbackContext,
        llm_response: LlmResponse,
    ) -> LlmResponse | None:
        usage = llm_response.usage_metadata
        if usage is not None:
            prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
            output_tokens = getattr(usage, "candidates_token_count", 0) or 0
            thought_tokens = getattr(usage, "thoughts_token_count", 0) or 0
            total_tokens = getattr(usage, "total_token_count", 0) or 0
            emit_structured_metric(
                "model_usage",
                agent_name=callback_context.agent_name,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                thought_tokens=thought_tokens,
                total_tokens=total_tokens,
            )
        return None

    async def after_run_callback(
        self,
        *,
        invocation_context: InvocationContext,
    ) -> None:
        """Persist completed session turns into MemoryService for cross-session recall."""
        if invocation_context.memory_service is None:
            return
        try:
            updated_session = await invocation_context.session_service.get_session(
                app_name=invocation_context.app_name,
                user_id=invocation_context.user_id,
                session_id=invocation_context.session.id,
            )
            session_to_index = updated_session or invocation_context.session
            await invocation_context.memory_service.add_session_to_memory(
                session_to_index
            )
            emit_structured_metric(
                "memory_synced",
                invocation_id=invocation_context.invocation_id,
                session_id=invocation_context.session.id,
            )
        except Exception as exc:  # pragma: no cover - non-fatal background sync
            telemetry_logger.warning("Memory sync skipped: %s", exc)
