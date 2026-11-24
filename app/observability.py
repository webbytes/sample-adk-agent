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

"""OpenTelemetry tracing, pre-execution intent logging, and PII redaction pipeline."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from google.adk.plugins.base_plugin import BasePlugin
from google.genai import types
from opentelemetry import trace

if TYPE_CHECKING:
    from google.adk.agents.base_agent import BaseAgent
    from google.adk.agents.callback_context import CallbackContext
    from google.adk.agents.invocation_context import InvocationContext
    from google.adk.models.llm_request import LlmRequest
    from google.adk.models.llm_response import LlmResponse
    from google.adk.tools.base_tool import BaseTool
    from google.adk.tools.tool_context import ToolContext

tracer = trace.get_tracer("onboarding_trail.agent", "0.2.0")

_URL_ONLY_RE = re.compile(r"^\s*https?://\S+\s*$", re.IGNORECASE)

# Deterministic PII & secret detection patterns applied across logs, OTel spans,
# model inputs, and persisted state/memory pipelines.
_PII_PATTERNS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    (
        "secret_token",
        re.compile(
            r"\b(?:Bearer\s+[A-Za-z0-9._~+/=-]{12,}|AIza[0-9A-Za-z_-]{20,}|ya29\.[0-9A-Za-z._-]+|sk-[A-Za-z0-9]{16,})"
        ),
        "[REDACTED_SECRET]",
    ),
    (
        "email",
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        "[REDACTED_EMAIL]",
    ),
    (
        "ssn",
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "[REDACTED_SSN]",
    ),
    (
        "credit_card",
        re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
        "[REDACTED_CARD]",
    ),
    (
        "phone",
        re.compile(
            r"(?<!\d)(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]\d{3}[-.\s]\d{4}(?!\d)"
        ),
        "[REDACTED_PHONE]",
    ),
)

_SENSITIVE_QUERY_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "key",
        "token",
        "access_token",
        "refresh_token",
        "secret",
        "password",
        "passwd",
        "auth",
        "signature",
        "sig",
        "email",
        "ssn",
    }
)

_SENSITIVE_DICT_KEYS = _SENSITIVE_QUERY_KEYS | frozenset(
    {"authorization", "cookie", "set-cookie", "client_secret", "private_key"}
)

_TOOL_INTENT_DESCRIPTIONS: dict[str, str] = {
    "save_link_to_trail": "fetch_and_persist_public_url",
    "save_screenshot_to_trail": "extract_and_persist_browser_screenshot",
    "confirm_category_proposal": "apply_human_category_governance_decision",
    "search_trail_library": "retrieve_saved_onboarding_trail_entries",
    "update_onboarding_profile": "persist_user_onboarding_profile_context",
    "list_trail_status": "inspect_trail_inventory_and_categories",
    "parallel_knowledge_lookup": "execute_parallel_trail_and_search_workflow",
    "get_user_choice": "request_human_in_the_loop_choice",
    "load_memory": "query_long_term_semantic_memory",
    "load_artifacts": "load_archived_session_artifact",
}


def sanitize_url_pii(url: str) -> str:
    """Redact sensitive query parameters and userinfo credentials from a URL."""
    raw = url.strip()
    if not raw.startswith(("http://", "https://")):
        return redact_pii_text(raw)
    try:
        parts = urlsplit(raw)
        netloc = parts.netloc
        if "@" in netloc:
            _, hostpart = netloc.rsplit("@", 1)
            netloc = f"[REDACTED_AUTH]@{hostpart}"
        if parts.query:
            cleaned_params = [
                (
                    k,
                    "[REDACTED_PARAM]"
                    if k.lower() in _SENSITIVE_QUERY_KEYS
                    else redact_pii_text(v),
                )
                for k, v in parse_qsl(parts.query, keep_blank_values=True)
            ]
            query = urlencode(cleaned_params)
        else:
            query = ""
        return urlunsplit((parts.scheme, netloc, parts.path, query, parts.fragment))
    except ValueError:
        return redact_pii_text(raw)


def redact_pii_text(text: str) -> str:
    """Scrub PII and credentials from free-form text."""
    redacted, _ = redact_pii_with_types(text)
    return redacted


def redact_pii_with_types(text: str) -> tuple[str, list[str]]:
    """Scrub PII/secrets from text and return (scrubbed_text, detected_pii_types)."""
    if not text:
        return text, []
    scrubbed = text
    detected: list[str] = []

    # Sanitize inline URLs with sensitive query strings first.
    def _clean_url_match(match: re.Match[str]) -> str:
        cleaned = sanitize_url_pii(match.group(0))
        if "[REDACTED_" in cleaned and "url_secret" not in detected:
            detected.append("url_secret")
        return cleaned

    scrubbed = re.sub(r"https?://[^\s\"'<>\])},]+", _clean_url_match, scrubbed)

    for pii_type, pattern, placeholder in _PII_PATTERNS:
        if pattern.search(scrubbed):
            detected.append(pii_type)
            scrubbed = pattern.sub(placeholder, scrubbed)

    return scrubbed, detected


def redact_pii_data(value: Any) -> Any:
    """Recursively scrub PII and sensitive keys across dicts, lists, and strings."""
    if isinstance(value, str):
        return redact_pii_text(value)
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for k, v in value.items():
            key_str = str(k)
            if key_str.lower() in _SENSITIVE_DICT_KEYS:
                cleaned[key_str] = "[REDACTED_SECRET]"
            else:
                cleaned[key_str] = redact_pii_data(v)
        return cleaned
    if isinstance(value, (list, tuple)):
        return [redact_pii_data(item) for item in value]
    return value


class PiiRedactingLogFilter(logging.Filter):
    """Logging filter guaranteeing no raw PII or bearer secrets leave the logger."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact_pii_text(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = redact_pii_data(record.args)
            elif isinstance(record.args, tuple):
                record.args = tuple(redact_pii_data(a) for a in record.args)
        return True


telemetry_logger = logging.getLogger("onboarding_trail.telemetry")
if not any(isinstance(f, PiiRedactingLogFilter) for f in telemetry_logger.filters):
    telemetry_logger.addFilter(PiiRedactingLogFilter())


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


def classify_user_intent(user_message: types.Content | None) -> str:
    """Predict high-level user intent BEFORE agent/model execution for auditability."""
    modality = classify_message_modality(user_message)
    if modality == "screenshot":
        return "capture_screenshot_to_trail"
    if modality == "url":
        return "capture_url_to_trail"
    if modality == "empty" or not user_message or not user_message.parts:
        return "empty_or_control_signal"

    text = " ".join(p.text.strip() for p in user_message.parts if p.text).lower()
    if any(
        phrase in text
        for phrase in ("status", "what have i saved", "list categories", "my trail")
    ):
        return "inspect_trail_status"
    if any(
        phrase in text
        for phrase in ("my role is", "i am a ", "i'm a ", "learning goal", "onboard")
    ) and "?" not in text:
        return "update_onboarding_profile"
    if any(phrase in text for phrase in ("approve", "reject")) and "category" in text:
        return "resolve_category_governance"
    if "http://" in text or "https://" in text:
        return "capture_url_with_context"
    return "knowledge_query_or_dialogue"


def emit_structured_metric(event_type: str, **fields: Any) -> None:
    """Emit a PII-redacted single-line JSON structured telemetry record."""
    safe_fields = redact_pii_data(fields)
    payload = {"event_type": event_type, **safe_fields}
    telemetry_logger.info(json.dumps(payload, default=str, sort_keys=True))


class TrailObservabilityPlugin(BasePlugin):
    """App-wide pre/post intent tracing, PII redaction, artifact archival, & Memory sync.

    Stateless singleton: never mutates ``self.*`` with per-session state; all
    per-invocation state uses ``temp:`` keys on the ADK context.
    """

    def __init__(self, name: str = "trail_observability_plugin") -> None:
        super().__init__(name=name)

    async def on_user_message_callback(
        self,
        *,
        invocation_context: InvocationContext,
        user_message: types.Content,
    ) -> types.Content | None:
        """Scrub inbound PII, record pre-execution user intent, and archive screenshots."""
        modality = classify_message_modality(user_message)
        predicted_intent = classify_user_intent(user_message)
        archived_artifact: str | None = None
        detected_pii_types: list[str] = []
        mutated_parts: list[types.Part] = []
        message_was_redacted = False

        for idx, part in enumerate(user_message.parts or []):
            if part.text:
                scrubbed_text, found_types = redact_pii_with_types(part.text)
                if found_types:
                    detected_pii_types.extend(found_types)
                    message_was_redacted = True
                    mutated_parts.append(types.Part.from_text(text=scrubbed_text))
                else:
                    mutated_parts.append(part)
                continue

            blob = part.inline_data
            if (
                modality == "screenshot"
                and blob is not None
                and (blob.mime_type or "").startswith("image/")
                and invocation_context.artifact_service
            ):
                raw_name = redact_pii_text(blob.display_name or f"screenshot_{idx}.png")
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
            mutated_parts.append(part)

        deduped_pii = sorted(set(detected_pii_types))

        # Pre-execution intent trace span BEFORE any agent, model, or tool runs.
        with tracer.start_as_current_span("onboarding_trail.intent.user_turn") as span:
            span.set_attribute("onboarding_trail.phase", "pre_execution")
            span.set_attribute("onboarding_trail.predicted_intent", predicted_intent)
            span.set_attribute("onboarding_trail.input_modality", modality)
            span.set_attribute("onboarding_trail.pii_redacted", bool(deduped_pii))
            if deduped_pii:
                span.set_attribute(
                    "onboarding_trail.pii_types_redacted", ",".join(deduped_pii)
                )
            span.set_attribute(
                "onboarding_trail.invocation_id", invocation_context.invocation_id
            )
            if archived_artifact:
                span.set_attribute(
                    "onboarding_trail.archived_artifact", archived_artifact
                )

        invocation_context.session.state["temp:input_modality"] = modality
        invocation_context.session.state["temp:predicted_intent"] = predicted_intent
        invocation_context.session.state["temp:pii_redacted_types"] = deduped_pii
        if archived_artifact:
            invocation_context.session.state["temp:staged_artifact"] = archived_artifact

        emit_structured_metric(
            "user_intent_recorded",
            phase="pre_execution",
            invocation_id=invocation_context.invocation_id,
            predicted_intent=predicted_intent,
            modality=modality,
            pii_redacted=bool(deduped_pii),
            pii_types=deduped_pii,
            archived_artifact=archived_artifact,
        )

        if message_was_redacted:
            return types.Content(role=user_message.role, parts=mutated_parts)
        return None

    async def before_agent_callback(
        self,
        *,
        agent: BaseAgent,
        callback_context: CallbackContext,
    ) -> types.Content | None:
        """Record pre-execution agent intent before the agent begins processing."""
        callback_context.state[f"temp:agent_t0:{agent.name}"] = time.perf_counter()
        if staged := callback_context.state.get("temp:staged_artifact"):
            callback_context.state["last_uploaded_artifact"] = staged

        predicted_intent = str(
            callback_context.state.get("temp:predicted_intent") or "sub_agent_delegation"
        )
        with tracer.start_as_current_span(
            f"onboarding_trail.intent.agent.{agent.name}"
        ) as span:
            span.set_attribute("onboarding_trail.phase", "pre_execution")
            span.set_attribute("gen_ai.agent.name", agent.name)
            span.set_attribute("onboarding_trail.predicted_intent", predicted_intent)

        emit_structured_metric(
            "agent_intent_recorded",
            phase="pre_execution",
            agent_name=agent.name,
            predicted_intent=predicted_intent,
        )
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
            sp.set_attribute("onboarding_trail.phase", "post_execution")
            sp.set_attribute("gen_ai.agent.name", agent.name)
            sp.set_attribute("onboarding_trail.duration_ms", duration_ms)
        emit_structured_metric(
            "agent_completed",
            phase="post_execution",
            agent_name=agent.name,
            duration_ms=duration_ms,
        )
        return None

    async def before_model_callback(
        self,
        *,
        callback_context: CallbackContext,
        llm_request: LlmRequest,
    ) -> LlmResponse | None:
        """Scrub outbound prompt PII and log model-call intent prior to inference."""
        model_name = str(getattr(llm_request, "model", "") or "configured_model")
        tool_count = len(getattr(llm_request, "tools_dict", {}) or {})
        pii_scrubbed_in_prompt = False

        for content in getattr(llm_request, "contents", []) or []:
            for part in getattr(content, "parts", None) or []:
                if part.text:
                    cleaned, found = redact_pii_with_types(part.text)
                    if found:
                        part.text = cleaned
                        pii_scrubbed_in_prompt = True

        with tracer.start_as_current_span(
            f"onboarding_trail.intent.model.{callback_context.agent_name}"
        ) as span:
            span.set_attribute("onboarding_trail.phase", "pre_execution")
            span.set_attribute("gen_ai.agent.name", callback_context.agent_name)
            span.set_attribute("gen_ai.request.model", model_name)
            span.set_attribute("onboarding_trail.available_tool_count", tool_count)
            span.set_attribute(
                "onboarding_trail.prompt_pii_scrubbed", pii_scrubbed_in_prompt
            )

        emit_structured_metric(
            "model_request_intent",
            phase="pre_execution",
            agent_name=callback_context.agent_name,
            model=model_name,
            available_tool_count=tool_count,
            prompt_pii_scrubbed=pii_scrubbed_in_prompt,
        )
        return None

    async def before_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
    ) -> dict[str, Any] | None:
        """Record tool intent and redact PII in tool_args BEFORE tool execution."""
        tool_context.state[f"temp:tool_t0:{tool.name}"] = time.perf_counter()
        planned_intent = _TOOL_INTENT_DESCRIPTIONS.get(
            tool.name, f"invoke_{tool.name}"
        )
        safe_args_preview = redact_pii_data(tool_args)

        intent_log = list(tool_context.state.get("temp:tool_intents") or [])
        intent_log.append(
            {
                "tool": tool.name,
                "planned_intent": planned_intent,
                "args": safe_args_preview,
            }
        )
        tool_context.state["temp:tool_intents"] = intent_log

        with tracer.start_as_current_span(
            f"onboarding_trail.intent.tool.{tool.name}"
        ) as span:
            span.set_attribute("onboarding_trail.phase", "pre_execution")
            span.set_attribute("gen_ai.tool.name", tool.name)
            span.set_attribute("onboarding_trail.planned_intent", planned_intent)
            span.set_attribute(
                "onboarding_trail.arg_keys", ",".join(sorted(tool_args.keys()))
            )

        emit_structured_metric(
            "tool_intent_recorded",
            phase="pre_execution",
            tool_name=tool.name,
            planned_intent=planned_intent,
            redacted_args=safe_args_preview,
        )
        return None

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        result: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Scrub PII from tool outputs and emit post-execution outcome metrics."""
        t0 = tool_context.state.get(f"temp:tool_t0:{tool.name}")
        duration_ms = round((time.perf_counter() - t0) * 1000, 2) if t0 else 0.0
        status = result.get("status", "ok") if isinstance(result, dict) else "ok"

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
            span.set_attribute("onboarding_trail.phase", "post_execution")
            span.set_attribute("gen_ai.tool.name", tool.name)
            span.set_attribute("onboarding_trail.duration_ms", duration_ms)
            span.set_attribute("onboarding_trail.tool_status", str(status))

        emit_structured_metric(
            "tool_completed",
            phase="post_execution",
            tool_name=tool.name,
            duration_ms=duration_ms,
            status=status,
            arg_keys=sorted(tool_args.keys()),
        )

        if isinstance(result, dict):
            scrubbed_result = redact_pii_data(result)
            if scrubbed_result != result:
                return scrubbed_result
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
        safe_err_msg = redact_pii_text(str(error))
        with tracer.start_as_current_span(
            f"onboarding_trail.tool_error.{tool.name}"
        ) as span:
            span.set_attribute("onboarding_trail.phase", "post_execution")
            span.set_attribute("gen_ai.tool.name", tool.name)
            span.set_attribute("onboarding_trail.duration_ms", duration_ms)
            span.set_attribute("error.type", type(error).__name__)

        emit_structured_metric(
            "tool_error",
            phase="post_execution",
            tool_name=tool.name,
            duration_ms=duration_ms,
            error_type=type(error).__name__,
            error_message=safe_err_msg,
        )
        return {
            "status": "error",
            "tool": tool.name,
            "error_type": type(error).__name__,
            "message": f"Tool '{tool.name}' encountered an error: {safe_err_msg}",
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
                phase="post_execution",
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
        """Persist PII-scrubbed session turns into MemoryService for semantic recall."""
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
                phase="post_execution",
                invocation_id=invocation_context.invocation_id,
                session_id=invocation_context.session.id,
            )
        except Exception as exc:  # pragma: no cover - non-fatal background sync
            telemetry_logger.warning("Memory sync skipped: %s", redact_pii_text(str(exc)))
