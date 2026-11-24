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

"""Pydantic data models for the Personal Onboarding Trail Agent."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

DEFAULT_CATEGORIES: tuple[str, ...] = (
    "architecture",
    "adk-sdk",
    "deployment",
    "evaluation",
    "security",
    "tutorials",
)


class TrailEntry(BaseModel):
    """A single captured link or screenshot stored in the user's trail library."""

    entry_id: str = Field(description="Stable short identifier for the saved item.")
    url: str = Field(description="Source URL captured from link input or screenshot.")
    title: str = Field(description="Human-readable page or repository title.")
    summary: str = Field(description="Concise summary of the captured resource.")
    content_excerpt: str = Field(
        default="",
        description="Capped plain-text excerpt used for grounded keyword retrieval.",
    )
    key_takeaways: list[str] = Field(
        default_factory=list,
        description="Actionable bullet points extracted from the resource.",
    )
    category: str = Field(
        description="Approved category slug from the user's category list."
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Normalized free-form topic tags for fine-grained filtering.",
    )
    source_type: Literal["url", "screenshot"] = Field(
        description="How the item was captured: direct URL fetch or browser screenshot."
    )
    url_was_truncated: bool = Field(
        default=False,
        description="True if the address bar URL in a screenshot appeared cut off.",
    )
    artifact_name: str | None = Field(
        default=None,
        description="Session/user artifact filename when captured from a screenshot.",
    )
    saved_at: str = Field(description="UTC ISO-8601 timestamp when the item was saved.")


class OnboardingProfile(BaseModel):
    """Persistent user-scoped onboarding context remembered across sessions."""

    role: str = Field(default="", description="User's engineering role or title.")
    team_or_domain: str = Field(
        default="", description="Team, product area, or technical domain."
    )
    focus_topics: list[str] = Field(
        default_factory=list,
        description="Current onboarding topics the user is actively studying.",
    )
    learning_goals: list[str] = Field(
        default_factory=list,
        description="Open onboarding goals or questions still to master.",
    )
    completed_topics: list[str] = Field(
        default_factory=list,
        description="Topics the user has already mastered during onboarding.",
    )
    updated_at: str = Field(
        default="", description="UTC ISO-8601 timestamp of the last profile update."
    )


class KnowledgeQueryInput(BaseModel):
    """Input schema for the parallel trail-memory + web-search workflow tool."""

    question: str = Field(
        description="Self-contained technical question to answer for the user."
    )
    category_filter: str = Field(
        default="",
        description="Optional category slug to narrow saved trail lookup.",
    )


class JudgedAnswer(BaseModel):
    """Structured output produced by the judge agent comparing parallel sources."""

    winning_source: Literal["trail_memory", "google_search", "combined"] = Field(
        description=(
            "Which source provided the stronger answer: 'trail_memory' when saved "
            "onboarding notes answer it well, 'google_search' when not in saved "
            "notes, or 'combined' when both contribute."
        )
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confidence in the synthesized answer."
    )
    rationale: str = Field(
        description="One-sentence explanation of why the winning source was chosen."
    )
    answer_markdown: str = Field(
        description=(
            "Clear, concise answer tailored to the user's onboarding context, "
            "referencing concrete sources."
        )
    )
    citations: list[str] = Field(
        default_factory=list,
        description="Exact source URLs supporting the answer.",
    )
    judge_tier: str = Field(
        default="",
        description="Model routing tier selected by the complexity router ('fast_judge' or 'pro_judge').",
    )
