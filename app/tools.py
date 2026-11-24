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

"""Typed ADK function tools for the Personal Onboarding Trail Agent."""

from __future__ import annotations

import datetime
import hashlib
import re
from typing import Any
from urllib.parse import urlparse

from google.adk.tools import ToolContext
from google.adk.tools.load_web_page import load_web_page

from app.observability import tracer
from app.schemas import DEFAULT_CATEGORIES, OnboardingProfile, TrailEntry

MAX_EXCERPT_CHARS = 3500
TRAIL_STATE_KEY = "user:trail_library"
CATEGORIES_STATE_KEY = "user:categories"
PENDING_CATEGORIES_STATE_KEY = "user:pending_categories"
PROFILE_STATE_KEY = "user:onboarding_profile"


def normalize_slug(value: str) -> str:
    """Normalize a category or tag name into a lowercase hyphenated slug."""
    cleaned = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
    return cleaned or "uncategorized"


def _utc_now_iso() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


def _entry_id_for_url(url: str) -> str:
    return hashlib.sha256(url.strip().encode("utf-8")).hexdigest()[:12]


def get_allowed_categories(state: Any) -> list[str]:
    """Return the user's active category list, initializing defaults if unset."""
    existing = state.get(CATEGORIES_STATE_KEY)
    if isinstance(existing, list) and existing:
        return [normalize_slug(str(cat)) for cat in existing]
    defaults = list(DEFAULT_CATEGORIES)
    state[CATEGORIES_STATE_KEY] = defaults
    return defaults


def _validate_public_http_url(url: str) -> str | None:
    """Return an error string if the URL is not a valid http(s) URL."""
    parsed = urlparse(url.strip())
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return (
            f"Invalid URL '{url}'. Provide a complete http:// or https:// URL "
            "(or upload a browser screenshot showing the address bar)."
        )
    return None


def _derive_title_and_summary(
    url: str, page_text: str, provided_summary: str
) -> tuple[str, str]:
    lines = [ln.strip() for ln in page_text.splitlines() if ln.strip()]
    title = lines[0][:140] if lines else urlparse(url).netloc
    if provided_summary.strip():
        summary = provided_summary.strip()[:600]
    else:
        preview_lines = lines[:4]
        summary = " ".join(preview_lines)[:500] or f"Saved page from {url}"
    return title, summary


def _check_or_propose_category(
    category: str, state: Any
) -> tuple[str, dict[str, Any] | None]:
    """Validate category against user:categories or stage a proposal."""
    slug = normalize_slug(category)
    allowed = get_allowed_categories(state)
    if slug in allowed:
        return slug, None

    pending = list(state.get(PENDING_CATEGORIES_STATE_KEY) or [])
    if slug not in pending:
        pending.append(slug)
        state[PENDING_CATEGORIES_STATE_KEY] = pending

    return slug, {
        "status": "category_approval_required",
        "proposed_category": slug,
        "allowed_categories": allowed,
        "message": (
            f"Category '{slug}' is not in the approved list {allowed}. "
            "Ask the user whether to approve adding this new category "
            "(for example using get_user_choice) or choose an existing category, "
            "then call confirm_category_proposal before saving."
        ),
    }


def _upsert_trail_entry(entry: TrailEntry, state: Any) -> dict[str, Any]:
    """Insert or update a TrailEntry in user:trail_library by entry_id."""
    existing = list(state.get(TRAIL_STATE_KEY) or [])
    record = entry.model_dump()
    replaced = False
    for idx, item in enumerate(existing):
        if isinstance(item, dict) and item.get("entry_id") == entry.entry_id:
            existing[idx] = record
            replaced = True
            break
    if not replaced:
        existing.append(record)
    # Assign the list back so ADK records a state_delta across sessions.
    state[TRAIL_STATE_KEY] = existing
    return {
        "status": "updated" if replaced else "saved",
        "total_saved": len(existing),
        "entry": record,
    }


def rank_trail_entries(
    entries: list[dict[str, Any]],
    query: str,
    category_filter: str = "",
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Score and rank saved trail entries by query token and category match."""
    norm_cat = normalize_slug(category_filter) if category_filter.strip() else ""
    tokens = [
        tok
        for tok in re.findall(r"[a-z0-9]{2,}", query.lower())
        if tok not in {"what", "when", "where", "which", "with", "from", "about", "does", "have", "that", "this"}
    ]

    scored: list[tuple[int, str, dict[str, Any]]] = []
    for item in entries:
        if not isinstance(item, dict):
            continue
        if norm_cat and normalize_slug(str(item.get("category", ""))) != norm_cat:
            continue

        if not tokens:
            scored.append((1, str(item.get("saved_at", "")), item))
            continue

        title = str(item.get("title", "")).lower()
        summary = str(item.get("summary", "")).lower()
        excerpt = str(item.get("content_excerpt", "")).lower()
        url = str(item.get("url", "")).lower()
        category = str(item.get("category", "")).lower()
        tags = " ".join(str(t).lower() for t in (item.get("tags") or []))
        takeaways = " ".join(str(t).lower() for t in (item.get("key_takeaways") or []))

        score = 0
        for tok in tokens:
            if tok in title:
                score += 4
            if tok in tags or tok in category:
                score += 3
            if tok in summary or tok in takeaways:
                score += 2
            if tok in excerpt or tok in url:
                score += 1

        if score > 0:
            scored.append((score, str(item.get("saved_at", "")), item))

    scored.sort(key=lambda row: (row[0], row[1]), reverse=True)
    return [row[2] for row in scored[: max(1, limit)]]


def save_link_to_trail(
    url: str,
    category: str,
    summary: str = "",
    key_takeaways: list[str] | None = None,
    tags: list[str] | None = None,
    *,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Fetch a public web page and save it to the user's onboarding trail library.

    Call this tool when the user pastes a link so they do not have to type notes.
    If fetching fails (for example due to login walls, redirects, or client-side
    rendering), ask the user to upload a full-window browser screenshot instead.

    Args:
        url: Complete http:// or https:// URL to fetch and save.
        category: Category slug from the approved list (or a proposed new category).
        summary: Optional concise summary if known; otherwise derived from page text.
        key_takeaways: Optional list of key bullet points from the page.
        tags: Optional list of short topic tags (for example ['adk', 'state']).

    Returns:
        A status dictionary with the saved entry, a category-approval prompt, or
        a fetch-failure fallback instruction.
    """
    url_clean = url.strip()
    if error := _validate_public_http_url(url_clean):
        return {"status": "invalid_url", "url": url_clean, "message": error}

    cat_slug, approval_needed = _check_or_propose_category(
        category, tool_context.state
    )
    if approval_needed is not None:
        return approval_needed

    with tracer.start_as_current_span("onboarding_trail.fetch_web_page") as span:
        span.set_attribute("url.full", url_clean)
        fetched_text = load_web_page(url_clean)
        fetch_failed = fetched_text.startswith("Failed to fetch url:") or not fetched_text.strip()
        span.set_attribute("onboarding_trail.fetch_failed", fetch_failed)

    if fetch_failed:
        return {
            "status": "fetch_failed",
            "url": url_clean,
            "message": (
                f"Could not fetch '{url_clean}' directly (page may redirect, "
                "require JavaScript, or need authentication)."
            ),
            "fallback_instruction": (
                "Ask the user to upload a full-window browser screenshot "
                "(click '+' -> 'Upload local file') showing both the address bar "
                "URL and the visible page content so you can capture it via "
                "save_screenshot_to_trail."
            ),
        }

    title, final_summary = _derive_title_and_summary(url_clean, fetched_text, summary)
    normalized_tags = sorted(
        {normalize_slug(t) for t in (tags or []) if str(t).strip()}
    )
    entry = TrailEntry(
        entry_id=_entry_id_for_url(url_clean),
        url=url_clean,
        title=title,
        summary=final_summary,
        content_excerpt=fetched_text[:MAX_EXCERPT_CHARS],
        key_takeaways=[t.strip() for t in (key_takeaways or []) if t.strip()][:8],
        category=cat_slug,
        tags=normalized_tags,
        source_type="url",
        saved_at=_utc_now_iso(),
    )
    return _upsert_trail_entry(entry, tool_context.state)


def save_screenshot_to_trail(
    extracted_url: str,
    page_title: str,
    content_summary: str,
    category: str,
    key_takeaways: list[str] | None = None,
    tags: list[str] | None = None,
    url_was_truncated: bool = False,
    *,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Save content and source URL extracted from an uploaded browser screenshot.

    Call this tool immediately after inspecting an image uploaded by the user.
    Read the source URL from the browser address bar in the screenshot, summarize
    the visible page content, and classify it into an approved category.

    Args:
        extracted_url: URL read from the screenshot's browser address bar (or
            best visible host/path if no scheme is shown).
        page_title: Visible page, document, or repository title from the image.
        content_summary: Clear summary of the technical content shown on screen.
        category: Category slug from the approved list (or a proposed new category).
        key_takeaways: Bullet points capturing commands, APIs, or facts on screen.
        tags: Short topic tags describing the screenshot.
        url_was_truncated: True if the address bar cut off the tail of a long URL.

    Returns:
        A status dictionary with the saved screenshot entry or category approval request.
    """
    raw_url = extracted_url.strip() or "https://unknown-source.local/screenshot"
    if not raw_url.startswith(("http://", "https://")):
        raw_url = f"https://{raw_url}"

    cat_slug, approval_needed = _check_or_propose_category(
        category, tool_context.state
    )
    if approval_needed is not None:
        return approval_needed

    artifact_name = (
        tool_context.state.get("last_uploaded_artifact")
        or tool_context.state.get("temp:staged_artifact")
    )
    takeaways = [t.strip() for t in (key_takeaways or []) if t.strip()][:10]
    normalized_tags = sorted(
        {normalize_slug(t) for t in (tags or []) if str(t).strip()}
    )
    excerpt_parts = [page_title.strip(), content_summary.strip(), *takeaways]
    entry = TrailEntry(
        entry_id=_entry_id_for_url(f"{raw_url}#{page_title.strip()}"),
        url=raw_url,
        title=page_title.strip()[:140] or "Captured Screenshot",
        summary=content_summary.strip()[:800],
        content_excerpt="\n".join(p for p in excerpt_parts if p)[:MAX_EXCERPT_CHARS],
        key_takeaways=takeaways,
        category=cat_slug,
        tags=normalized_tags,
        source_type="screenshot",
        url_was_truncated=url_was_truncated,
        artifact_name=str(artifact_name) if artifact_name else None,
        saved_at=_utc_now_iso(),
    )
    return _upsert_trail_entry(entry, tool_context.state)


def confirm_category_proposal(
    category_name: str,
    approved: bool,
    *,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Record the user's approval or rejection of a proposed trail category.

    Args:
        category_name: The proposed category name to approve or reject.
        approved: True if the user approved adding the category, False otherwise.

    Returns:
        Updated list of approved categories and pending proposals.
    """
    slug = normalize_slug(category_name)
    allowed = get_allowed_categories(tool_context.state)
    pending = [
        normalize_slug(str(c))
        for c in (tool_context.state.get(PENDING_CATEGORIES_STATE_KEY) or [])
    ]
    pending = [c for c in pending if c != slug]
    tool_context.state[PENDING_CATEGORIES_STATE_KEY] = pending

    if approved and slug not in allowed:
        allowed.append(slug)
        tool_context.state[CATEGORIES_STATE_KEY] = allowed
        return {
            "status": "approved",
            "category": slug,
            "allowed_categories": allowed,
        }
    return {
        "status": "already_exists" if approved else "rejected",
        "category": slug,
        "allowed_categories": allowed,
    }


def search_trail_library(
    query: str,
    category: str = "",
    limit: int = 5,
    *,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Search saved links and screenshots in the user's onboarding trail library.

    Args:
        query: Search terms or question keywords to match against saved items.
        category: Optional category slug to restrict results.
        limit: Maximum number of matching entries to return (1 to 10).

    Returns:
        Dictionary with matched_count and ranked entries including source URLs.
    """
    entries = list(tool_context.state.get(TRAIL_STATE_KEY) or [])
    capped_limit = max(1, min(10, limit))
    matches = rank_trail_entries(
        entries=entries,
        query=query,
        category_filter=category,
        limit=capped_limit,
    )
    return {
        "status": "ok",
        "total_library_size": len(entries),
        "matched_count": len(matches),
        "matches": matches,
    }


def update_onboarding_profile(
    role: str = "",
    team_or_domain: str = "",
    focus_topics: list[str] | None = None,
    learning_goals: list[str] | None = None,
    completed_topics: list[str] | None = None,
    *,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Update persistent user-scoped onboarding profile facts across sessions.

    Call this when the user shares their role, team, current focus stack, open
    onboarding goals, or topics they have finished learning.

    Args:
        role: User's engineering role (for example 'Forward Deployed Engineer').
        team_or_domain: Team or technical domain (for example 'Cloud AI Agents').
        focus_topics: Topics to add to the user's active study list.
        learning_goals: Open onboarding questions or milestones to track.
        completed_topics: Topics the user has finished and mastered.

    Returns:
        Updated onboarding profile stored under user:onboarding_profile.
    """
    existing_raw = tool_context.state.get(PROFILE_STATE_KEY) or {}
    profile = (
        OnboardingProfile(**existing_raw)
        if isinstance(existing_raw, dict)
        else OnboardingProfile()
    )
    if role.strip():
        profile.role = role.strip()
    if team_or_domain.strip():
        profile.team_or_domain = team_or_domain.strip()
    if focus_topics:
        merged_focus = list(dict.fromkeys([*profile.focus_topics, *focus_topics]))
        profile.focus_topics = merged_focus
    if learning_goals:
        merged_goals = list(dict.fromkeys([*profile.learning_goals, *learning_goals]))
        profile.learning_goals = merged_goals
    if completed_topics:
        merged_done = list(
            dict.fromkeys([*profile.completed_topics, *completed_topics])
        )
        profile.completed_topics = merged_done
        profile.learning_goals = [
            g for g in profile.learning_goals if g not in merged_done
        ]
    profile.updated_at = _utc_now_iso()
    dumped = profile.model_dump()
    tool_context.state[PROFILE_STATE_KEY] = dumped
    return {"status": "updated", "profile": dumped}


def list_trail_status(*, tool_context: ToolContext) -> dict[str, Any]:
    """Return a summary of saved links/screenshots, categories, and user profile.

    Returns:
        Dictionary summarizing categories, counts by category/source, recent
        items, and the user's onboarding profile.
    """
    entries = list(tool_context.state.get(TRAIL_STATE_KEY) or [])
    categories = get_allowed_categories(tool_context.state)
    pending = list(tool_context.state.get(PENDING_CATEGORIES_STATE_KEY) or [])
    profile = tool_context.state.get(PROFILE_STATE_KEY) or {}

    by_category: dict[str, int] = {}
    by_source: dict[str, int] = {"url": 0, "screenshot": 0}
    for item in entries:
        if not isinstance(item, dict):
            continue
        cat = str(item.get("category", "uncategorized"))
        by_category[cat] = by_category.get(cat, 0) + 1
        src = str(item.get("source_type", "url"))
        by_source[src] = by_source.get(src, 0) + 1

    recent = [
        {
            "title": item.get("title"),
            "url": item.get("url"),
            "category": item.get("category"),
            "source_type": item.get("source_type"),
            "saved_at": item.get("saved_at"),
        }
        for item in entries[-10:]
        if isinstance(item, dict)
    ]
    recent.reverse()

    return {
        "status": "ok",
        "total_saved": len(entries),
        "allowed_categories": categories,
        "pending_categories": pending,
        "counts_by_category": by_category,
        "counts_by_source_type": by_source,
        "recent_items": recent,
        "onboarding_profile": profile,
    }
