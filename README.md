# Personal Onboarding Trail Agent

**Track:** Concierge / Enterprise Productivity Agents  
**Framework:** Google Agent Development Kit (ADK 2.0, Python) + Vertex AI Agent Runtime + A2A Protocol

## Problem & Solution

* **Problem:** Engineers onboarding onto a new technical stack read dozens of public documentation pages, GitHub repositories, and tutorials across weeks. Manually typing notes breaks flow, browser history buries important pages under noise, and standard chat assistants have no memory of which pages you already studied or what your open onboarding goals are.
* **Solution:** The **Personal Onboarding Trail Agent** is a zero-typing daily onboarding assistant. Paste a URL or upload a full-window browser screenshot with no extra text: the agent fetches or visually extracts the source URL, title, summary, and takeaways, categorizes it against your governed category list (pausing for human approval before adding a new category), and builds a persistent cross-session memory of your reading trail and onboarding profile. When you ask a question, an ADK 2.0 `Workflow` graph queries your **Personal Trail & Memory Bank** and **Live Google Search** in parallel, then runs a structured **Judge Agent** to return the stronger cited answer.

---

## Architecture

```mermaid
flowchart TD
    User["User Input (URL / Browser Screenshot / Question)"] --> Plugin["TrailObservabilityPlugin (BasePlugin)\n• Classifies modality\n• Archives screenshot PNG to ArtifactService\n• Emits OTel spans & structured JSON metrics"]
    Plugin --> Root["root_agent (ADK 2.0 Agent)\nState-templated user:onboarding_profile & user:categories"]

    Root -->|"Pasted URL"| SaveLink["save_link_to_trail\n(load_web_page + SSRF guard)"]
    SaveLink -->|"JS / Redirect / Auth failure"| Fallback["Deterministic Fallback:\nPrompt for full-window screenshot"]
    Root -->|"Uploaded Screenshot"| SaveShot["save_screenshot_to_trail\n(Multimodal address-bar URL + summary)"]

    SaveLink --> CatCheck{"Approved in\nuser:categories?"}
    SaveShot --> CatCheck
    CatCheck -->|"No"| HITL["get_user_choice + confirm_category_proposal\n(Human-in-the-Loop Category Approval)"]
    CatCheck -->|"Yes"| StateStore[("3-Tier Context & Memory\n1. user:trail_library & user:onboarding_profile\n2. VertexAiMemoryBankService / InMemoryMemoryService\n3. GcsArtifactService / InMemoryArtifactService")]

    Root -->|"Technical Question"| WF["parallel_knowledge_lookup (ADK 2.0 Workflow)"]
    WF --> BranchA["retrieve_from_trail_and_memory\n(Ranked Trail + search_memory)"]
    WF --> BranchB["web_grounding_agent (gemini-3.5-flash)\nIsolated Google Search Grounding"]
    BranchA --> Join["JoinNode (join_knowledge_sources)"]
    BranchB --> Join
    Join --> Router{"route_judge_by_complexity\n(Strategic Model Router)"}
    Router -->|"Single-source factual"| FastJudge["fast_answer_judge_agent (gemini-3.8-flash)\nLow-latency Flash Judge (temp=0.0)"]
    Router -->|"Multi-source / Architectural"| ProJudge["pro_answer_judge_agent (gemini-3.1-pro-preview)\nDeep-reasoning Pro Judge (temp=0.0)"]
    FastJudge --> Root
    ProJudge --> Root
```

### Strategic Model Routing (Cost, Latency & Reasoning Tiers)

Instead of hardcoding a single LLM across every sub-task, the agent routes across three specialized Gemini tiers (each overridable via environment variables `RETRIEVAL_MODEL`, `ORCHESTRATOR_MODEL`, `REASONING_MODEL`):

1. **Retrieval Tier (`gemini-3.5-flash` -> `web_grounding_agent`):** High-throughput, cost-efficient Flash model dedicated to parallel Google Search grounding and URL extraction.
2. **Multimodal & Fast Orchestration Tier (`gemini-3.8-flash` -> `root_agent`, `fast_answer_judge_agent`):** Handles browser screenshot visual extraction, deterministic tool calling, and sub-second `fast_judge` evaluation for single-source factual lookups.
3. **Deep Reasoning Tier (`gemini-3.1-pro-preview` -> `pro_answer_judge_agent`):** Dynamically selected via `Event(route="pro_judge")` in `route_judge_by_complexity` whenever a query requires reconciling both Personal Trail Memory and Live Web Search or involves architectural/comparative trade-off reasoning.

---

## Mapping to Assessment Evaluation Criteria

| Criterion | Implementation in This Repository | Key Files |
| :--- | :--- | :--- |
| **1. Tool & Interface Design** | Six single-responsibility typed `FunctionTool`s with Pydantic `BaseModel` validation (`TrailEntry`, `OnboardingProfile`, `KnowledgeQueryInput`, `JudgedAnswer`), `ToolContext` state isolation, deterministic `fetch_failed` -> screenshot fallback, `get_user_choice` (`LongRunningFunctionTool`) for zero-typing approval, and an A2A (`Agent2Agent`) JSON-RPC + `/.well-known/agent-card.json` server interface. | `app/tools.py`, `app/schemas.py`, `app/app_utils/a2a.py` |
| **2. Context & Memory** | Three complementary persistence layers: (1) `user:`-prefixed session state (`user:trail_library`, `user:categories`, `user:onboarding_profile`) injected via `{user:...}` instruction templates plus `EventsCompactionConfig`; (2) cross-session semantic recall via `VertexAiMemoryBankService` (Agent Runtime) / `InMemoryMemoryService` (local) automatically synced on `after_run_callback` and queried via `load_memory` & `ctx.search_memory`; (3) screenshot binary archival via `GcsArtifactService` / `InMemoryArtifactService` + `load_artifacts`. | `app/tools.py`, `app/app_utils/services.py`, `app/observability.py`, `app/agent.py` |
| **3. Orchestration & Logic** | Combines multimodal routing on `root_agent` (`gemini-3.8-flash`) with an ADK 2.0 `Workflow` DAG (`parallel_knowledge_lookup`): concurrent tuple fan-out to `retrieve_from_trail_and_memory` and an isolated `mode="single_turn"` `web_grounding_agent` (`gemini-3.5-flash`), synchronization at `JoinNode`, **strategic model routing** (`route_judge_by_complexity` emitting `Event(route="fast_judge" \| "pro_judge")`) between `fast_answer_judge_agent` (`gemini-3.8-flash`) and `pro_answer_judge_agent` (`gemini-3.1-pro-preview`, `output_schema=JudgedAnswer`, `temperature=0.0`), plus `ResumabilityConfig(is_resumable=True)` and `ReflectAndRetryToolPlugin`. | `app/workflows.py`, `app/agent.py` |
| **4. Observability & Tracing** | Custom stateless `TrailObservabilityPlugin(BasePlugin)` implementing **both pre-execution intent recording and post-execution outcome telemetry**: logs and traces predicted user intent (`user_intent_recorded`), agent intent (`agent_intent_recorded`), outbound model intent (`model_request_intent`), tool intent (`tool_intent_recorded`), and parallel workflow branch intent (`parallel_branch_intent`) **before** execution, paired with post-execution latency (`duration_ms`), token counts, and judge winner metrics. Enforces an **end-to-end PII & secret redaction pipeline** (`redact_pii_text`, `redact_pii_data`, `sanitize_url_pii`, `PiiRedactingLogFilter`) across inbound user messages, outbound LLM prompts, persisted state/memory, tool outputs, OpenTelemetry spans, and structured JSON logs. | `app/observability.py`, `app/tools.py`, `app/workflows.py`, `app/fast_api_app.py` |
| **5. Infrastructure & CI/CD** | Production multi-stage `Dockerfile`, Terraform IaC (`deployment/terraform/`) targeting Vertex AI Agent Runtime with least-privilege IAM and Cloud Storage telemetry buckets, GitHub Actions pipelines (`pr_checks.yaml` for Ruff linting + offline `pytest` unit/workflow tests + gated WIF cloud integration tests, `staging.yaml`, and `deploy-to-prod.yaml`), load tests (`tests/load_test/`), and LLM-as-judge evaluation suite (`tests/eval/`). | `Dockerfile`, `deployment/terraform/`, `.github/workflows/`, `tests/` |

---

## Project Structure

```text
.
├── app/
│   ├── agent.py                 # Root ADK 2.0 Agent, App config, plugins, compaction & resumability
│   ├── schemas.py               # Pydantic schemas (TrailEntry, OnboardingProfile, JudgedAnswer)
│   ├── tools.py                 # Typed link/screenshot capture, category governance & search tools
│   ├── workflows.py             # ADK 2.0 parallel fan-out + JoinNode + Judge Workflow graph
│   ├── observability.py         # OpenTelemetry spans & TrailObservabilityPlugin (BasePlugin)
│   ├── fast_api_app.py          # FastAPI entrypoint (ADK SSE, Dev UI, A2A, Reasoning Engine adapter)
│   └── app_utils/               # Shared Session/Artifact/Memory services & A2A/Runtime adapters
├── deployment/terraform/        # Terraform IaC for staging/prod & CI/CD Workload Identity
├── .github/workflows/           # GitHub Actions (pr_checks.yaml, staging.yaml, deploy-to-prod.yaml)
├── tests/
│   ├── unit/                    # Offline deterministic unit & workflow tests (ScriptedLlm + InMemoryRunner)
│   ├── integration/             # Live streaming, A2A agent-card, and Reasoning Engine E2E tests
│   ├── eval/                    # Offline evaluation dataset & LLM-as-judge rubric
│   └── load_test/               # Locust load tests
├── Dockerfile                   # Container build for Cloud Run / Agent Runtime
└── pyproject.toml               # Dependencies, Ruff lint rules, and pytest configuration
```

---

## Quickstart

### 1. Prerequisites & Installation

Requires Python 3.11–3.13 and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync --group dev --extra lint
```

Copy `.env.example` to `.env` and set either your Google Cloud project (Vertex AI / Gemini Enterprise Agent Platform) or a Gemini API key:

```bash
cp .env.example .env
# Edit .env:
# GOOGLE_GENAI_USE_ENTERPRISE=true
# GOOGLE_CLOUD_PROJECT=your-gcp-project-id
# GOOGLE_CLOUD_LOCATION=global
```

### 2. Run Locally in the Playground

```bash
agents-cli playground
# Or directly with ADK:
uv run adk web . --port 8000
```

Try these three flows in the UI (`http://127.0.0.1:8000/dev-ui/?app=app`):
1. **Zero-typing Link Capture:** Paste `https://github.com/google/adk-python` and press Enter.
2. **Zero-typing Screenshot Capture:** Click `+` -> **Upload local file** and attach a full-window screenshot of any public technical doc (including the browser address bar).
3. **Parallel Trail + Web Q&A:** Ask *"How do the four session state prefixes work in Google ADK, and what have I saved so far?"*

### 3. Run Linter, Unit Tests, and Evaluations

```bash
# Lint & offline unit/workflow tests (no cloud credentials required)
uv run ruff check app tests/unit tests/integration
uv run pytest tests/unit -v

# Live integration tests (requires active ADC / API key)
uv run pytest tests/integration -v

# Run LLM-as-judge evaluation suite
agents-cli eval run
```

### 4. Deploy to Vertex AI Agent Runtime

```bash
agents-cli deploy
```
