# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Session memory

Before exploring the codebase, read `.claude/memory/state.md` - a living snapshot of what's
currently implemented, in progress, or configured that wouldn't be obvious from the code alone
(e.g. credential state, deferred decisions). `.claude/memory/sessions/` has one dated log per past
session with more narrative detail if `state.md` isn't enough context.

After a session where something meaningful changed (a feature landed, a non-obvious bug was found,
a credential/environment fact changed, a design decision was made and deferred), update
`state.md` and add a dated entry under `sessions/`. See `.claude/memory/README.md` for the full
convention and what does/doesn't belong there vs. this file vs. code comments.

## What this is

Smart Travel Planner (AI bootcamp Week 4 project, see `brief.md`): a natural-language trip
request flows through LLM field-extraction (provider-agnostic, Gemini by default) → a structured
pre-filter + pgvector cosine destination recommender → pgvector RAG retrieval → live weather
(Open-Meteo) → LLM synthesis → Postgres persistence → Discord webhook. The SVC travel-style
classifier that used to sit in this pipeline was fully removed 2026-07-11 (see `backend/MODEL_CARD.md`).
See `README.md` for the full write-up (chunking rationale, model comparison, known gaps).

## Stack

- **Backend**: FastAPI, SQLAlchemy 2.x async (asyncpg), LangGraph, pydantic-settings, PyJWT,
  pgvector, scikit-learn/joblib, umap-learn/hdbscan (offline clustering only), uv for dependency
  management, Python 3.14.
- **Frontend**: React 19 + TypeScript + Vite, no router library (manual `pushState`).
- **Infra**: Docker Compose — `db` (pgvector/pgvector:0.8.2-pg17), `backend`, `frontend` (nginx).

## Running locally

```powershell
# Backend (from backend/)
uv run uvicorn main:app --reload

# Frontend (from frontend/)
npm install
npm run dev

# Full stack (from repo root)
docker compose up --build
```

Backend needs `backend/.env` (copy from `backend/.env.example`) with `DATABASE_URL`,
`JWT_SECRET_KEY`, `VOYAGE_API_KEY`, `ANTHROPIC_API_KEY`, `DISCORD_WEBHOOK_URL`,
`FRONTEND_ORIGIN`.

RAG ingestion and eval scripts (from `backend/`):

```powershell
uv run python scripts/ingest_rag.py       # fetches Wikivoyage pages, embeds, writes to pgvector
uv run python scripts/evaluate_rag.py     # runs hand-written eval queries, writes artifacts/rag/*
```

Destination corpus (v2, not yet wired into the agent — see backend/README.md):

```powershell
uv run alembic upgrade head                          # applies all migrations (every table)
uv run python scripts/ingest_destinations.py --limit 5  # smoke test; omit --limit for the full ~219
```

Offline destination clustering (weighted travel-style tags, not wired into the agent — see
backend/README.md's "Destination Clustering" section): three separable phases,
`uv run python scripts/cluster_destinations.py {cluster,name,apply-tags}`. Requires >=50
destinations with a non-null `embedding`; aborts otherwise.

## Architecture

```
backend/app/
├── core/       # Settings (pydantic-settings), lifespan singletons, JWT/password hashing
├── db/         # SQLAlchemy async engine/session, ORM models (users, agent_runs, tool_logs,
│               #   destination_documents, recommendations, feedback, tag_definitions — all on
│               #   app.db.base.Base). `destination.py`'s `Destination` model is on its own
│               #   DeclarativeBase (DestinationCorpusBase) — see alembic/ below
├── agent/      # LangGraph state machine (graph.py) + BaseTool/ToolRegistry (tools/)
├── api/routes/ # Thin FastAPI routers, one file per concern, all behind JWT auth
├── schemas/    # Pydantic models — the validation boundary for every route/tool
├── services/   # Business logic: llm (extraction+synthesis+cluster naming, provider-agnostic
│               #   via llm_providers/'s LLMProvider interface - Anthropic/Gemini),
│               #   destination_recommendations (pre-filter + cosine, the essential recommender),
│               #   ranker/ranker_training (optional LightGBM re-rank, RANKER_ENABLED-gated),
│               #   clustering (offline UMAP+HDBSCAN, scripts/cluster_destinations.py only),
│               #   discord_webhook, live_conditions, rag_ingestion, rag_retrieval,
│               #   voyage_embeddings. No ML classifier here anymore - removed 2026-07-11.
└── prompts/    # Raw prompt templates (request_field_extraction_prompt.txt)
```

**Lifespan singletons** (`app/core/lifespan.py`, exposed via `app.state.resources`): DB engine,
session factory, shared `httpx.AsyncClient`, loaded ML model, destination catalog DataFrame,
tool registry. Routes/services get these via `Depends` or `request.app.state`, never by
instantiating clients inline.

**LangGraph pipeline** (`app/agent/graph.py`): `initialize → extract_request_fields → classify
→ recommend_destinations → retrieve_context → live_conditions → synthesize_response`. Every
node appends to `tool_logs` and `response_sections` in the shared `TripPlannerState` dict, and
catches its own exceptions — a failed/skipped tool degrades the run to `status="partial"`
rather than crashing. `run_trip_planner()` (`app/agent/planner.py`) is the entrypoint invoked
from `services/agent_runs.py`.

**Tools** (`app/agent/tools/`): each tool is a `BaseTool` subclass with a `name`, Pydantic
`input_model`, and async `arun(payload, context)`. Registered in `registry.py`
(`build_default_tool_registry`) — this is the allowlist; nothing outside it is callable.

**Two-model routing** (`services/llm.py`): `choose_model()` picks the fast vs. strong model of
whichever provider is configured (`LLM_PROVIDER`) based on prompt length, number of failed tools,
and response richness. Fast model does field extraction; strong model does final synthesis and
cluster naming. Provider dispatch lives in `services/llm_providers.py`'s `LLMProvider` interface
(`AnthropicProvider`, `GeminiProvider`) - see backend/README.md's "Provider-Agnostic LLM Layer".

## Conventions to follow when editing

- **Pydantic at every boundary**: HTTP bodies, tool inputs/outputs, LLM structured output all go
  through a schema in `app/schemas/`. Don't add ad-hoc dict validation elsewhere.
- **Async everywhere**: routes, services, DB calls (asyncpg), HTTP calls
  (`httpx.AsyncClient`, shared instance from lifespan — never construct a new client per
  request). No `requests`, no `time.sleep` in a request path.
- **No globals for state**: singletons live on `app.state.resources` and flow through
  `Depends()` or `ToolContext`, not module-level variables.
- **Settings only through `app/core/config.py`**: no `os.getenv` scattered in code; add new
  config keys to the `Settings` class and `.env.example`.
- **Tool failures are data, not exceptions**: a tool failing inside the LangGraph should produce
  a `tool_logs` entry with `status="failed"` and let the graph continue — see any `_node`
  function in `graph.py` for the pattern.
- **New agent tool checklist**: add a Pydantic input/output schema in `schemas/`, implement a
  `BaseTool` subclass in `agent/tools/`, register it in `registry.py`, add a graph node in
  `graph.py` that logs to `tool_logs`/`response_sections`, and (if user-facing) a route in
  `api/routes/`.

## Known gaps (see README "Known Gaps" for the full list)

**Discord webhook retry-with-backoff now exists** (2026-07-06, `app/services/discord_webhook.py`) -
retries `429`/`5xx`/network errors, does not retry other `4xx` (permanently broken webhook URL). If
another integration needs the same pattern, this is the one to extend, alongside
`app/services/voyage_embeddings.py`'s own retry loop (real-vs-permanent-failure split, `Retry-After`
handling).

**Token/cost logging and structured tracing now exist** (2026-07-06) - real Python `logging`
(structured, via `extra={}`), not LangSmith/OpenTelemetry (no new external account/service). The
LLM provider layer logs token counts + an estimated dollar cost per call
(`app/services/llm_providers/usage_logging.py`); `app/services/tool_logs.py`'s `create_tool_log()`
logs every tool execution in the trip-planner pipeline (the one place graph nodes, recommendation
persistence, and Discord delivery all pass through). `configure_logging()`
(`app/core/logging_config.py`) must be called for these to actually emit - already wired into
`main.py` and `scripts/cluster_destinations.py`'s `name` phase (the only offline script that makes
LLM calls); a new script that does the same needs the same call.

**Automated tests + CI now exist** (`backend/tests/`, `.github/workflows/ci.yml`) - pytest +
pytest-asyncio against a dedicated test Postgres (never the dev DB), truncate-based isolation
(rollback-based isolation does NOT work here - several services commit internally), every external
HTTP boundary mocked. `backend/tests/conftest.py` is the pattern to extend for new test coverage;
see `backend/README.md`'s "Running Tests" section for the full write-up.

All tables are now Alembic-managed (`backend/alembic/`); `Base.metadata.create_all()` has been
removed from startup. `target_metadata` in `alembic/env.py` is a list of both declarative bases
(`app.db.base.Base` and `app.db.models.destination.DestinationCorpusBase`). See backend/README.md's
"Database Migrations (Alembic)" section for the migration chain and the stamp sequence needed on a
DB that predates this change.

## Data / artifacts (don't regenerate casually)

- The SVC travel-style classifier (`artifacts/ml/`, `services/classifier.py`,
  `travel_destinations_labeled.csv`) that used to be documented here was fully removed 2026-07-11
  - it had been dormant/unreachable since 2026-07-05. See `backend/MODEL_CARD.md`'s intro for the
  recovery path if it's ever needed again (`git log --diff-filter=D -- backend/app/services/classifier.py`).
- `backend/artifacts/rag/`, `backend/data/rag_eval_queries.json` — retrieval eval fixtures/output
  from `scripts/evaluate_rag.py`.
- `backend/data/destination_seed_manifest.json` — versioned seed list (219 destinations) for the
  `destinations` corpus; `backend/artifacts/destinations/data_quality_report.*` is its pipeline
  output, from `scripts/ingest_destinations.py`.
