## ML Workflow

This backend now includes a complete travel-style classification workflow built from `data/travel_destinations_labeled.csv`.

### Labels

The six target classes are:

- `Adventure`
- `Relaxation`
- `Culture`
- `Budget`
- `Luxury`
- `Family`

The labeled CSV is treated as a curated assignment artifact. It preserves the original destination features and adds:

- `travel_style`
- `label_status`
- `label_notes`

### Features

Training excludes `destination`, `country`, and the label/audit columns. The model uses:

- Categorical: `region`, `budget_level`, `tourism_level`
- Binary numeric: `has_hiking`, `has_beach`
- Continuous numeric: `culture_score`, `luxury_score`, `family_friendly`, `nightlife_level`, `avg_temp_peak`

### Training

Run the notebook from `backend/notebook/ml.ipynb`. The notebook now contains the full flow:

- exploratory data analysis
- compares Logistic Regression, Random Forest, and SVC
- uses 5-fold stratified cross-validation
- tunes Random Forest with grid search
- selects the winner by macro F1
- saves the trained model with `joblib`

### Artifacts

Training outputs are written to `artifacts/ml/`:

- `results.csv`
- `classification_report.json`
- `model_reports.json`
- `model_metadata.json`
- `best_model.joblib`

### Inference

Use the self-contained `predict_travel_style()` helper inside the notebook with a single destination-shaped feature dictionary to get a predicted travel style and probabilities when supported by the model.

## API Skeleton

The backend now has an initial FastAPI skeleton with:

- typed settings in `app/core/config.py`
- lifespan-managed app state in `app/core/lifespan.py`
- async database engine/session wiring in `app/db/`
- a starter health route in `app/api/routes/health.py`

Run it from the `backend` directory with:

```powershell
uv run python main.py
```

Then open `http://localhost:8000/health`.

### Database Foundation

The backend is now prepared for async Postgres usage with:

- `DATABASE_URL` and `DATABASE_ECHO` in settings
- a shared SQLAlchemy async engine created during lifespan startup
- a shared async session factory stored on app state
- a `get_db_session()` dependency for future routes and services

At this step, we have only added the connection layer. ORM models, migrations, and user tables come next.

### Docker Database

The project now includes a standalone Postgres service in the root [docker-compose.yaml](/abs/path/c:/Users/Kayan/OneDrive/Desktop/SE%20Factory/smart_travel_assistant/docker-compose.yaml).

- service name: `db`
- image: `pgvector/pgvector:0.8.2-pg17`
- named volume: `postgres_data`
- init script: `db/init/01-enable-pgvector.sql`

Run only the database with:

```powershell
docker compose up db
```

This keeps Postgres separate from the backend service, which matches the assignment structure. The pgvector extension is created automatically the first time the database volume is initialized.

### First ORM Table

We have now added the first SQLAlchemy ORM model:

- `users` in `app/db/models/user.py`

Originally the backend created tables automatically at startup with `Base.metadata.create_all(...)`. That bootstrap step has been replaced entirely by Alembic migrations - see "Database Migrations (Alembic)" below. `create_all()` is no longer called anywhere; every table, on every DB, is created by a migration.

### Auth Foundation

We have started auth in the smallest useful way:

- `app/schemas/auth.py` defines the request/response data shapes
- `app/core/security.py` handles password hashing and verification

We now also have the first auth route:

- `POST /auth/signup`
- `POST /auth/login`
- `GET /auth/me`

It:

- validates the input with `UserCreate`
- normalizes the email to lowercase
- rejects duplicate emails
- hashes the password before storing it
- returns the created user with `UserRead`
- verifies login credentials against the stored password hash
- rejects invalid credentials with a `401`
- returns a JWT bearer token from login
- resolves the current user from the bearer token on `/auth/me`

The HTTP route stays thin, while the user-creation logic now lives in `app/services/auth.py`.

We are still intentionally keeping auth small. This is now a minimal JWT-based auth flow suitable for the later React frontend.

### Agent Run Persistence

We have now added the first authenticated domain entity after users:

- `agent_runs` in `app/db/models/agent_run.py`

This gives us a simple persisted record of:

- who initiated a run
- what prompt they sent
- what response was stored
- what status the run finished with
- when it happened

There is also a protected route:

- `POST /agent-runs`

It creates a placeholder agent run for the authenticated user and now also creates a linked `tool_log` record. This lets us verify both user-scoped run persistence and tool-level logging before building the full agent/tool workflow.

### Tool Logs

We have now added:

- `tool_logs` in `app/db/models/tool_log.py`

Each tool log belongs to an `agent_run` and stores:

- `tool_name`
- `input_payload`
- `output_payload`
- `status`
- `created_at`

This is the persistence hook we will later use for the classifier, RAG, and live-data tools.

## Database Migrations (Alembic)

Every table in this project - `users`, `agent_runs`, `tool_logs`, `destination_documents`,
`destinations`, `recommendations`, `feedback`, `tag_definitions` - is now created by an Alembic
migration. `Base.metadata.create_all()` has been removed from the app startup path
(`app/core/lifespan.py`) entirely; there is no fallback table creation left.

Migration chain, oldest first:

1. `0e2bdbc1cc5a_create_destinations_table.py` - `destinations` (+ `CREATE EXTENSION vector`)
2. `5f2b7a3d9c14_create_baseline_tables.py` - `users`, `agent_runs`, `tool_logs`,
   `destination_documents` (the tables that used to rely on `create_all()`)
3. `9e4d1f6a8b02_create_ml_feedback_tables.py` - `recommendations`, `feedback`,
   `tag_definitions`, plus `deleted_at` columns on `destinations` and `agent_runs`

`alembic/env.py` sources `sqlalchemy.url` from `app.core.config.get_settings().database_url` (never
hardcoded) and builds `target_metadata` from both declarative bases in the project
(`app.db.base.Base` and `app.db.models.destination.DestinationCorpusBase`), so `alembic check` and
future autogenerate runs have the full schema to diff against. Migrations so far are hand-written
rather than autogenerated, mainly because pgvector's HNSW index type and vector ops classes aren't
reflected by Alembic's autogenerate.

### Common commands

```powershell
# Apply all pending migrations
uv run alembic upgrade head

# Roll back one migration
uv run alembic downgrade -1

# Create a new migration (edit the generated file - autogenerate won't catch
# pgvector-specific DDL like HNSW indexes)
uv run alembic revision -m "add some_table"

# Show current revision / full history
uv run alembic current
uv run alembic history
```

### Bootstrapping an existing, already-populated DB

If your DB already has `users`, `agent_runs`, `tool_logs`, and `destination_documents` from the old
`create_all()` path (true for every DB that ever ran this app before this change), do **not** run
`alembic upgrade head` directly - migration `5f2b7a3d9c14` would try to `CREATE TABLE users` etc.
and fail with "relation already exists". Instead:

```powershell
# 1. Mark the baseline migration as already applied (it matches your
#    existing create_all()-created schema) WITHOUT running its SQL.
#    Use the explicit revision id, not `head` - `head` is one migration
#    further along (9e4d1f6a8b02) and does need to actually run.
uv run alembic stamp 5f2b7a3d9c14

# 2. Now run the real upgrade: this creates recommendations/feedback/
#    tag_definitions and adds the two new deleted_at columns.
uv run alembic upgrade head
```

If your DB has never run this app before (a true fresh DB), just run `uv run alembic upgrade head`
- it will run all three migrations in order.

## Destination Corpus Ingestion (`destinations` table)

A second, richer destination corpus lives alongside the original `travel_destinations_labeled.csv` /
`destination_documents` RAG table (both left untouched). It is not yet wired into the agent - this
is the ingestion pipeline only.

### Schema

`destinations` (Alembic-managed - see "Database Migrations (Alembic)" below):

- `id` (UUID PK), `name`, `country`, `region`, `budget_level` (`low`/`medium`/`high`)
- `details` - the composed text that actually gets embedded
- `raw_sources` (JSONB) - unembedded raw per-source text/data, for future re-composition
- `source_provenance` (JSONB) - which source (or failure) produced each field
- `embedding` (`vector(1024)`), `embedding_model`, `embedding_version`
- `content_hash` - sha256 of `details`, used to skip re-embedding unchanged rows
- Indexes: HNSW on `embedding` (`vector_cosine_ops`), btree on `region` and `budget_level`,
  unique on `(name, country)` (the idempotent upsert key)

### Why `Destination` stays on its own declarative base

`destinations` was the first table in the project managed by Alembic rather than
`Base.metadata.create_all()`, back when `create_all()` still handled everything else. The
`Destination` model (`app/db/models/destination.py`) intentionally lives on its **own**
`DeclarativeBase` (`DestinationCorpusBase`), separate from `app.db.base.Base`. That separation
predates and is independent of `create_all()` having since been removed entirely (see "Database
Migrations (Alembic)" below) - it stays because the two model bases represent two conceptually
separate corpora (the legacy `destination_documents` RAG table vs. the richer `destinations`
corpus), not because of a migration-ownership concern anymore. `alembic/env.py` still builds
`target_metadata` from both bases, so autogenerate sees the full schema either way.

### Sources

- **Wikivoyage** - primary destination prose, scraped with the same extraction logic as the
  existing RAG ingestion (`app/services/rag_ingestion.py::_extract_main_text`, reused directly).
- **OpenTripMap** - POI `kinds` aggregated within a radius of the geocoded destination, appended
  to `details` as a one-line summary; raw kind counts kept in `raw_sources`. Requires a free key
  from [opentripmap.io](https://opentripmap.io/product) in `OPENTRIPMAP_API_KEY`. **Not configured
  in this environment** - the step is skipped (not failed) when the key is blank, and `details`
  composes from Wikivoyage + region line only.
- **Numbeo** - has no free API. Instead of scraping per-city pages (which don't expose a numeric
  index without JS), the pipeline fetches Numbeo's public `rankings_current.jsp` table **once per
  run** (~550 cities) and buckets `budget_level` by quartile of that run's index values
  (`< Q1` = low, `Q1-Q3` = medium, `> Q3` = high). Cities not in that ranking table get
  `budget_level = null` - this is expected, not a bug, for smaller/less-common destinations.
- **Open-Meteo geocoding** - substituted for the spec's "optional GeoNames": it's free, keyless,
  and already used by `services/live_conditions.py`, so it resolves canonical lat/lon (needed for
  the OpenTripMap radius query) without introducing a second geocoding provider.

Every source fetch is retried with exponential backoff (`DESTINATION_MAX_RETRIES`,
`DESTINATION_RETRY_BACKOFF_SECONDS`) and failures are isolated per source per destination - a failed
Wikivoyage fetch does not block OpenTripMap/Numbeo for that destination, and a failed destination
does not block the rest of the run. Embedding failures (e.g. an invalid Voyage key) degrade the
same way: rows are still upserted with `embedding = null` and get picked up automatically on the
next successful run via the content-hash cache.

### Seed manifest

`data/destination_seed_manifest.json` - 219 hand-curated, real destinations (`name`, `country`,
`region`, `wikivoyage_url`), versioned and committed, mirroring `rag_source_manifest.json`'s
pattern. Deliberately does **not** bake in coordinates or OpenTripMap/Numbeo identifiers - those
are resolved dynamically per run so a bad guess never gets committed to the manifest.

### Running ingestion from empty

```powershell
# 1. Bring up Postgres (pgvector image) if it isn't already running
docker compose up -d db

# 2. Apply migrations (creates every table, including destinations - see
#    "Database Migrations (Alembic)" above for the existing-DB bootstrap
#    sequence if this isn't a fresh database)
uv run alembic upgrade head

# 3. Set VOYAGE_API_KEY (required) and optionally OPENTRIPMAP_API_KEY in .env

# 4. Smoke-test on a handful of destinations first
uv run python scripts/ingest_destinations.py --limit 5

# 5. Full run (219 destinations; respects VOYAGE_REQUESTS_PER_MINUTE, so budget several minutes)
uv run python scripts/ingest_destinations.py
```

Re-running is always safe: the upsert key is `(name, country)`, and unchanged `details` skip
re-embedding entirely via `content_hash`.

### Data-quality report

Every run writes `artifacts/destinations/data_quality_report.{json,csv}`: destination count per
region, missing-field rates (`budget_level`, `poi_summary`, `wikivoyage_summary`, `embedding`),
`details` length distribution, and a per-source failure count. The committed artifact reflects a
real 5-destination run (Paris, Lyon, Nice, Marseille, Bordeaux) with all three sources and the
embedding provider live - all missing-field rates are 0.0 and `sources_failed_counts` is empty. A
re-run over the same 5 immediately afterwards produced `embedded_count: 0`,
`skipped_embedding_count: 5` - the content-hash cache, confirmed working with real (not synthetic)
embeddings.

Note: an earlier version of the OpenTripMap integration silently mis-parsed the API's default
GeoJSON response (its published schema nests `kinds` under a doubled `properties.properties` key);
`_fetch_opentripmap_pois` now requests `format=json` and reads the documented flat `SimpleFeature`
list instead, verified against the live API.

## ML Feedback Schema (`recommendations`, `feedback`, `tag_definitions`)

Schema-only groundwork for learning-to-rank over recommended destinations. Not yet wired into the
agent or any route - no code writes to these tables yet.

- **`recommendations`** (`app/db/models/recommendation.py`) - the **full ranked slate** shown for
  an agent run, one row per destination position, not just the destination the user picked. This
  is deliberate: learning-to-rank needs the whole slate (including what was shown but not chosen),
  not just positive examples.
  - `agent_run_id` (FK -> `agent_runs.id`), `destination_id` (FK -> `destinations.id`, a **UUID**,
    matching that table's PK - not an int), `rank_position`, `score`
  - `features` (JSONB, **not null**) - a snapshot of the ranker's feature row *at recommend time*.
    This is the most important column here: weather, prices, and other live signals drift after
    the fact, so if `features` isn't captured at the moment of recommendation, training data
    quietly desyncs from what the model actually saw. Never derive this column lazily from live
    state later.
  - `deleted_at` (soft delete)
- **`feedback`** (`app/db/models/feedback.py`) - a verdict on one `recommendation` row.
  - `recommendation_id` (FK -> `recommendations.id`), `session_uuid` (an anonymous client UUID -
    **not** a `users` FK, so feedback works without an authenticated session), `verdict`
    (`smallint`, `+1`/`-1`, not null)
  - Partial index `ix_feedback_recommendation_id_verdict_not_null` on `(recommendation_id) WHERE
    verdict IS NOT NULL` - currently equivalent to a plain index since `verdict` is `NOT NULL` at
    the column level, kept as specified for forward-compatibility if that constraint is ever
    relaxed (e.g. a withdrawn-feedback state)
  - `deleted_at` (soft delete)
- **`tag_definitions`** (`app/db/models/tag_definition.py`) - human/LLM-readable labels for
  clusters produced by some (not-yet-built) offline clustering step.
  - `cluster_id` (unique), `tag_name`, `description` (LLM-generated rationale), `quality_metrics`
    (JSONB - e.g. silhouette score, cluster size, noise ratio)
  - No `deleted_at` - not part of the per-run audit trail the other two tables are.

`recommendations.destination_id` has no ORM `relationship()` to `Destination`: that model lives on
its own declarative base/registry (see "Why `Destination` stays on its own declarative base"
above), so `relationship()` can't resolve it by class name across bases. The FK column and its DB
constraint exist regardless - only the ORM-level convenience accessor is skipped.
