#!/bin/sh
set -e

# Applies pending migrations before the app starts serving traffic - this
# was previously only ever run in CI (against the disposable test DB), never
# against an actual deploy target. A separate `alembic upgrade head` process,
# not an in-process call from main.py's lifespan: alembic/env.py's
# run_migrations_online() calls asyncio.run() internally, which would nest
# inside the event loop FastAPI's own async startup is already running in -
# the same reason tests/conftest.py shells out to a subprocess instead of
# calling alembic.command.upgrade() directly.
uv run --frozen --no-dev alembic upgrade head

exec uv run --frozen --no-dev uvicorn main:app --host 0.0.0.0 --port 8000
