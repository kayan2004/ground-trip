FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app/backend

# libgomp1 (GNU OpenMP runtime) is required by lightgbm's native extension at
# model-load time - without it, recommend_destinations() fails on every
# request once RANKER_ENABLED=true (the default) and a trained ranker
# artifact is present, with "OSError: libgomp.so.1: cannot open shared
# object file". python:3.14-slim doesn't ship it.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY . .
RUN chmod +x docker-entrypoint.sh

# Non-root: creates a real user (with a home dir - `uv` wants a writable
# HOME to resolve into even with --frozen) and hands over ownership of
# everything under the working directory, including the venv `uv sync`
# just built as root.
RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app/backend
USER appuser

EXPOSE 8000

# urllib is stdlib - no extra package needed just for a health probe.
# Non-zero exit (an exception, here) is exactly what HEALTHCHECK needs to
# see as "unhealthy".
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD ["python3", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"]

CMD ["./docker-entrypoint.sh"]
