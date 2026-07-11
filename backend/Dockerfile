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

EXPOSE 8000

CMD ["uv", "run", "--frozen", "--no-dev", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
