# API image: FastAPI + the ML pipeline (so background jobs can run ingestion/analysis).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# CPU-only torch keeps the image far smaller than the default CUDA build.
RUN pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY ml ml
COPY backend backend
COPY migrations migrations
COPY configs configs
COPY alembic.ini ./
RUN pip install ".[api,ml]"

EXPOSE 8000
# Apply migrations, then serve. A single worker: the in-process job runner is not shared across workers.
CMD ["sh", "-c", "python -m backend.app.cli migrate && uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --workers 1"]
