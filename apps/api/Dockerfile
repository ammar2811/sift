# Build context is the repository root, because the API imports packages/sift_core.
#
#   docker build -f apps/api/Dockerfile -t sift-api .

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies are installed from the manifest alone so this layer is cached across
# source changes, which is most of the build time.
COPY pyproject.toml ./
RUN pip install --no-cache-dir \
      "fastapi>=0.115" "uvicorn[standard]>=0.32" "httpx>=0.28" \
      "pydantic>=2.10" "pydantic-settings>=2.7" \
      "psycopg[binary,pool]>=3.2" "pgvector>=0.3" "redis>=5.2" \
      "azure-storage-queue>=12.12" "openai>=1.59" "pyyaml>=6.0"

COPY packages/ ./packages/
COPY apps/api/ ./apps/api/
COPY apps/__init__.py ./apps/__init__.py

# A non-root user is the default expectation for Container Apps and costs nothing.
RUN useradd --create-home --uid 10001 sift && chown -R sift:sift /app
USER sift

# Container Apps injects PORT; defaulting keeps `docker run` working unchanged.
ENV PORT=8000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import os,urllib.request;urllib.request.urlopen(f\"http://127.0.0.1:{os.environ.get('PORT','8000')}/health\").read()"

CMD ["sh", "-c", "uvicorn apps.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
