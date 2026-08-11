# Build context is the repository root, because the API imports packages/sift_core.
#
#   docker build -f apps/api/Dockerfile -t sift-api .

FROM python:3.12-slim AS base

# GHCR attaches a package to a repository when a pushed image carries this label. An
# unlinked package is owned by whoever first pushed it, and the repository's
# GITHUB_TOKEN gets 403 on it however many `packages: write` permissions the workflow
# declares - which is exactly how the publish workflow first failed.
LABEL org.opencontainers.image.source="https://github.com/ammar2811/sift"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies are installed from the manifest alone so this layer is cached across
# source changes, which is most of the build time.
COPY pyproject.toml ./
# Dependencies come out of pyproject.toml rather than being restated here. The list was
# duplicated for layer caching, which worked until the two copies disagreed; copying only
# pyproject.toml first keeps the cached layer and leaves one source of truth.
COPY pyproject.toml ./
RUN python -c "import tomllib; print('\n'.join(tomllib.load(open('pyproject.toml','rb'))['project']['dependencies']))" > /tmp/requirements.txt \
    && pip install --no-cache-dir -r /tmp/requirements.txt

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
