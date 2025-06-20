# ----------------------------------------------------------------------------------
# Dockerfile for qna-chatbot
#
# This Dockerfile builds a multi-stage container image for the qna-chatbot
# application. It uses Python 3.12.6-slim as the base image and leverages
# uv (a fast Python package manager) for dependency management and installation.
#
# The build stage installs all dependencies and prepares the application,
# while the runtime stage creates a minimal environment for running the app
# as a non-root user.
#
# For development and production or staging entrypoints, see docker/entrypoint.sh.
# ----------------------------------------------------------------------------------


# Build stage
FROM python:3.12.6-slim-bookworm AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1 \
    PATH="/root/.local/bin:$PATH"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -sSL https://astral.sh/uv/install.sh | sh

WORKDIR /app

# Copy only dependency files
COPY pyproject.toml .

# Copy source code
COPY . .

# Sync dependencies without installing the project (no dev)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv && \
    . .venv/bin/activate && \
    uv pip install -e .

# Install the project in editable mode
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Runtime stage
FROM python:3.12.6-slim-bookworm AS runtime

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -s /bin/bash app

WORKDIR /app
RUN chown -R app:app /app

# Remove any local venv
RUN rm -rf .venv

# Copy .venv and app from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app /app

# Ensure entrypoint is executable before switching to non-root user
RUN chmod +x docker/entrypoint.sh

ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"

USER app

ENTRYPOINT ["docker/entrypoint.sh"]    