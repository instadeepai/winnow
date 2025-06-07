FROM python:3.12-slim AS winnow

# Create workdir
WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.6.6 /uv /uvx /bin/

# Install git
RUN apt-get update && \
    apt-get install -y git && \
    apt-get install -y tmux && \
    rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN uv venv -p 3.12

#  Copy dependency files
COPY pyproject.toml pyproject.toml
COPY uv.lock uv.lock

#  Copy module files
COPY winnow winnow
COPY configs configs
RUN mkdir checkpoints
RUN mkdir logs

# Install project
RUN uv sync --locked --group cuda

ENV XLA_PYTHON_CLIENT_PREALLOCATE=false

CMD ["/bin/bash"]