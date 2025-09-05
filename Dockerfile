FROM alpine/curl AS vscode-installer

RUN mkdir /aichor
RUN curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' --output /aichor/vscode_cli.tar.gz
RUN tar -xf /aichor/vscode_cli.tar.gz -C /aichor

# Use a lightweight Python image with uv pre-installed
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# When your experiment writes result files into a bind mount (directories mirrored using -v or --volume),
# your results will (for most containers) be read only, because docker containers run as root by default.
# This means, to clear up your result files, you need to be a root user.
# To avoid this, you can modify your Dockerfile to use a non-root user.
ARG USER=app
ARG UID=1001
ARG GID=1000
ARG HOME_DIRECTORY=/app
ARG RUNS_DIRECTORY=/runs
ARG VENV_DIRECTORY=/app/.venv

ARG VERSION=latest
ARG LAST_COMMIT=latest
ENV VERSION=$VERSION
ENV LAST_COMMIT=$LAST_COMMIT

# Avoid unnecessary writes to disk
ENV LANG=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# Enable terminal 256 colors
ENV TERM=xterm-256color

# Set application working directory
WORKDIR $HOME_DIRECTORY

# Install dependencies for Google Cloud CLI and Make commands
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        apt-transport-https \
        ca-certificates \
        gnupg \
        curl \
        make

# Install Google Cloud CLI
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    apt-get update -y && \
    apt-get install google-cloud-cli -y
RUN apt-get clean && \
    rm -rf /var/lib/{apt,dpkg,cache,log}

# Install AWS CLI
RUN uv tool install awscli

# Create group and user
RUN groupadd --force --gid $GID $USER && \
        useradd -M --home $HOME_DIRECTORY --base-dir $HOME_DIRECTORY \
        --uid $UID --gid $GID --shell "/bin/bash" $USER

# Create runs directory
RUN mkdir -p $RUNS_DIRECTORY
RUN mkdir -p $VENV_DIRECTORY

# Copy files into HOME_DIRECTORY
COPY . $HOME_DIRECTORY

# Makes HOME_DIRECTORY and RUNS_DIRECTORY files owned by USER
RUN chown -R $USER:$GID $HOME_DIRECTORY
RUN chown -R $USER:$GID $RUNS_DIRECTORY
RUN chown -R $USER:$GID $VENV_DIRECTORY

# Default user
USER $USER

# Configure path for virtual environment creation
ENV UV_PROJECT_ENVIRONMENT=$VENV_DIRECTORY

# Install dependencies from uv.lock file
RUN uv sync --frozen --no-cache --dev

# Set path to virtual environment
ENV PATH="$VENV_DIRECTORY/bin:$PATH"

COPY --from=vscode-installer /aichor /aichor
