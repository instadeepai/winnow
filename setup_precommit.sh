#!/bin/bash

# Ensure we are in the repo root
cd "$(git rev-parse --show-toplevel)" || exit 1

# Symlink the git hook
ln -sf ../../.github/hooks/pre-commit .git/hooks/pre-commit

# Install pre-commit
pre-commit install

echo "Pre-commit setup complete!"
