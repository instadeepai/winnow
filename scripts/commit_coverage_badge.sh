#!/usr/bin/env bash
# Script to commit updated coverage badge
# Used by GitHub Actions to commit badge changes

set -e  # Exit on any error

echo "🔧 Configuring git..."

# Configure git for GitHub Actions
git config --local user.email "action@github.com"
git config --local user.name "GitHub Action"

echo "📝 Staging README.md..."
git add README.md

echo "🔍 Checking for changes..."
if ! git diff --staged --quiet; then
    # Extract coverage percentage for commit message
    COVERAGE_BADGE=$(grep -o 'coverage-[0-9]*%25' README.md)
    COMMIT_MSG="Update coverage badge: ${COVERAGE_BADGE}"

    echo "💾 Committing changes: ${COMMIT_MSG}"
    git commit -m "${COMMIT_MSG}"

    echo "🚀 Pushing to repository..."
    git push

    echo "✅ Coverage badge updated and committed"
else
    echo "ℹ️ No changes to coverage badge"
fi
