#!/usr/bin/env bash
# Script to update coverage badge in README
# Used by GitHub Actions to automatically update coverage badge on push

set -e  # Exit on any error

echo "🔍 Extracting coverage percentage..."

# Run tests with coverage and capture the percentage
COVERAGE_OUTPUT=$(uv run pytest tests --cov=winnow --cov-report=term-missing --cov-fail-under=0 --quiet)
COVERAGE_PERCENT=$(echo "$COVERAGE_OUTPUT" | grep -o "TOTAL.*[0-9]*%" | grep -o "[0-9]*%" | sed 's/%//')

if [ -z "$COVERAGE_PERCENT" ]; then
    echo "❌ Could not extract coverage percentage"
    exit 1
fi

echo "📊 Coverage: ${COVERAGE_PERCENT}%"

# Determine badge color based on coverage
if [ "$COVERAGE_PERCENT" -ge 80 ]; then
    COLOR="brightgreen"
    EMOJI="🟢"
elif [ "$COVERAGE_PERCENT" -ge 60 ]; then
    COLOR="yellow"
    EMOJI="🟡"
else
    COLOR="red"
    EMOJI="🔴"
fi

# Generate badge URL
BADGE_URL="https://img.shields.io/badge/coverage-${COVERAGE_PERCENT}%25-${COLOR}?logo=coverage"

echo "${EMOJI} Badge URL: ${BADGE_URL}"

# Update README with the new badge
sed -i "s|https://img.shields.io/badge/coverage-[0-9]*%25-[a-z]*?logo=coverage|${BADGE_URL}|g" README.md

echo "✅ README updated with coverage badge: ${COVERAGE_PERCENT}%"
