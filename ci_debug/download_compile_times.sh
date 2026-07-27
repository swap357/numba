#!/usr/bin/env bash
# Download every compile-times-* artifact for a commit into ONE directory,
# ready for analyze_compile_times.py / chart_compile_times.py.
#
# Each wheel-builder platform is a separate workflow run, so we enumerate all
# runs for the commit and pull the matching artifacts from each into $DEST.
# The `-p 'compile-times-*'` glob skips the (large) wheel artifacts.
#
# Usage: ci_debug/download_compile_times.sh <sha> [dest-dir] [repo]
set -euo pipefail

SHA="${1:?usage: download_compile_times.sh <sha> [dest-dir] [repo]}"
DEST="${2:-/tmp/compile_times_${SHA:0:9}}"
REPO="${3:-swap357/numba}"

rm -rf "$DEST"; mkdir -p "$DEST"

runs=$(gh api "repos/$REPO/actions/runs?head_sha=$SHA&per_page=100" \
         --paginate -q '.workflow_runs[].id')

for run in $runs; do
  gh run download "$run" --repo "$REPO" -p 'compile-times-*' -D "$DEST" \
    2>/dev/null || true   # runs with no compile-times artifacts are skipped
done

n=$(find "$DEST" -name compile_times.txt | wc -l | tr -d ' ')
echo "downloaded $n compile_times.txt files -> $DEST"
