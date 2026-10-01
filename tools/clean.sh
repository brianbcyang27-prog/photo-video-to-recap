#!/usr/bin/env bash
# Remove scratch files: test renders, orphaned render directories, caches.
#
# Safe to run mid-session. NOT safe to run while a render is in flight, and it
# refuses to rather than finding out for you: the render keeps its segments and
# stills in .work/run-<pid>-<n>/, and deleting that out from under it kills the
# run several minutes in, after the analysis it cannot redo has finished.
#
#   ./tools/clean.sh          # scratch only, keeps your finished videos
#   ./tools/clean.sh --all    # also drop everything in out/, including videos
set -euo pipefail
cd "$(dirname "$0")/.."

# ---- refuse if anything is rendering -------------------------------
if pgrep -f "make_video\.py" >/dev/null 2>&1; then
  echo "A render is running (pids: $(pgrep -f 'make_video\.py' | tr '\n' ' '))."
  echo "Wait for it to finish, or stop it first. Cleaning now would delete the"
  echo "run directory it is writing to."
  exit 1
fi

freed=0
freed=$((freed + $(du -sk out 2>/dev/null | cut -f1 || echo 0)))

# ---- .work: every run dir is orphaned once nothing is rendering ----
if [ -d .work ]; then
  find .work -mindepth 1 -maxdepth 1 -name 'run-*' -print -exec rm -rf {} + 2>/dev/null || true
fi

# ---- python and tool caches ----------------------------------------
rm -rf __pycache__ pipeline/__pycache__ tools/__pycache__ .pytest_cache
find . -name '*.pyc' -not -path './.venv/*' -delete 2>/dev/null || true
rm -f .DS_Store
freed=$((freed + 0))

# ---- out/ ----------------------------------------------------------
if [ "${1:-}" = "--all" ]; then
  echo "Removing everything in out/, including finished videos."
  rm -rf out
else
  # Keep videos and their sidecars; drop the reports and EDLs of renders you
  # are unlikely to look at again.
  rm -f out/*.md
  find out -name '*.edl.json' -mtime +7 -delete 2>/dev/null || true
fi

echo "Cleaned. out/ now holds:"
ls -1 out 2>/dev/null | sed 's/^/  /' || echo "  (empty)"
echo
echo "Reclaim more with:  rm -rf out .cache test_media* .work"
echo "Test libraries regenerate with tools/make_test_media.py"