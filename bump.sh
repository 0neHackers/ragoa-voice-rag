#!/usr/bin/env bash
# Version bump helper — implements master prompt section 1.3 steps 1-4, 6, 7.
# Usage: ./bump.sh major|minor
# Steps 5 (make the change), 8 (changelog entry) and 10 (commit) stay manual by design:
# they require judgement this script cannot supply.
set -euo pipefail
cd "$(dirname "$0")"

KIND="${1:?usage: bump.sh major|minor}"
CUR="$(tr -d '[:space:]' < VERSION)"
MAJ="${CUR%%.*}"; MIN="${CUR##*.}"

case "$KIND" in
  major) NEW="$((MAJ + 1)).0" ;;
  minor) NEW="${MAJ}.$((MIN + 1))" ;;
  *) echo "bump kind must be 'major' or 'minor'" >&2; exit 1 ;;
esac

SRC="V${CUR}"; DST="V${NEW}"
[ -d "$SRC" ] || { echo "source version folder $SRC missing" >&2; exit 1; }
[ -e "$DST" ] && { echo "$DST already exists — refusing to overwrite a frozen snapshot" >&2; exit 1; }

mkdir -p "$DST"
# Copy everything except build artifacts: the venv, caches, and downloaded models are
# reproducible from requirements.txt and must not be duplicated into every snapshot.
tar -c --exclude='.venv' --exclude='__pycache__' --exclude='.pytest_cache' \
       --exclude='hf_cache' --exclude='model_cache' --exclude='corpus_cache' \
       -C "$SRC" . | tar -x -C "$DST"

printf '%s' "$NEW" > VERSION
printf '%s' "$NEW" > "$DST/VERSION"

echo "bumped $CUR -> $NEW  (work now happens in $DST/, $SRC/ is frozen)"
