#!/usr/bin/env bash
#
# Clean up stale openpi/orbax training checkpoints to reclaim scratch quota.
#
# A "run directory" is any directory whose immediate children include one or
# more purely-numeric step directories (openpi's checkpoint layout is
# checkpoints/<exp>/<run>/<step>/, e.g. .../yor_icl_pi05_expanded_full/49999).
# For each run directory this script keeps the N most recent (highest-step)
# *complete* checkpoints and deletes the rest. A checkpoint is "complete"
# only if it contains orbax's _CHECKPOINT_METADATA marker file; anything
# without it (still being written) is left alone.
#
# Defaults to a dry run. Pass --apply to actually delete.
#
# Usage:
#   clean_stale_checkpoints.sh [OPTIONS] [ROOT_DIR ...]
#
# Options:
#   --keep N        Number of most recent checkpoints to keep per run (default: 1)
#   --min-age MIN   Skip checkpoints modified within the last MIN minutes,
#                   as a safety margin against a job that is still writing
#                   (default: 15)
#   --apply         Actually delete. Without this, only prints what would happen.
#   -h, --help      Show this help.
#
# If no ROOT_DIR is given, defaults to third_party/openpi/checkpoints
# relative to the repo root (resolved from this script's location).

set -euo pipefail

KEEP=1
MIN_AGE_MIN=15
APPLY=0
ROOTS=()

usage() {
    sed -n '2,29p' "$0" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --keep)
            KEEP="$2"
            shift 2
            ;;
        --min-age)
            MIN_AGE_MIN="$2"
            shift 2
            ;;
        --apply)
            APPLY=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            ROOTS+=("$1")
            shift
            ;;
    esac
done

if ! [[ "$KEEP" =~ ^[0-9]+$ ]] || [[ "$KEEP" -lt 1 ]]; then
    echo "error: --keep must be a positive integer" >&2
    exit 1
fi

if [[ ${#ROOTS[@]} -eq 0 ]]; then
    SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
    REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." &>/dev/null && pwd)"
    ROOTS=("$REPO_ROOT/third_party/openpi/checkpoints")
fi

if [[ "$APPLY" -eq 1 ]]; then
    echo "MODE: APPLY (checkpoints will be permanently deleted)"
else
    echo "MODE: DRY RUN (pass --apply to actually delete)"
fi
echo "keep=$KEEP most-recent checkpoint(s) per run, min-age=${MIN_AGE_MIN}min"
echo

total_bytes_freed=0
now_epoch=$(date +%s)

for root in "${ROOTS[@]}"; do
    if [[ ! -d "$root" ]]; then
        echo "warn: root '$root' does not exist, skipping" >&2
        continue
    fi
    echo "== scanning $root =="

    # Find every directory that has at least one purely-numeric child directory:
    # that marks it as an openpi checkpoint run directory.
    while IFS= read -r -d '' run_dir; do
        # Collect complete numeric step dirs (must contain _CHECKPOINT_METADATA).
        steps=()
        while IFS= read -r -d '' step_dir; do
            step_name="$(basename -- "$step_dir")"
            if [[ -f "$step_dir/_CHECKPOINT_METADATA" ]]; then
                steps+=("$step_name")
            fi
        done < <(find "$run_dir" -mindepth 1 -maxdepth 1 -type d -regextype posix-extended -regex '.*/[0-9]+' -print0)

        [[ ${#steps[@]} -le "$KEEP" ]] && continue

        # Sort numerically descending; the first KEEP are kept, the rest are stale.
        IFS=$'\n' sorted=($(printf '%s\n' "${steps[@]}" | sort -rn))
        unset IFS
        stale=("${sorted[@]:$KEEP}")

        for step_name in "${stale[@]}"; do
            step_dir="$run_dir/$step_name"
            mtime=$(stat -c '%Y' "$step_dir")
            age_min=$(( (now_epoch - mtime) / 60 ))
            if [[ "$age_min" -lt "$MIN_AGE_MIN" ]]; then
                echo "  skip (too recent, ${age_min}min old): $step_dir"
                continue
            fi

            size_h=$(du -sh "$step_dir" 2>/dev/null | cut -f1)
            size_b=$(du -sb "$step_dir" 2>/dev/null | cut -f1)
            total_bytes_freed=$(( total_bytes_freed + size_b ))

            if [[ "$APPLY" -eq 1 ]]; then
                echo "  deleting (${size_h}): $step_dir"
                rm -rf -- "$step_dir"
            else
                echo "  would delete (${size_h}): $step_dir"
            fi
        done
    done < <(find "$root" -type d -regextype posix-extended -regex '.*/[0-9]+' -printf '%h\0' | sort -zu)
done

echo
human_total=$(numfmt --to=iec-i --suffix=B "$total_bytes_freed" 2>/dev/null || echo "${total_bytes_freed} bytes")
if [[ "$APPLY" -eq 1 ]]; then
    echo "Freed: $human_total"
else
    echo "Would free: $human_total (re-run with --apply to delete)"
fi
