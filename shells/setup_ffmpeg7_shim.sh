#!/bin/bash
# ============================================================================
# (Re)generates third_party/openpi/ffmpeg7_shim/ -- plain-soname symlinks
# pointing at pyav's bundled FFmpeg 7 shared libs (third_party/openpi/.venv/
# lib/python3.12/site-packages/av.libs/), so torchcodec's dlopen("libavutil.
# so.59") etc. can find them. pyav vendors its FFmpeg build with
# auditwheel-style hash-mangled filenames (e.g. libavutil-a63ffd27.so.59.39.
# 100) that change whenever the `av` package is reinstalled/upgraded -- this
# script re-derives the mapping by globbing rather than hardcoding hashes, so
# it's safe to rerun after `uv sync` bumps `av`.
#
# See shells/slurm/victr_openpi_train_job.sh's LD_LIBRARY_PATH comment for the
# full story (torchcodec/torch ABI mismatch -> silent pyav fallback -> GPU-
# utilization bottleneck).
#
# Usage: shells/setup_ffmpeg7_shim.sh [repo_dir] [venv_dir]
#
# venv_dir defaults to third_party/openpi/.venv (openpi's JAX-training venv, the
# original caller). viktr's own venv (repo-root .venv/) has the identical issue --
# unpinned torchcodec resolved against a newer torch (2.11 vs openpi's pinned 2.7.1)
# with no system FFmpeg -- so any script running in viktr's venv that constructs a
# LeRobotDataset (e.g. scripts/build_retrieval_pool.py, scripts/
# fit_expanded_fast_tokenizer.py) should pass "$REPO_DIR/.venv" here too. Shim output
# goes to a sibling ffmpeg7_shim/ dir next to whichever venv_dir was given, so the two
# venvs' shims don't collide.
# ============================================================================
set -euo pipefail

REPO_DIR="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV_DIR="${2:-$REPO_DIR/third_party/openpi/.venv}"
AVLIBS="$VENV_DIR/lib/python3.12/site-packages/av.libs"
SHIM="$(dirname "$VENV_DIR")/ffmpeg7_shim"

if [ ! -d "$AVLIBS" ]; then
    echo "setup_ffmpeg7_shim: $AVLIBS not found -- is the venv at $VENV_DIR synced (uv sync)?" >&2
    exit 1
fi

mkdir -p "$SHIM"

# torchcodec 0.5's FFmpeg-7 target wants exactly these sonames (torchcodec's
# ffmpeg_versions.cmake, ffmpeg_major_version == 7).
SONAMES=(
    libavutil.so.59
    libavcodec.so.61
    libavformat.so.61
    libavdevice.so.61
    libavfilter.so.10
    libswscale.so.8
    libswresample.so.5
)

for soname in "${SONAMES[@]}"; do
    # av.libs names things like libavutil-<hash>.so.59.39.100 -- match by the
    # "libavutil-*.so.59.*" glob (soname's ".so.N" prefix), pick the (only)
    # match, and symlink the plain soname to it.
    prefix="${soname%%.so.*}"
    major="${soname##*.so.}"
    match=$(find "$AVLIBS" -maxdepth 1 -name "${prefix}-*.so.${major}.*" | head -1)
    if [ -z "$match" ]; then
        echo "setup_ffmpeg7_shim: no match for $soname under $AVLIBS" >&2
        exit 1
    fi
    ln -sf "$match" "$SHIM/$soname"
    echo "  $soname -> $match"
done

echo "wrote $SHIM"
