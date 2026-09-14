"""Pre-warms scripts/render_divergent_retrieval_grids.py's --wrist-vision-cache-dir
for one task: loads that task's raw scripts/build_pool_canonical_3cam.py pool (has
fish0/fish1 images per chunk already; its key_embeddings field is zed-only, carried
over unchanged by that script) and computes+caches DINOv2 embeddings for every chunk's
fish0/fish1 first frame. Standalone script (not the render script's own lazy-compute
path) so a SLURM job can shard this expensive, embarrassingly-parallel-by-task step
(some task pools are 5GB+, thousands of chunks) across many processes instead of
paying for it serially inside the single-process render step.

Idempotent / resumable: exits immediately if --cache-dir/<task_slug>.npz already has
the right number of rows for this task's current pool (same check
render_divergent_retrieval_grids._wrist_pool_embeddings uses), so a re-run after a
partial job failure only redoes the tasks that didn't finish.

Usage (one task at a time -- see shells/slurm/victr_wrist_vision_grids_job.sh for the
parallel-across-tasks driver):

    third_party/openpi/.venv/bin/python3 scripts/precompute_wrist_vision_embeddings.py \
        --pool-dir third_party/openpi/assets/victr_icl_pool_canonical_3cam_224 \
        --cache-dir outputs/victr/wrist_vision_embeddings \
        --task "clean the plate"
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "third_party/openpi/src"))

from openpi.policies.yor_retrieval import embed_frames, load_pool, task_slug  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-dir", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--task", required=True, type=str)
    args = parser.parse_args()

    slug = task_slug(args.task)
    cache_path = args.cache_dir / f"{slug}.npz"

    start = time.time()
    pool = load_pool(args.pool_dir / f"{slug}.pkl")
    print(f"task {args.task!r}: loaded pool ({len(pool)} chunks) in {time.time() - start:.1f}s", flush=True)

    if cache_path.exists():
        cached = np.load(cache_path)
        if len(cached["fish0"]) == len(pool):
            print(f"task {args.task!r}: cache already up to date at {cache_path}, skipping", flush=True)
            return
        print(f"task {args.task!r}: stale cache ({len(cached['fish0'])} rows vs {len(pool)} chunks), recomputing", flush=True)

    start = time.time()
    fish0_frames = np.stack([c.images["observation.images.fish0"][0] for c in pool.chunks], axis=0)
    fish1_frames = np.stack([c.images["observation.images.fish1"][0] for c in pool.chunks], axis=0)
    fish0_embeddings = embed_frames(fish0_frames)
    fish1_embeddings = embed_frames(fish1_frames)
    print(f"task {args.task!r}: embedded {len(pool)} chunks x 2 cameras in {time.time() - start:.1f}s", flush=True)

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, fish0=fish0_embeddings, fish1=fish1_embeddings)
    print(f"task {args.task!r}: wrote {cache_path}", flush=True)


if __name__ == "__main__":
    main()
