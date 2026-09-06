"""Precomputes each task's retrieval-pool ChunkDictionary (DINOv2 key embeddings +
oracle key values) once, so all 3 training arms (vision/value/vision+value) and the
offline held-out eval reuse it instead of re-embedding the pool per run.

Splits (train/pool/test, 85/10/5 per task -- see viktr.data.icl_dataset.build_splits)
are generated once here and written alongside the pool, since training and eval must
both read the exact same split.

Usage:
    uv run python scripts/build_retrieval_pool.py \
        --out-dir outputs/victr/icl_pool --seed 42
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

from viktr.data.icl_dataset import (
    CONTEXT_CHUNK_SIZE,
    DEFAULT_ROOT,
    PRIMARY_CAMERA,
    TASKS,
    all_tasks,
    build_splits,
    expanded_tasks,
    load_episode_arrays,
    save_splits,
    task_slug,
)
from viktr.retrieval.chunk_dictionary import build_chunk_dictionary, save_pool
from viktr.retrieval.embeddings import load_dinov2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--primary-camera", default=PRIMARY_CAMERA)
    parser.add_argument("--chunk-size", type=int, default=CONTEXT_CHUNK_SIZE)
    parser.add_argument("--train-frac", type=float, default=0.85)
    parser.add_argument("--pool-frac", type=float, default=0.10)
    parser.add_argument("--test-frac", type=float, default=0.05, help="0.0 for a plain train/pool split, no held-out eval set")
    parser.add_argument(
        "--tasks",
        choices=["default", "all", "expanded"],
        default="default",
        help="'default': the 4-task pnp subset (TASKS). 'all': every task in the full "
        "icl-dataset (viktr.data.icl_dataset.all_tasks()). 'expanded': all_tasks() minus "
        "gears/notebook/grocery-bag/octagon (viktr.data.icl_dataset.expanded_tasks()).",
    )
    parser.add_argument(
        "--only-task",
        default=None,
        help="Build the pool for just this one task (must be a member of --tasks' set). "
        "splits.json/all_episodes.json are still written for the full --tasks set (needed "
        "for consistency with other tasks' already-built pools), but only this task's pkl "
        "is (re)built. Useful for rebuilding a single unusually-large task's pool in its "
        "own process/job, isolated from every other task's memory footprint.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.tasks == "default":
        tasks = TASKS
    elif args.tasks == "all":
        tasks = all_tasks(root=args.root)
    else:
        tasks = expanded_tasks(root=args.root)
    if args.only_task is not None and args.only_task not in tasks:
        raise ValueError(f"--only-task {args.only_task!r} is not in the --tasks={args.tasks!r} set")
    splits = build_splits(
        tasks, seed=args.seed, root=args.root, train_frac=args.train_frac, pool_frac=args.pool_frac, test_frac=args.test_frac
    )
    splits_path = args.out_dir / "splits.json"
    save_splits(splits, splits_path)
    print(f"wrote splits -> {splits_path}")
    for task, s in splits.items():
        print(f"  {task!r}: train={len(s['train'])} pool={len(s['pool'])} test={len(s['test'])}")

    # All successful episodes per task (train+pool+test combined), matching how
    # assets/yor_icl_pi05_easy_pnp_v2_episodes.json already generalizes the 4-task
    # subset's query set -- consumed by scripts/precompute_retrieval_context.py to
    # enumerate every query frame without needing viktr importable in openpi's venv
    # (same cross-venv-via-static-JSON convention training/config.py already uses).
    all_episodes = {task: sorted(s["train"] + s["pool"] + s["test"]) for task, s in splits.items()}
    all_episodes_path = args.out_dir / "all_episodes.json"
    all_episodes_path.write_text(json.dumps(all_episodes, indent=2))
    print(f"wrote all_episodes -> {all_episodes_path}")

    # Flat, cross-task train-only episode list (no pool/test leakage into training
    # queries) -- for an openpi TrainConfig's episodes_path, mirroring the flat-list
    # format assets/yor_icl_expanded_episodes.json already uses (which is actually
    # train+pool+test combined, not train-only -- see notes/training_runs.md for why
    # that matters for VICTR arms specifically: a query drawn from a pool episode can
    # retrieve its own episode's chunks back out of the pool).
    train_episodes = sorted({ep for s in splits.values() for ep in s["train"]})
    train_episodes_path = args.out_dir / "train_episodes.json"
    train_episodes_path.write_text(json.dumps(train_episodes, indent=2))
    print(f"wrote train_episodes ({len(train_episodes)} episodes, no pool/test overlap) -> {train_episodes_path}")

    dinov2 = load_dinov2()

    build_tasks = tasks if args.only_task is None else [args.only_task]
    for task in build_tasks:
        out_path = args.out_dir / f"{task_slug(task)}.pkl"
        if out_path.exists():
            print(f"task {task!r}: {out_path} already exists, skipping")
            continue
        pool_episode_indices = splits[task]["pool"]
        episodes = load_episode_arrays(
            pool_episode_indices, camera_keys=[args.primary_camera], root=args.root
        )
        pool = build_chunk_dictionary(
            episodes,
            task=task,
            chunk_size=args.chunk_size,
            primary_camera=args.primary_camera,
            embed_model=dinov2,
        )
        save_pool(pool, out_path)
        print(
            f"task {task!r}: {len(pool_episode_indices)} pool episodes -> "
            f"{len(pool)} chunks, key_embeddings {pool.key_embeddings.shape}, "
            f"key_values {'present' if pool.key_values is not None else 'MISSING'} "
            f"-> {out_path}"
        )
        # A few tasks (e.g. the long-horizon sort task) have episodes 10-50x longer
        # than the typical pick-place task, so load_episode_arrays' eager full-episode
        # load can peak near the per-task job's memory ceiling on its own -- drop the
        # raw frame buffers and pool object explicitly rather than waiting on
        # refcounting at the next loop iteration, so peak RSS reflects one task at a
        # time instead of a growing high-water mark across the whole run.
        del episodes, pool
        gc.collect()


if __name__ == "__main__":
    main()
