"""One-off converter: re-pickles outputs/victr/icl_pool/*.pkl (viktr.retrieval.
chunk_dictionary.Chunk/ChunkDictionary instances) as plain dicts/lists of numpy
arrays, so openpi's vendored pool loader (third_party/openpi/src/openpi/policies/
yor_retrieval.py) can unpickle them without importing `viktr` -- openpi is a
separate uv workspace/venv, and pickle resolves classes by their original module
path, so a pool pickled with viktr's own Chunk/ChunkDictionary classes fails to
load from any environment that doesn't have `viktr` importable.

Run once, from viktr's own venv (where the original pool loads normally):
    uv run python scripts/convert_pool_for_openpi.py \
        --pool-dir outputs/victr/icl_pool \
        --out-dir third_party/openpi/assets/victr_icl_pool
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from viktr.data.icl_dataset import DEFAULT_ROOT, TASKS, all_tasks, expanded_tasks, task_slug
from viktr.retrieval.chunk_dictionary import load_pool


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--tasks", choices=["default", "all", "expanded"], default="default")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.tasks == "default":
        tasks = TASKS
    elif args.tasks == "all":
        tasks = all_tasks(root=args.root)
    else:
        tasks = expanded_tasks(root=args.root)
    for task in tasks:
        pool = load_pool(args.pool_dir / f"{task_slug(task)}.pkl")
        plain = {
            "chunks": [
                {
                    "task": c.task,
                    "episode_index": c.episode_index,
                    "start_frame": c.start_frame,
                    "end_frame": c.end_frame,
                    "images": dict(c.images),
                    "proprio": c.proprio,
                    "actions": c.actions,
                    "value": c.value,
                }
                for c in pool.chunks
            ],
            "key_embeddings": pool.key_embeddings,
            "key_values": pool.key_values,
        }
        out_path = args.out_dir / f"{task_slug(task)}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(plain, f)
        print(f"task {task!r}: {len(pool)} chunks -> {out_path}")


if __name__ == "__main__":
    main()
