"""Extracts just the DINOv2 key_embeddings (+ key_values + chunk metadata) out of each
task's ChunkDictionary pool pickle, without keeping the (much larger) per-chunk images/
proprio/actions arrays around -- those pickles are 1-36GB each (176GB total) because
they carry full chunk image stacks, whereas the embeddings alone are tiny. Loading a
whole task pickle still needs its full on-disk size in RAM momentarily (unpickling
builds the Chunk list before we can discard it), so this runs as a SLURM job with
enough memory headroom for the largest task file (sort_the_items_into_their_containers,
36GB), not on the login node (a plain interactive load of even the 11GB clean_the_plate
pool got OOM-killed there).

Writes one .npz per task (key_embeddings, key_values, and metadata) to --out-dir, then
zips the whole directory.
"""

from __future__ import annotations

import argparse
import gc
import json
import pickle
import zipfile
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--zip-path", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pkl_paths = sorted(args.pool_dir.glob("*.pkl"))
    print(f"found {len(pkl_paths)} task pools in {args.pool_dir}", flush=True)

    for i, pkl_path in enumerate(pkl_paths):
        task_slug = pkl_path.stem
        out_path = args.out_dir / f"{task_slug}.npz"
        if out_path.exists():
            print(f"[{i + 1}/{len(pkl_paths)}] skip (exists): {task_slug}", flush=True)
            continue

        print(f"[{i + 1}/{len(pkl_paths)}] loading {pkl_path} ({pkl_path.stat().st_size / 1e9:.1f} GB)...", flush=True)
        with open(pkl_path, "rb") as f:
            pool = pickle.load(f)

        key_embeddings = pool.key_embeddings
        key_values = pool.key_values if pool.key_values is not None else np.array([])
        episode_index = np.array([c.episode_index for c in pool.chunks], dtype=np.int64)
        start_frame = np.array([c.start_frame for c in pool.chunks], dtype=np.int64)
        end_frame = np.array([c.end_frame for c in pool.chunks], dtype=np.int64)
        task_names = np.array([c.task for c in pool.chunks])

        np.savez_compressed(
            out_path,
            key_embeddings=key_embeddings,
            key_values=key_values,
            episode_index=episode_index,
            start_frame=start_frame,
            end_frame=end_frame,
            task=task_names,
        )
        print(
            f"    -> {out_path.name}: key_embeddings {key_embeddings.shape} {key_embeddings.dtype}, "
            f"{len(pool.chunks)} chunks",
            flush=True,
        )

        del pool, key_embeddings, key_values, episode_index, start_frame, end_frame, task_names
        gc.collect()

    npz_paths = sorted(args.out_dir.glob("*.npz"))
    manifest = {"num_tasks": len(npz_paths), "source_pool_dir": str(args.pool_dir)}
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"zipping {len(npz_paths)} files -> {args.zip_path}", flush=True)
    args.zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip_path, "w", zipfile.ZIP_STORED) as zf:
        for p in npz_paths:
            zf.write(p, arcname=p.name)
        zf.write(args.out_dir / "manifest.json", arcname="manifest.json")

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
