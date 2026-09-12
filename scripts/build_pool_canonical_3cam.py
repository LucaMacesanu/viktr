"""Extends the canonical VICTR retrieval pool (`victr_icl_pool_canonical_224`,
zed-only, 10-frame chunks, correct canonical proprio/actions per
notes/ICRA_plan.md Sec 1a addendum) with the two things the redesigned
context-encoding scheme (notes/context_encoding_redesign.md) needs that it
doesn't have yet:

1. **All 3 camera views per chunk** (`observation.images.{zed,fish0,fish1}`),
   not just the head camera -- `load_episode_arrays`/`build_chunk_dictionary`
   already support arbitrary `camera_keys` (see their docstrings), the
   original pool build just never requested more than the primary camera
   (memory reasons that no longer apply once this is a live design question --
   see the artifact-visualization + probe discussion this session that
   confirmed all-3-cameras/chunk fits k=3 at batch_size=128, 32/device).
2. **The redesign's actual per-chunk action encoding**: a NATIVE 30-step
   canonical action window starting at the chunk's own start_frame (not the
   pool's 10-step chunk_size window -- see context_encoding_redesign.md Sec 3:
   "matching action_horizon, not the pool's 10-step chunking"), frequency-
   truncated (zero DCT bins >= 20 of 30, chosen cutoff per Sec 7), and
   FAST-tokenized through the exact fitted tokenizer
   (nyu-finger-robot/outputs/fast_tokenizer/yor-icl-canonical) training will
   use -- baked in now so a future build_context_block rewrite is an O(1)
   array read, not a live DCT+BPE call per sample.

Copies task/episode_index/start_frame/end_frame/proprio/actions/value/
key_embeddings/key_values through UNCHANGED from the source canonical pool --
none of those depend on camera count or action-window length. Writes to a NEW
directory (victr_icl_pool_canonical_3cam_224/), leaving the existing
canonical pool untouched, same convention patch_pool_canonical.py established.

Chunks whose episode doesn't have 30 frames available from start_frame
onward are dropped (not padded) -- same choice measure_30step_broad.py made
(~4.9% of native windows dataset-wide in that exhaustive scan); a fully valid,
fixed-length window is worth more than forcing every chunk to have one.

Must run in openpi's venv (torch/av for video decode, transformers for the
FAST tokenizer) with the ffmpeg7 shim set up (see run_pool_canonical_3cam.sh).

Usage:
    third_party/openpi/.venv/bin/python3 scripts/build_pool_canonical_3cam.py
"""

from __future__ import annotations

import gc
import pathlib
import pickle
import sys
import time

import numpy as np
import torch
from scipy.fft import dct, idct

sys.path.insert(0, "/scratch/lim2045/icl_ws/viktr/third_party/openpi/src")
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

from openpi.models import tokenizer as _tokenizer  # noqa: E402
from openpi.policies import yor_rotation  # noqa: E402
from openpi.shared.image_tools import resize_with_pad_torch  # noqa: E402

ROOT = "/scratch/lim2045/icl_ws/icl-dataset-fixed-obs"
SRC_POOL_DIR = pathlib.Path("/scratch/lim2045/icl_ws/viktr/third_party/openpi/assets/victr_icl_pool_canonical_224")
DST_POOL_DIR = pathlib.Path("/scratch/lim2045/icl_ws/viktr/third_party/openpi/assets/victr_icl_pool_canonical_3cam_224")
FAST_TOKENIZER_PATH = "/scratch/lim2045/icl_ws/viktr/third_party/nyu-finger-robot/outputs/fast_tokenizer/yor-icl-canonical"

CAMERAS = ["observation.images.zed", "observation.images.fish0", "observation.images.fish1"]
ACTION_HORIZON = 30
KEEP_BINS = 20  # notes/context_encoding_redesign.md Sec 7's chosen k=3 cutoff
ACTION_TOKEN_MAX_LEN = 160  # dataset-wide truncated max measured at 120 (Sec 7); generous margin, no silent-truncation risk

_LEFT_QUAT, _LEFT_POS = slice(0, 4), slice(4, 7)
_RIGHT_QUAT, _RIGHT_POS = slice(7, 11), slice(11, 14)
_LEFT_GRIP, _RIGHT_GRIP = slice(14, 15), slice(15, 16)


def canonical_action_window(action: np.ndarray) -> np.ndarray:
    action = action.astype(np.float64)
    left_delta_pos = action[:, _LEFT_POS] - action[0:1, _LEFT_POS]
    right_delta_pos = action[:, _RIGHT_POS] - action[0:1, _RIGHT_POS]
    left_rot6d = yor_rotation.quat_wxyz_to_rot6d(action[:, _LEFT_QUAT])
    right_rot6d = yor_rotation.quat_wxyz_to_rot6d(action[:, _RIGHT_QUAT])
    return np.concatenate(
        [left_delta_pos, left_rot6d, action[:, _LEFT_GRIP], right_delta_pos, right_rot6d, action[:, _RIGHT_GRIP]],
        axis=-1,
    ).astype(np.float32)


def truncate_and_tokenize(canon30: np.ndarray, tok: _tokenizer.FASTTokenizer) -> dict:
    coeff = dct(canon30[None], axis=1, norm="ortho")
    coeff[:, KEEP_BINS:, :] = 0
    recon = idct(coeff, axis=1, norm="ortho")[0].astype(np.float32)
    ids, mask = tok.tokenize_action_only(recon, max_len=ACTION_TOKEN_MAX_LEN)
    return {
        "action_native30_raw": canon30,
        "action_native30_truncated": recon,
        "fast_tokens_truncated": ids.astype(np.int32),
        "fast_tokens_truncated_mask": mask,
    }


def main() -> None:
    DST_POOL_DIR.mkdir(parents=True, exist_ok=True)
    src_paths = sorted(SRC_POOL_DIR.glob("*.pkl"))
    print(f"[build_pool_canonical_3cam] {len(src_paths)} task pools under {SRC_POOL_DIR}")

    tok = _tokenizer.FASTTokenizer(max_len=1024, fast_tokenizer_path=FAST_TOKENIZER_PATH)

    for pi, src_path in enumerate(src_paths):
        dst_path = DST_POOL_DIR / src_path.name
        if dst_path.exists():
            print(f"[{pi + 1}/{len(src_paths)}] {src_path.name}: already done, skipping")
            continue

        t0 = time.time()
        with open(src_path, "rb") as f:
            pool = pickle.load(f)
        chunks = pool["chunks"]
        episode_indices = sorted({int(c["episode_index"]) for c in chunks})
        print(f"[{pi + 1}/{len(src_paths)}] {src_path.name}: {len(chunks)} chunks, {len(episode_indices)} episodes", flush=True)

        # Decode each episode's full fish0/fish1 + action stream once (sequential access,
        # matches load_episode_arrays' eager per-episode convention), slice per chunk after.
        episode_data: dict[int, dict] = {}
        for ep in episode_indices:
            dataset = LeRobotDataset("icl-dataset", root=ROOT, episodes=[ep], tolerance_s=2e-4)
            fish0, fish1, action = [], [], []
            for i in range(len(dataset)):
                item = dataset[i]
                for cam, out in ((CAMERAS[1], fish0), (CAMERAS[2], fish1)):
                    frame = item[cam]
                    arr = frame.permute(1, 2, 0).numpy() if hasattr(frame, "permute") else np.asarray(frame)
                    if arr.dtype != np.uint8:
                        arr = (arr * 255).clip(0, 255).astype(np.uint8)
                    out.append(arr)
                action.append(np.asarray(item["action"], dtype=np.float32))
            episode_data[ep] = {
                "fish0": np.stack(fish0, axis=0),
                "fish1": np.stack(fish1, axis=0),
                "action": np.stack(action, axis=0),
            }
            del dataset
            if len(episode_data) % 20 == 0:
                print(f"    decoded {len(episode_data)}/{len(episode_indices)} episodes...", flush=True)

        keep_mask = np.zeros(len(chunks), dtype=bool)
        for i, c in enumerate(chunks):
            ep = int(c["episode_index"])
            s, e = int(c["start_frame"]), int(c["end_frame"])
            ed = episode_data[ep]
            n_frames = len(ed["action"])

            if s + ACTION_HORIZON > n_frames:
                continue  # keep_mask[i] stays False -- also drops this chunk's key_embeddings/key_values row

            fish0_224 = resize_with_pad_torch(torch.from_numpy(ed["fish0"][s:e]), 224, 224).numpy()
            fish1_224 = resize_with_pad_torch(torch.from_numpy(ed["fish1"][s:e]), 224, 224).numpy()
            c["images"] = {
                CAMERAS[0]: c["images"][CAMERAS[0]],  # zed, already 224x224 from the source pool
                CAMERAS[1]: fish0_224,
                CAMERAS[2]: fish1_224,
            }
            canon30 = canonical_action_window(ed["action"][s : s + ACTION_HORIZON])
            c.update(truncate_and_tokenize(canon30, tok))
            keep_mask[i] = True

        pool["chunks"] = [c for c, keep in zip(chunks, keep_mask, strict=True) if keep]
        pool["key_embeddings"] = pool["key_embeddings"][keep_mask]
        if pool.get("key_values") is not None:
            pool["key_values"] = pool["key_values"][keep_mask]

        with open(dst_path, "wb") as f:
            pickle.dump(pool, f, protocol=pickle.HIGHEST_PROTOCOL)

        dt = time.time() - t0
        n_kept = int(keep_mask.sum())
        print(
            f"[{pi + 1}/{len(src_paths)}] {src_path.name}: {n_kept}/{len(chunks)} chunks kept "
            f"({len(chunks) - n_kept} dropped, native30 didn't fit) -> {dst_path} ({dt:.1f}s)",
            flush=True,
        )
        del pool, chunks, episode_data
        gc.collect()


if __name__ == "__main__":
    main()
