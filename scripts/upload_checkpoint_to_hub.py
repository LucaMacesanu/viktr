#!/usr/bin/env python3
"""Upload an openpi checkpoint's params+assets (no train_state) to the HF Hub.

Matches the existing checkpoint-repo convention (see notes/ICRA_plan.md, and the
manually-uploaded lair-nyu/yor_icl_pi05_canonical_sanity15k-step5000 repo): the
top-level `_CHECKPOINT_METADATA` marker plus `params/` and `assets/` are uploaded;
`train_state/` (optimizer state, ~1.5x the size of params) is dropped since it's
only needed to resume training, not to run the policy.

Also uploads a README.md documenting which openpi TrainConfig (by name, from
openpi/src/openpi/training/config.py) is needed to load these weights --
create_trained_policy looks up the model architecture and data transforms by
config name, NOT from anything in this checkpoint, so the repo is otherwise
not self-contained. Pass --config-name so that instruction is correct; add
--notes for any config-specific context worth documenting (architecture,
retrieval metric, dataset subset, etc.) -- see notes/ICRA_plan.md or the
config's own comment block in config.py for material to draw from.

Usage:
    python scripts/upload_checkpoint_to_hub.py <checkpoint_step_dir> <repo_id> \\
        --config-name <name> [--notes "..."] [--private]

Example:
    python scripts/upload_checkpoint_to_hub.py \\
        third_party/openpi/checkpoints/yor_icl_pi05_canonical_extended/yor_icl_pi05_canonical_extended/20000 \\
        lair-nyu/yor_icl_pi05_canonical_extended-step20000 \\
        --config-name yor_icl_pi05_canonical_extended
"""

import argparse
import re
from pathlib import Path

from huggingface_hub import HfApi

README_TEMPLATE = """\
# {config_name} -- step {step} checkpoint

{notes}

Only `params/` (model weights), `assets/` (norm_stats.json) and
`_CHECKPOINT_METADATA` are included -- `train_state/` (optimizer state, ~1.5x
the size of params, only needed to resume training) is dropped, matching the
existing checkpoint-repo convention in this project.

**Loading this checkpoint requires the training config it was built from**
(`{config_name}` in `openpi/src/openpi/training/config.py`) -- this repo only
carries weights + norm stats, not the model architecture or data transforms.
With that repo checked out:

```python
from openpi.policies import policy_config
from openpi.training import config as _config

policy = policy_config.create_trained_policy(
    _config.get_config("{config_name}"),
    "<local snapshot_download of this repo>",
)
```
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint_dir", type=Path, help="Path to a step dir, e.g. checkpoints/<exp>/<run>/<step>")
    parser.add_argument("repo_id", help="Target HF repo id, e.g. lair-nyu/<name>-step<N>")
    parser.add_argument(
        "--config-name",
        required=True,
        help="openpi TrainConfig name (config.py) needed to load these weights -- documented in the README",
    )
    parser.add_argument("--notes", default="", help="Free-text context to include in the README (architecture, "
                         "retrieval metric, dataset subset, etc.)")
    parser.add_argument("--private", action="store_true", help="Create/push as a private repo (default: public)")
    args = parser.parse_args()

    ckpt_dir = args.checkpoint_dir.resolve()
    params_dir = ckpt_dir / "params"
    assets_dir = ckpt_dir / "assets"
    metadata_file = ckpt_dir / "_CHECKPOINT_METADATA"
    if not params_dir.is_dir():
        raise SystemExit(f"error: {params_dir} does not exist (not a valid checkpoint step dir?)")

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)

    print(f"uploading {params_dir} -> {args.repo_id}:params/ ({'private' if args.private else 'public'})")
    api.upload_folder(repo_id=args.repo_id, repo_type="model", folder_path=str(params_dir), path_in_repo="params")

    if assets_dir.is_dir():
        print(f"uploading {assets_dir} -> {args.repo_id}:assets/")
        api.upload_folder(repo_id=args.repo_id, repo_type="model", folder_path=str(assets_dir), path_in_repo="assets")
    else:
        print(f"note: no assets/ dir at {assets_dir}, skipping")

    if metadata_file.is_file():
        print(f"uploading {metadata_file} -> {args.repo_id}:_CHECKPOINT_METADATA")
        api.upload_file(
            repo_id=args.repo_id,
            repo_type="model",
            path_or_fileobj=str(metadata_file),
            path_in_repo="_CHECKPOINT_METADATA",
        )
    else:
        print(f"note: no _CHECKPOINT_METADATA at {metadata_file}, skipping")

    step_match = re.search(r"(\d+)$", ckpt_dir.name)
    step = step_match.group(1) if step_match else ckpt_dir.name
    readme = README_TEMPLATE.format(
        config_name=args.config_name,
        step=step,
        notes=args.notes or "(no additional notes provided)",
    )
    print(f"uploading README.md -> {args.repo_id}")
    api.upload_file(
        repo_id=args.repo_id, repo_type="model", path_or_fileobj=readme.encode("utf-8"), path_in_repo="README.md"
    )

    print(f"done: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
