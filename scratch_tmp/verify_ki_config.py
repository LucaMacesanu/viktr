import sys

import openpi.training.config as _config
import openpi.training.data_loader as _data_loader

name = "yor_icl_ki_subtask"
print(f"=== {name} ===")
config = _config.get_config(name)
print("model:", config.model)
dc = config.data.create(config.assets_dirs, config.model)
print("repo_id:", dc.repo_id, "root:", dc.root, "episodes:", len(dc.episodes))
print("use_quantile_norm:", dc.use_quantile_norm, "action_sequence_keys:", dc.action_sequence_keys)
print("norm_stats present:", dc.norm_stats is not None)

dataset = _data_loader.create_torch_dataset(dc, config.model.action_horizon, config.model)
dataset = _data_loader.transform_dataset(dataset, dc)
for idx in [0, 100, len(dataset) - 1]:
    item = dataset[idx]
    print(f"--- item {idx} ---")
    for k, v in item.items():
        if hasattr(v, "shape"):
            print(f"  {k}: shape={v.shape} dtype={v.dtype}")
        else:
            print(f"  {k}: {type(v)} {v if not hasattr(v, '__len__') or len(v) < 5 else '...'}")
    # Spot-check the discrete targets aren't degenerate (all-zero/all-pad).
    import numpy as np

    fat = np.asarray(item["fast_action_tokens"])
    fam = np.asarray(item["fast_action_tokens_mask"])
    st = np.asarray(item["subtask_tokens"])
    stm = np.asarray(item["subtask_tokens_mask"])
    print(f"  fast_action_tokens real-token-count={fam.sum()} first10={fat[:10].tolist()}")
    print(f"  subtask_tokens real-token-count={stm.sum()} tokens={st[:stm.sum()].tolist()}")
    sys.stdout.flush()

print("OK")
