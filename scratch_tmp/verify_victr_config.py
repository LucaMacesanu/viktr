import sys

import openpi.training.config as _config
import openpi.training.data_loader as _data_loader

for name in ["yor_icl_victr_vision", "yor_icl_victr_value"]:
    print(f"=== {name} ===")
    config = _config.get_config(name)
    print("model:", config.model)
    dc = config.data.create(config.assets_dirs, config.model)
    print("repo_id:", dc.repo_id, "root:", dc.root, "episodes:", len(dc.episodes))
    print("use_quantile_norm:", dc.use_quantile_norm, "action_sequence_keys:", dc.action_sequence_keys)
    print("norm_stats present:", dc.norm_stats is not None)

    dataset = _data_loader.create_torch_dataset(dc, config.model.action_horizon, config.model)
    dataset = _data_loader.transform_dataset(dataset, dc, skip_norm_stats=True)
    item = dataset[0]
    for k, v in item.items():
        if hasattr(v, "shape"):
            print(f"  {k}: shape={v.shape} dtype={v.dtype}")
        else:
            print(f"  {k}: {type(v)} {v if not hasattr(v, '__len__') or len(v) < 5 else '...'}")
    sys.stdout.flush()

print("OK")
