import sys
sys.path.insert(0, "src")
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

import dataclasses

cfg = _config.get_config("yor_icl_fast_victr_vision_interp_expanded")
data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
print("norm_stats is None:", data_config.norm_stats is None)
data_config = dataclasses.replace(data_config, episodes=list(data_config.episodes)[:2])

dataset = _data_loader.create_torch_dataset(data_config, cfg.model.action_horizon, cfg.model)
print("dataset len:", len(dataset))

raw = dataset[0]
print("\n=== raw keys ===")
print(sorted(raw.keys()))

data = dict(raw)
pipeline = [
    *data_config.repack_transforms.inputs,
    *data_config.data_transforms.inputs,
]
for t in pipeline:
    data = t(data)
    has = "nearest_action_tokens" in data
    print(f"after {type(t).__name__}: has nearest_action_tokens={has}, value type={type(data.get('nearest_action_tokens'))}")

from openpi import transforms as _transforms
norm = _transforms.Normalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm)
data = norm(data)
print(f"after Normalize: has nearest_action_tokens={'nearest_action_tokens' in data}")

for t in data_config.model_transforms.inputs:
    data = t(data)
    has = "nearest_action_tokens" in data
    print(f"after {type(t).__name__}: has nearest_action_tokens={has}")

print("\nFINAL nearest_action_tokens:", data.get("nearest_action_tokens"))
