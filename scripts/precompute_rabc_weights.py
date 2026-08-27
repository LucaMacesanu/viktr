"""Offline precompute of Reward-Aligned Behavior Cloning (RA-BC, notes/reward_aligned_bc.md)
per-frame loss weights for one training episode set.

RA-BC (SARM paper, third_party/opensarm/2509.25358, Sec 3.2, Eq. 6-9) weights each
training item by the progress delta b_t = phi(o_{t+delta}) - phi(o_t), mapped through
running dataset statistics (mu, sigma) and a hard threshold kappa. The paper computes
mu/sigma/kappa online (Welford) during training; here they're computed once, offline,
over the same fixed episode set training will use -- see reward_aligned_bc.md Sec 5(b)
for why that's equivalent and simpler in this codebase (phi is already fully
precomputed/precomputable, so there's nothing an online estimator buys us).

phi defaults to RoboDopamine's real per-frame progress estimates
(viktr.data.icl_dataset.robodopamine_values); oracle t/(T-1) (Eq. 7 of notes/vktr.pdf)
is available as a fallback for episodes RoboDopamine doesn't cover, or as an explicit
--phi-source for ablations, matching the fallback convention icl_dataset.py itself
documents.

Run once per episode set, from viktr's own venv (where viktr.data.icl_dataset is
importable):
    uv run python scripts/precompute_rabc_weights.py \
        --episodes-path third_party/openpi/assets/yor_icl_pi05_easy_pnp_v2_episodes.json \
        --out-dir third_party/openpi/assets/rabc_weights/yor_icl_pi05_rabc

Output: one <episode_index>.npy (T,) float32 weight curve per episode in --out-dir,
plus a stats.json recording {mu, sigma, kappa, epsilon, delta, phi_source} for
provenance. openpi.policies.yor_rabc.RabcWeightInputs reads these at training time via
an O(1) per-episode lookup.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from viktr.data.icl_dataset import DEFAULT_ROOT, episode_length, oracle_values, robodopamine_values


def _phi_curve(episode_index: int, phi_source: str, root: Path) -> np.ndarray:
    if phi_source == "oracle":
        return oracle_values(episode_length(episode_index, root=root))
    try:
        return robodopamine_values(episode_index, root=root)
    except KeyError:
        print(f"episode {episode_index}: no RoboDopamine coverage, falling back to oracle")
        return oracle_values(episode_length(episode_index, root=root))


def _delta_curve(phi: np.ndarray, delta: int) -> np.ndarray:
    """b_t = phi(clamp(t+delta, T-1)) - phi(t), Eq. 6 -- clamped at the episode's end so
    near-end frames measure progress over a shorter-than-delta window instead of being
    undefined."""
    length = len(phi)
    shifted_idx = np.minimum(np.arange(length) + delta, length - 1)
    return phi[shifted_idx] - phi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-path", required=True, type=Path, help="JSON list of episode indices (same file passed to LeRobotYorDataConfig.episodes_path)")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--phi-source", choices=["robodopamine", "oracle"], default="robodopamine")
    parser.add_argument("--delta", type=int, default=30, help="progress-delta stride; default matches ACTION_CHUNK_SIZE / Pi0Config.action_horizon")
    parser.add_argument("--kappa", type=float, default=None, help="override the hard-weight threshold (Eq. 9); default derives it from --kappa-percentile")
    parser.add_argument("--kappa-percentile", type=float, default=0.95, help="percentile of pooled deltas used as kappa when --kappa is not set (paper's kappa=0.01 corresponds to ~top 5%)")
    parser.add_argument("--epsilon", type=float, default=1e-6)
    args = parser.parse_args()

    episode_indices = json.loads(args.episodes_path.read_text())
    print(f"{len(episode_indices)} episodes, phi_source={args.phi_source}, delta={args.delta}")

    phi_curves = {ep: _phi_curve(ep, args.phi_source, args.root) for ep in episode_indices}
    delta_curves = {ep: _delta_curve(phi, args.delta) for ep, phi in phi_curves.items()}

    pooled = np.concatenate(list(delta_curves.values()))
    mu = max(float(np.mean(pooled)), 0.0)  # clamp mu >= 0, Sec 3.2
    sigma = float(np.std(pooled))
    kappa = args.kappa if args.kappa is not None else float(np.quantile(pooled, args.kappa_percentile))
    print(f"pooled {len(pooled)} deltas: mu={mu:.4f} sigma={sigma:.4f} kappa={kappa:.4f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    lo, span = mu - 2 * sigma, 4 * sigma + args.epsilon
    for ep, b in delta_curves.items():
        w_soft = np.clip((b - lo) / span, 0.0, 1.0)  # Eq. 8
        weight = np.where(b > kappa, 1.0, np.where(b >= 0, w_soft, 0.0)).astype(np.float32)  # Eq. 9
        np.save(args.out_dir / f"{ep}.npy", weight)

    stats = {
        "mu": mu,
        "sigma": sigma,
        "kappa": kappa,
        "epsilon": args.epsilon,
        "delta": args.delta,
        "phi_source": args.phi_source,
        "num_episodes": len(episode_indices),
        "episodes_path": str(args.episodes_path),
    }
    (args.out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"wrote {len(episode_indices)} weight curves + stats.json to {args.out_dir}")


if __name__ == "__main__":
    main()
