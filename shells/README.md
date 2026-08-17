# shells/

Thin `.sh` wrappers around `scripts/*.py`, one per entrypoint. They exist so
there's a single, uniform way to invoke this project's code whether you're
running it locally or from a SLURM job — the SLURM templates in `slurm/`
call these wrappers rather than `uv run python3 scripts/...` directly, so the
env setup (`_common.sh`) only lives in one place.

| Wrapper | Wraps | Args |
|---|---|---|
| `smoke_chunk_retrieval.sh` | `scripts/smoke_test_chunk_retrieval.py` | none |
| `smoke_robometer.sh` | `scripts/smoke_test_robometer.py` | none |
| `smoke_fused_retrieval.sh` | `scripts/smoke_test_fused_retrieval.py` | none |
| `smoke_victr_policy.sh` | `scripts/smoke_test_victr_policy.py` | none |
| `eval_libero.sh` | `scripts/eval_libero_victr.py` | passthrough — see `--help` |

`_common.sh` is sourced by every wrapper above (not run directly): it
resolves the repo root from the wrapper's own path (not the caller's cwd) and
cds there, so `uv run` always picks up this project's env regardless of where
the wrapper was invoked from.

Run any of them exactly like the script they wrap, e.g.:

```bash
shells/eval_libero.sh --suite libero_object --task-ids 0,1,2 --n-episodes 10 \
    --retrieval-metrics vision,value,vision+value,none \
    --output outputs/eval_libero_object.json
```

## slurm/

SLURM job templates for the eventual HPC move — see `notes/hpc.md` for the
full migration writeup.

Cluster/account-specific settings (partition, account, scratch dir) are kept
out of the job scripts and live in `cluster.conf` instead, since `#SBATCH`
directives are parsed straight from the job script before any shell code
runs — they can't be sourced from a file. Setup, once per cluster/allocation:

```bash
cp shells/slurm/cluster.conf.example shells/slurm/cluster.conf
# edit shells/slurm/cluster.conf — fill in SLURM_PARTITION, SLURM_ACCOUNT, etc.
```

`cluster.conf` is gitignored. Then submit jobs through `submit.sh`, which
reads `cluster.conf` and passes `--partition`/`--account`/`--gres` to
`sbatch` on the command line (these override any conflicting `#SBATCH` line
in the job script itself):

- `submit.sh` — `shells/slurm/submit.sh <job-script> [job-args...]`; any
  leading `--` flags (e.g. `--array=0-9`) are forwarded to `sbatch`.
- `run.slurm.sh` — generic launcher: `shells/slurm/submit.sh shells/slurm/run.slurm.sh <wrapper> [args...]`
- `eval_libero_array.slurm.sh` — array template for the scaled eval sweep
  (one LIBERO task id per array index): `shells/slurm/submit.sh --array=0-9 shells/slurm/eval_libero_array.slurm.sh libero_object`
