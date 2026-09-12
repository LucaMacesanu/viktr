# VICTR context-block re-encoding (canonical arms)

Source: `openpi/src/openpi/policies/yor_retrieval.py` (`chunk_to_context_text`,
`build_context_block`, `RetrievalContextInputs`), `openpi/src/openpi/models/pi0_victr.py`
(`embed_context_chunks`), `openpi/src/openpi/models/pi0_fast_victr.py`,
`openpi/src/openpi/models/tokenizer.py` (`FASTTokenizer`), and the fitted tokenizer's
own `processing_action_tokenizer.py` (`UniversalActionProcessor`,
`nyu-finger-robot/outputs/fast_tokenizer/yor-icl-canonical/`).

Triggered by debugging job 17431526's OOM (`yor_icl_fast_victr_vision_interp_canonical`,
RICL). What started as a batch-size fix turned into finding that the *other* three
already-running VICTR arms are also badly broken, and a concrete, measured redesign of
how retrieval context gets encoded.

## 1. What's broken today

Every VICTR arm's per-chunk context block is: N image tokens (SigLIP, `So400m/14`,
256 tokens/frame) + one block of digitized "Task/State/Action" text
(`chunk_to_context_text`), tokenized through the plain Gemma text vocab and padded/
truncated to `context_text_max_length`.

`chunk_to_context_text` digitizes the chunk's proprio (18 canonical dims) and up to 8
sampled action frames (20 canonical dims each = 160 numbers) into space-separated
decimal strings: `"Task: {name}, State: {18 numbers}; Action: {160 numbers}"`. Each
digit-run tokenizes into multiple sub-tokens under Gemma's SentencePiece vocab, so this
routinely runs 700+ tokens for a real chunk.

`yor_icl_victr_vision_canonical`/`_value_canonical`/`_vision_value_canonical` (arms
3-5) all share `outputs/victr/retrieval_context_expanded`, precomputed with
`context_text_max_length` left at its **default of 64** (confirmed in
`precompute_retrieval_context.py`'s own docstring). Verified directly against the
precomputed `.npy` files: **100% of 4,211 sampled frames across 8 random canonical
episodes hit exactly 64/64 tokens with zero padding** -- total truncation, not a tail
effect. Decoded one real example:

```
'Task: put the circle on the peg, State: 255 255 0 255 102 118 129 0 255 255 255 105 118 133'
```

14 of 18 state values survive; the entire `Action:` section -- all 160 numbers -- never
reaches the model. So arms 3-5 are training VICTR retrieval-context conditioning on
image + task name + partial state only, never the neighbor's action, and nobody had
verified this before now.

RICL (arm 6, `yor_icl_fast_victr_vision_interp_canonical`) got `context_text_max_length`
bumped to 1024 specifically to fix this (real content measured at 700-713 tokens at the
time), which is *correct* but combined with `batch_size=256` (2x the `_expanded`
precedent's 128) OOM'd on job 17431526. Root-caused: **not primarily the context
length** -- an isolation test (`use_action_interpolation=False`, same tiny context)
still passed at `batch_size=256`/64-per-device, while the actual failure traces to
`interpolate_actions`' own `(batch, window~223, vocab~257k)` intermediate tensors in
`pi0_fast_victr.py`'s `compute_loss` (the same class of OOM a prior incident,
jobs 16561972/16612787, already hit and partially fixed -- see that code's own
comment). **Decision: drop pi0-fast/RICL from this redesign entirely**, focus on the
pi0.5 + VICTR backbone (arms 3-5), which has no discrete-vocab action-interpolation
cost at all.

## 2. Why the images are fine

Checked separately: `196_images.npy`/`196_image_masks.npy` for a real episode show
correct shape `(362, 1, 1, 224, 224, 3)`, masks 100% `True`, real non-degenerate pixel
content, 36 distinct retrieved neighbor images across 362 frames with 114 changepoints
-- exactly the expected behavior for vision-based nearest-neighbor retrieval as the
query moves through its trajectory. The failure is isolated to the text channel.

## 3. The redesign

Per-chunk context block, replacing the single digitized-text block:

| Component | Encoding | Cost |
|---|---|---|
| Image (unchanged) | SigLIP `So400m/14`, 224x224 | 256 tokens |
| Task label | Plain text, `"Task: {name}"` only (no digit dump) | measured max 22 tokens |
| State (18-dim) | **1 continuous token** via the model's *existing* `state_proj` linear layer (`pi0.py:98,159`) -- the same weights already used for the query's own state | 1 token, **zero new params** |
| Action (20-dim x 30 steps) | **FAST-tokenized**, full native 30-step window (matching `action_horizon`, not the pool's 10-step chunking), via the *existing* `_action_only_tokenizer`/`tokenize_action_only` machinery already used for RICL's interpolation target | measured, see below |

Two things fall out of existing model components for free: `state_proj` already embeds
the *query's* own proprioception as one continuous token (not text) -- reusing it for
the neighbor's state costs no new parameters and puts it in a subspace the model
already understands. The FAST action tokenizer is already fit and already used
elsewhere in this file (`action_interpolation_extras`) -- reusing it for the general
context block (not just RICL's interpolation target) is new plumbing, not new
infrastructure.

## 4. Measurements (exhaustive, all 20 canonical task pools, not sampled)

An initial sample (32 anchors across 6 of 20 task pools) suggested the 30-step
FAST-token length maxes around 44. **This significantly understated the tail.**
Exhaustive scans:

| Quantity | 10-step (pool's stored chunk) | 30-step (native, what we use) |
|---|---|---|
| n | 10,336 chunks | 9,856 windows (480 skipped, too short) |
| mean | 24.4 tok | 33.6 tok |
| max | 69 tok | **154 tok** (wrapped: bos+"Action: "+fast+"\|"+eos; 148 tok raw) |
| p99 | 33.0 | 120.0 |

The 30-step distribution is genuinely **bimodal**: >90% of windows compress to
20-60 tokens, but a distinct cluster of ~112 windows (1.1%) lands at 110-155 tokens.
Investigated two natural hypotheses for the cluster -- both wrong: gripper
open/close transitions show no difference between clusters (low-cluster mean 0.26 vs.
high-cluster 0.21, correlation 0.086), and max single-step position jump is actually
*smaller* in the high cluster (0.0061 vs. 0.0042, correlation only 0.235 overall). The
cluster is scattered across 6 different tasks at low per-task rates (0.8%-5.4%), not
concentrated in one "hard task" -- it's isolated complex moments within otherwise
normal episodes, plausibly rotation-channel dynamics (unverified). Root mechanism
(DCT + BPE is content-adaptive, not fixed-rate like naive per-step binning) is
understood; the specific driver of the outliers is not, and doesn't need to be for the
practical decision below.

Real-world corroboration: the tokenizer's own fit-time metadata
(`nyu-finger-robot/outputs/fast_tokenizer/yor-icl-canonical/metadata.json`,
`compression_stats`) independently recorded `max_token_length: 175.0` on a 10% fitting
sample of 30-step chunks -- same order of magnitude, same phenomenon.

## 5. Chunk-count probes (pi0.5 backbone, `yor_icl_victr_vision_canonical`,
   batch_size=256 = 64/device on 4xH200)

Real `train_step()` runs (not compile-only estimates) via a 1-GPU probe at matched
per-device batch (pure data-parallel, no FSDP -- a 1-GPU run at local batch 64
reproduces the exact per-device memory footprint of the real 4-GPU job). 1-GPU
requests queue in seconds under this cluster's scheduler vs. hours for a 4-GPU
request, which is why every probe in this investigation used that trick.

| `context_text_max_length` proxy | num_context_chunks tested | result |
|---|---|---|
| 64 (wrong -- pre-exhaustive-measurement) | 1,2,3,4 OK, 5 FAIL | superseded |
| 224 (honest: measured max 154 + margin) | 1,2 OK, **3 FAIL** | this is what "224" actually buys |

Both calibration points land on the same rule: **total sequence length is what
matters**, not how it's split between chunk count and per-chunk budget --
`k=2/budget=224` (total 2015) and `k=3/budget=64` (total 2015) both succeed at
identical totals. Interpolating the OOM wall between the last confirmed-OK point
(k=4/budget=64, total 2335) and the confirmed-FAIL point (k=3/budget=224, total 2495)
puts it around **total ~2400**, implying a per-chunk budget ceiling of roughly **192
tokens for k=3** and **~80 tokens for k=4** (query-side baseline for pi0.5:
3 cameras 768 + prompt 256 + state 1 + 30-step flow-matching suffix 30 = 1055).

## 6. Truncation: does it help, and what breaks

Read `processing_action_tokenizer.py` directly. Encoding: DCT along time
(`axis=1`, `norm="ortho"`) -> round+scale -> flatten **frequency-major**
(all 20 dims per frequency bin, ascending) into a character string -> BPE. This means
"cut high frequencies" is a clean, well-defined operation (zero a contiguous suffix of
frequency bins before flattening) -- genuinely different from "cap `max_action_tokens`"
(slicing the *compressed* BPE token list, which is what production does today on
overflow).

Tested both, on the real worst-case chunk (episode 1575, start_frame 300, 148 raw
tokens -- confirmed as the actual dataset-wide max, not a stand-in):

- **Naive truncation (cut the BPE token list at 50%, 74/148 kept)**: `decode()` threw
  `cannot reshape array of size 76 into shape (20)` -- BPE tokens don't align with
  coefficient boundaries, so an arbitrary cut essentially never reshapes cleanly. The
  tokenizer's own exception handler returns an **all-zero action**. Confirmed across
  every one of 20 canonical dims. This is exactly what happens today whenever a window
  exceeds `max_action_tokens` -- not "a blurrier version of the neighbor's action," a
  token sequence with no coherent signal behind it at all.
- **Frequency truncation (zero DCT bins >= F, then IDCT, then re-tokenize through the
  *unmodified* fitted tokenizer)**:

  | Keep bins (of 30) | Mean abs error | Re-tokenized length |
  |---|---|---|
  | full (148) | 0.012 | 148 |
  | 20 | 0.014 | 120 |
  | 15 | 0.021 | 99 |
  | 10 | 0.030 | 72 |
  | 7 | 0.034 | 58 |

  Error grows slowly and stays small (canonical action values are O(1) or smaller)
  even at aggressive cutoffs, because DCT/IDCT round-trips losslessly and the
  BPE-fit vocabulary already handles zero-heavy coefficient strings well (that's
  exactly what makes smooth chunks compress to ~20-30 tokens in the first place).
  **No tokenizer refit needed** -- `DCT -> zero bins >= F -> IDCT` just produces a
  smoothed action array; feeding that through the existing fitted tokenizer
  reproduces the same quantized coefficients a refit encode-time change would, so
  this is a pre-processing step on the neighbor's action, not a vocabulary change.
  Only ever applied to the *retrieved neighbor's* action (context conditioning) --
  never to the query's own action target.

## 7. Chosen cutoff and dataset-wide confirmation

Target: 3 neighbors (`num_context_chunks=3`), not 4 -- needs budget <=192, so picked
the mild cutoff (keep 20/30 bins, 0.014 error on the worst case) rather than the more
aggressive ones needed for 4.

Re-ran the exhaustive scan (all 9,856 windows, not just the one worst-case chunk) with
this truncation applied before tokenizing:

| | Original | Truncated (keep 20/30) |
|---|---|---|
| mean | 27.6 | 27.3 (barely moves) |
| max | 148 | **120** (18.9% smaller) |

Same chunk (episode 1575, frame 300) is the global worst case before *and* after
truncation -- the single-example test wasn't a fluke. Real per-chunk budget:
22 (task) + 1 (state) + 120 (action) = **143**, comfortably under the ~192 ceiling
for k=3.

**Verification probe at this real budget (160 = 143 + margin), pi0.5 backbone,
batch_size=256 (64/device), job 17497547:**

| num_context_chunks | result |
|---|---|
| 1 | OK |
| 2 | OK |
| **3** | **OK** |
| 4 | FAIL (OOM, 84.0GB single allocation) |

**3-neighbor retrieval is confirmed at batch_size=256, with the mild (keep 20/30 bins,
0.014-error) truncation cutoff.** 4 remains out of reach at this budget, consistent with
Sec 5's ~80-token estimate for k=4 -- would need the more aggressive 7/30-bin cutoff
(0.034 error, not attempted end-to-end here) and its own dataset-wide + probe
verification before trusting it the same way.

## 8. Open items

- Arms 3-5 (`yor_icl_victr_vision_canonical`/`_value_canonical`/`_vision_value_canonical`,
  jobs 17354977/17354978/17354979) are **currently running 47h jobs on the broken
  64-token-truncated context** (Sec 1). Not yet decided whether to interrupt+restart
  under the redesigned encoding or let them finish as a documented limitation.
- Implementation not yet started: new context-block assembly in `yor_retrieval.py`
  (replace `chunk_to_context_text`+`build_context_block`'s text path with the
  three-component scheme), model-side wiring in `pi0_victr.py` (a `context_state_proj`
  call reusing `state_proj`, and routing FAST-tokenized action embeddings alongside
  image/task-text in `embed_context_chunks`), precompute script changes
  (`precompute_retrieval_context.py` needs the frequency-truncation preprocessing step
  and to pull genuine 30-step windows instead of 10-step pool chunks), and a full
  precompute rebuild for arms 3-6's context pools.
- Every VICTR arm needs to **restart training from the base checkpoint**, not resume
  -- even though `state_proj`'s weights are reused, the model has never been trained
  with that projection applied to a retrieved neighbor's state in the same forward
  pass as its own.
- The bimodal cluster's root mechanism (which action-space dimensions actually drive
  the ~110-155 token outliers) is unconfirmed -- gripper transitions and position-jump
  magnitude were both ruled out; rotation-channel dynamics is the remaining
  hypothesis, untested.
- 4-neighbor retrieval is reachable in principle (budget <=~80, needs the 7/30 or more
  aggressive cutoff) but not pursued here since the immediate target was 3.
