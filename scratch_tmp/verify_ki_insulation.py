"""Standalone unit check for Pi0Ki's core insulation mechanism: does
jax.lax.stop_gradient on the KV cache actually zero gradient into PaliGemma
(expert 0) while sparing the action expert (expert 1, "_1"-suffixed params)?
Mirrors third_party/nyu-finger-robot/tools/fixes/smoke_test_ki_patch.py's
structure, but tests openpi.models.gemma.Module directly with the tiny "dummy"
variant -- no siglip/tokenizer/real Pi0Ki needed for this.
"""

import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp

import openpi.models.gemma as gemma

configs = [gemma.get_config("dummy"), gemma.get_config("dummy")]
rngs = nnx.Rngs(0)
llm = nnx_bridge.ToNNX(gemma.Module(configs=configs, embed_dtype="float32", adarms=True))
llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True])

B, P, S, D = 2, 5, 3, configs[0].width
prefix = jax.random.normal(jax.random.key(1), (B, P, D))
suffix = jax.random.normal(jax.random.key(2), (B, S, D))
prefix_mask = jnp.ones((B, P), dtype=bool)
suffix_mask = jnp.ones((B, S), dtype=bool)
prefix_ar = jnp.zeros((P,), dtype=bool).at[0].set(True)
suffix_ar = jnp.ones((S,), dtype=bool)
adarms_cond = jnp.ones((B, D))


def make_attn_mask(input_mask, mask_ar):
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


graphdef, state = nnx.split(llm)


def flow_loss_fn(state):
    model = nnx.merge(graphdef, state)
    prefix_attn = make_attn_mask(prefix_mask, prefix_ar)
    positions_p = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = model([prefix, None], mask=prefix_attn, positions=positions_p)
    kv_cache = jax.tree.map(jax.lax.stop_gradient, kv_cache)

    suffix_attn = make_attn_mask(suffix_mask, suffix_ar)
    prefix_attn_rep = jnp.broadcast_to(prefix_mask[:, None, :], (B, S, P))
    full_attn = jnp.concatenate([prefix_attn_rep, suffix_attn], axis=-1)
    positions_s = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
    (_prefix_out, suffix_out), _ = model(
        [None, suffix], mask=full_attn, positions=positions_s, kv_cache=kv_cache, adarms_cond=[None, adarms_cond]
    )
    return jnp.mean(jnp.square(suffix_out))


def ki_loss_fn(state):
    model = nnx.merge(graphdef, state)
    prefix_attn = make_attn_mask(prefix_mask, prefix_ar)
    positions_p = jnp.cumsum(prefix_mask, axis=1) - 1
    (lm_out, _), _ = model([prefix, None], mask=prefix_attn, positions=positions_p)
    return jnp.mean(jnp.square(lm_out))


flow_grad = nnx.grad(flow_loss_fn)(state)
ki_grad = nnx.grad(ki_loss_fn)(state)


def group_norms(grad_state):
    """Returns (paligemma/expert0 grad-abs-sum, action-expert/expert1 grad-abs-sum),
    split by whether a leaf's path contains an "_1"-suffixed name (gemma.py's
    _name(name, i) convention: expert 0 unsuffixed, expert 1 suffixed "_1")."""
    pg_total, ae_total = 0.0, 0.0
    flat = nnx.to_flat_state(grad_state) if hasattr(nnx, "to_flat_state") else list(grad_state.flat_state().items())
    for path, leaf in flat:
        path_str = "/".join(str(p) for p in path)
        value = leaf.value if hasattr(leaf, "value") else leaf
        n = float(jnp.sum(jnp.abs(value)))
        if "_1" in path_str:
            ae_total += n
        else:
            pg_total += n
    return pg_total, ae_total


pg_flow, ae_flow = group_norms(flow_grad)
pg_ki, ae_ki = group_norms(ki_grad)
print(f"flow_loss grad: paligemma(expert0)={pg_flow:.8f} action_expert(expert1)={ae_flow:.8f}")
print(f"ki_loss   grad: paligemma(expert0)={pg_ki:.8f} action_expert(expert1)={ae_ki:.8f}")

ok = True
if pg_flow > 1e-8:
    print("FAIL: flow_loss leaked gradient into paligemma (expert0) -- insulation broken")
    ok = False
if ae_flow <= 1e-8:
    print("FAIL: flow_loss did not reach action_expert (expert1) -- action expert wouldn't train")
    ok = False
if pg_ki <= 1e-8:
    print("FAIL: ki_loss did not reach paligemma (expert0) -- backbone wouldn't train via KI")
    ok = False
if ae_ki > 1e-8:
    print("FAIL: ki_loss leaked gradient into action_expert (expert1)")
    ok = False

print("PASS: insulation verified" if ok else "FAIL: insulation broken")
if not ok:
    raise SystemExit(1)
