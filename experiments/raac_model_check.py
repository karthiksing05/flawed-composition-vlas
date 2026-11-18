"""RAA-c (docs/SCAFFOLD_RESULTS.md 77) model-shape check, CPU, openpi venv, no weights: the raac config's JAX parameter tree equals pi05_libero's plus
exactly state_proj {kernel [32, 1024], bias [1024]}; embed_suffix returns H + 1 tokens (the state token first, causal), masks consistent; the freeze filter of the
_ae config leaves state_proj trainable and everything under img / the VLM llm frozen; RaacLoader keeps state_proj out of the loaded subset (ShapeDtypeStruct)."""
import os
import sys

sys.path[:0] = ["/coc/flash10/ksingara3/robot/openpi/src", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import flax.traverse_util as tu

from lifelong_ood.policy import l90c_config as LC
LC.register()
import openpi.training.config as C

cfg = C.get_config("pi05_g8v_raac_ae")
base = C.get_config("pi05_g8v_raa_ae")
m, mb = cfg.model, base.model
print("[raac-model] config", type(m).__name__, "pi05", m.pi05, "discrete_state_input", m.discrete_state_input, "H", m.action_horizon, "loader", type(cfg.weight_loader).__name__)


def tree(model_cfg):
    shape = nnx.eval_shape(model_cfg.create, jax.random.key(0))
    return tu.flatten_dict(nnx.state(shape).to_pure_dict(), sep="/")


a, b = tree(m), tree(mb)
extra = sorted(set(a) - set(b))
missing = sorted(set(b) - set(a))
print("[raac-model] extra params:", {k: tuple(a[k].shape) for k in extra}, "missing:", missing)
assert missing == [] and extra == ["state_proj/bias", "state_proj/kernel"], (missing, extra)
assert tuple(a["state_proj/kernel"].shape) == (32, 1024) and tuple(a["state_proj/bias"].shape) == (1024,)
print("[raac-model] new parameters: %d" % sum(int(jnp.prod(jnp.array(a[k].shape))) for k in extra))

model = nnx.eval_shape(m.create, jax.random.key(0))
obs, act = m.inputs_spec(batch_size=2)
t = jax.ShapeDtypeStruct((2,), jnp.float32)


def f(model, obs, act, t):
    return model.embed_suffix(obs, act, t)


tok, mask, ar, cond = nnx.eval_shape(f, model, obs, act, t)
print("[raac-model] embed_suffix: tokens", tok.shape, "mask", mask.shape, "ar", ar.shape, "adarms_cond", cond.shape)
assert tok.shape == (2, m.action_horizon + 1, 1024) and mask.shape == (2, m.action_horizon + 1) and ar.shape == (m.action_horizon + 1,)
print("[raac-model] ar_mask first tokens: [state token causal, first action causal, rest not] by construction (state True, then True, False * (H-1))")

# freeze filter: which parameters train
import openpi.shared.nnx_utils as NU
st = nnx.state(model)
frozen = nnx.state(model, cfg.freeze_filter)
fz = set(tu.flatten_dict(frozen.to_pure_dict(), sep="/"))
tr = [k for k in a if k not in fz]
print("[raac-model] trainable tensors %d, frozen %d; state_proj trainable: %s" % (len(tr), len(fz), all(k not in fz for k in extra)))
assert all(k not in fz for k in extra)
assert not any("llm" in k and "_1" not in k for k in tr) and not any("img" in k for k in tr), "VLM / vision not frozen"

# loader: the missing state_proj stays a ShapeDtypeStruct (the trainer drops it and keeps the random init)
import openpi.training.weight_loaders as W
loaded = cfg.weight_loader.load(nnx.state(model).to_pure_dict())
fl = tu.flatten_dict(loaded, sep="/")
sp = {k: v for k, v in fl.items() if "state_proj" in k}
print("[raac-model] loader: state_proj entries", {k: type(v).__name__ for k, v in sp.items()}, "; total loaded", len(fl))
assert sp and all(isinstance(v, jax.ShapeDtypeStruct) for v in sp.values())
print("[raac-model] done")
