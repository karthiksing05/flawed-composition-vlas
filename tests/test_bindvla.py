"""CPU unit tests for BindVLA (lifelong_ood/policy/bindvla.py).  SmolVLA venv:  python tests/test_bindvla.py [--full]
(also pytest-compatible).  --full also builds the real model (frozen SmolVLM2 from the HF cache) and runs forward / backward."""
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments")]
from lifelong_ood.policy import bindvla as BV  # noqa: E402


def test_paths():
    cfg = BV.BVConfig()
    assert BV.role_path("top drawer of the cabinet", 3) == ["cabinet", "drawer", "top"]
    assert BV.role_path("black bowl on the left", 3) == ["bowl", "black left", ""]
    assert BV.role_path("left compartment of the caddy", 3) == ["caddy", "compartment", "left"]
    assert BV.role_path("top drawer of the cabinet", 2) == ["cabinet", "drawer top"]
    assert BV.role_path("top drawer of the cabinet", 1) == ["cabinet drawer top"]
    p = BV.role_parse("put the black bowl in the top drawer of the cabinet", cfg)
    assert p[0]["verb"] == "put in" and p[0]["obj0"] == "bowl" and p[0]["obj1"] == "black", p
    assert (p[0]["goal0"], p[0]["goal1"], p[0]["goal2"]) == ("cabinet", "drawer", "top"), p
    p = BV.role_parse("pick up the ketchup and place it in the basket", cfg)
    assert p[0]["verb"] == "pick up in" and p[0]["obj0"] == "ketchup" and p[0]["goal0"] == "basket", p
    p = BV.role_parse("open the top drawer of the cabinet", cfg)
    assert p[0]["verb"] == "open" and (p[0]["obj0"], p[0]["obj1"], p[0]["obj2"]) == ("cabinet", "drawer", "top") and not p[0]["goal0"], p
    p = BV.role_parse("put the white mug on the left plate and put the yellow and white mug on the right plate", cfg)
    assert len(p) == 2 and p[1]["goal1"] == "right" and p[1]["goal0"] == "plate", p
    p = BV.role_parse("put the bowl to the right of the plate", cfg)
    assert p[0]["verb"] == "put to the right of" and p[0]["goal0"] == "plate", p
    flat = BV.role_parse("put the black bowl in the top drawer of the cabinet", BV.make_config("flat"))
    assert flat[0]["obj0"] == "bowl black" and flat[0]["goal0"] == "cabinet drawer top" and not flat[0]["goal1"], flat


def test_corrupt_keeps_words():
    cfg = BV.BVConfig()
    u = BV.role_parse("put the black bowl in the top drawer of the cabinet", cfg)
    rng = np.random.default_rng(0)
    for _ in range(50):
        c = BV.corrupt(u, rng, 0.5, cfg.units)
        assert set(" ".join(c[0].values()).split()) <= set(" ".join(u[0].values()).split())


def _binder(**kw):
    cfg = BV.make_config("bindvla", **kw)
    torch.manual_seed(0)
    b = BV.Binder(cfg, d_tok=32, d_vis=24, emb_scale=1.0)
    with torch.no_grad():
        b.geo.weight.normal_(0, 0.3)                       # turn geometry on so the test exercises it
    return cfg, b


def _inputs(cfg, B=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    U, L = cfg.n_units, cfg.phrase_len
    tok = torch.randn(B, U, L, 32, generator=g)
    m = torch.zeros(B, U, L, dtype=torch.bool)
    m[:, :, :3] = True
    return (torch.randn(B, 256, 24, generator=g), torch.randn(B, 256, 24, generator=g), torch.randn(B, 2, 32, generator=g), tok, m,
            torch.randn(B, 48, 32, generator=g), torch.ones(B, 48, dtype=torch.bool))


def test_nesting_containment():
    """hier: a child's mask never leaves its parent's: m_l <= m_{l-1} x max(child) / Z, so support(m_l) is in support(m_{l-1});
    concretely, zeroing the parent on a region zeroes the child there."""
    cfg, b = _binder()
    f, w, c, tok, m, fe, fm = _inputs(cfg)
    toks, valid, g, d = b(f, w, c, tok, m, fe, fm)
    lv = d["levels"]                                         # [B, S, 2, L, T]
    for l in range(1, cfg.levels):
        par, ch = lv[:, :, :, l - 1], lv[:, :, :, l]
        ratio = ch / par.clamp(min=1e-30)
        assert torch.isfinite(ratio).all()
    # hard check: a parent mask that is exactly zero somewhere gives an exactly-zero child there
    B, n, L, d_ = 1, 1, 2, cfg.d
    q = torch.randn(B, n, L, d_)
    vr = torch.ones(B, n, L, dtype=torch.bool)
    k = torch.randn(B, 256, d_)
    k[:, 128:] = 0
    k[:, 128:, 0] = -1e4
    q[:, :, 0] = 0
    q[:, :, 0, 0] = 1.0                                      # level 1 strongly prefers tokens 0..127
    masks, *_ = b.hierarchy(q, vr, k, torch.randn(B, 256, d_), BV.grid_pos("cpu"))
    assert masks[0, 0, 0, 128:].max() < 1e-6
    assert masks[0, 0, 1, 128:].max() < 1e-6


def test_no_cross_role():
    """cross_role off: the object queries do not change when the goal phrase changes."""
    cfg, b = _binder()
    f, w, c, tok, m, fe, fm = _inputs(cfg)
    q1, _ = b.queries(tok, m)
    tok2 = tok.clone()
    gi = [1 + cfg.levels + l for l in range(cfg.levels)]      # step 0's goal units
    tok2[:, gi] = torch.randn_like(tok2[:, gi])
    q2, _ = b.queries(tok2, m)
    oi = [1 + l for l in range(cfg.levels)]
    assert torch.allclose(q1[:, oi], q2[:, oi], atol=1e-6)
    assert not torch.allclose(q1[:, gi], q2[:, gi])
    cfg, b = _binder(cross_role=True)
    q1, _ = b.queries(tok, m)
    q2, _ = b.queries(tok2, m)
    assert not torch.allclose(q1[:, oi], q2[:, oi], atol=1e-6)


def test_level1_no_geometry():
    """hier: level 1 is position-free: permuting the tokens permutes its mask (no absolute-position term)."""
    cfg, b = _binder()
    B, d_ = 1, cfg.d
    q = torch.randn(B, 1, cfg.levels, d_)
    vr = torch.zeros(B, 1, cfg.levels, dtype=torch.bool)
    vr[:, :, 0] = True
    k = torch.randn(B, 256, d_)
    perm = torch.randperm(256)
    m1, *_ = b.hierarchy(q, vr, k, torch.randn(B, 256, d_), BV.grid_pos("cpu"))
    m2, *_ = b.hierarchy(q, vr, k[:, perm], torch.randn(B, 256, d_), BV.grid_pos("cpu"))
    assert torch.allclose(m1[0, 0, 0][perm], m2[0, 0, 0], atol=1e-6)
    cfg, b = _binder(hier=False)                             # flat: geometry in the image frame -> not permutation-equivariant
    m1, *_ = b.hierarchy(q, vr, k, torch.randn(B, 256, d_), BV.grid_pos("cpu"))
    m2, *_ = b.hierarchy(q, vr, k[:, perm], torch.randn(B, 256, d_), BV.grid_pos("cpu"))
    assert not torch.allclose(m1[0, 0, 0][perm], m2[0, 0, 0], atol=1e-4)


def test_shapes_and_grad():
    for setting in ("bindvla", "flat", "levels2", "cross_role", "learned_parser", "pelican", "otter", "noisy_parser"):
        cfg = BV.make_config(setting)
        b = BV.Binder(cfg, d_tok=32, d_vis=24, emb_scale=1.0)
        f, w, c, tok, m, fe, fm = _inputs(cfg)
        toks, valid, g, d = b(f, w, c, tok, m, fe, fm)
        assert toks.shape == (2, cfg.n_slots, 32), (setting, toks.shape)
        assert valid.shape == (2, cfg.n_slots) and g.shape == (2, 1, 32)
        (toks.sum() + g.sum()).backward()
        assert b.inp.weight.grad is not None and torch.isfinite(b.inp.weight.grad).all(), setting


def test_nullspace():
    rng = np.random.default_rng(0)
    pos = BV.grid_pos("cpu").numpy()
    A = rng.standard_normal((2, 16))
    X = rng.standard_normal((40, 256, 16)) * 0.3 + (pos @ A)[None] + 0.2 * np.tanh(pos[:, :1] * 3) @ rng.standard_normal((1, 16))
    P, mu, log = BV.fit_nullspace(X.astype(np.float32), pos)
    assert log[0]["r2_row"] > 0.5 and log[-1]["r2_row"] < 0.05 and log[-1]["r2_col"] < 0.05, (log[0], log[-1])
    assert np.allclose(P, P.T, atol=1e-5) and np.allclose(P @ P, P, atol=1e-4)


def test_full_model():
    """the real frozen SmolVLM2 + fresh expert: bottleneck prefix length, frozen VLM, gradient reaches binder and expert only."""
    from lifelong_ood.policy import bindvla as BV
    torch.manual_seed(0)
    for setting in ("bindvla", "vanilla_frozen"):
        cfg = BV.make_config(setting, foveate=False)
        model = BV.BindVLA(cfg, device="cpu")
        cond = BV.Conditioner(model.tokenizer, cfg)
        texts = ["put the black bowl in the top drawer of the cabinet", "turn on the stove and put the moka pot on it"]
        cs = [cond(t) for t in texts]
        B = 2
        batch = dict(img=torch.rand(B, 3, 224, 224), wrist=torch.rand(B, 3, 224, 224), state=torch.zeros(B, 32),
                     actions=torch.randn(B, cfg.chunk, 32), lang_tokens=torch.from_numpy(np.stack([c[0] for c in cs])),
                     lang_masks=torch.from_numpy(np.stack([c[1] for c in cs])), unit_tokens=torch.from_numpy(np.stack([c[2] for c in cs])),
                     unit_masks=torch.from_numpy(np.stack([c[3] for c in cs])))
        pe, pp, pa, d = model.build_prefix(batch)
        if setting == "bindvla":
            assert pe.shape[1] == cfg.n_slots + 2, pe.shape          # role tokens + global + state: no image or language tokens
            assert d["levels"].shape == (B, cfg.n_steps, 2, cfg.levels, 256)
        else:
            assert pe.shape[1] == 64 + 64 + cfg.tok_len + 1, pe.shape
        loss, _ = model(batch, torch.randn(B, cfg.chunk, 32), torch.rand(B))
        loss[:, :, :7].mean().backward()
        vlm = model.flow.vlm_with_expert.vlm
        assert all(p.grad is None for p in vlm.parameters())
        assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in model.flow.vlm_with_expert.lm_expert.parameters())
        if model.binder is not None:
            assert float(model.binder.slot_mlp[0].weight.grad.abs().sum()) > 0
            assert float(model.binder.wk.weight.grad.abs().sum()) > 0
        x = model.sample(batch, torch.randn(B, cfg.chunk, 32), num_steps=2)
        assert x.shape == (B, cfg.chunk, 32) and torch.isfinite(x).all()
        print(setting, "params", model.n_trainable())


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and (k != "test_full_model" or "--full" in sys.argv)]
    for t in tests:
        t()
        print("ok", t.__name__, flush=True)
    print(f"all {len(tests)} passed")
