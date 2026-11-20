"""SR 87 serving test (sbatch only; one GPU; openpi venv).  Bit-identity of where_own modes 0-4 against the SR 85 code and semantics of the new modes 5 / 6 / 7 (W5 / W5c2 / W5c1).  Not a result.
(1) FILES: experiments/raa.py, wa_s.py, lifelong_ood/policy/backbone.py, raas_model.py, raac_model.py, serve.py equal the SR 82 prereg row's hashes; raa_w.py / wa_sw.py still differ from raa.py / wa_s.py by the one
    replaced construction line; wa_sw5.py differs from wa_sw.py by exactly two replaced lines (the rg_comp_x import and the selftest's meta path); rg_comp_x.py differs from rg_comp.py in the listed constants only.
(2) DEFAULT PATH (mode 0): _make_bk(where_own=0) builds exactly PolicyBackbone(name, checkpoint_dir=ckpt, device="cuda").
(3) MODES 1-4 UNCHANGED: the SR 85 backbone_w.py (git commit 77772f3d, loaded as a separate module) vs the current one: torch.equal outputs (and equal info fields) of where_own_apply on 800 random
    chunks / states per mode, and np.array_equal outputs of PolicyBackboneW.sample on the RAA-sL checkpoint, same noise, for modes 1-4 (default / full / 4 steps) on 13 frames.
(4) MODES 5 / 6 / 7 semantics (crafted chunks, CPU): the aim point, the handoff radius R = 3 cm, the gate radius, CLEAR, held / not held, mu = 0 equalities.
(5) MODEL-LEVEL: the class lookup from obs["prompt"] (the training demos' role phrases, a channel-suffixed phrase, an unknown phrase, an object phrase); modes 5 / 7 on the checkpoint equal where_own_apply(base sample, mu of the class);
    mode 7 with an unknown class equals SR 85's mode 4 bit for bit (mu = 0, no R, gate 2 cm).
    python experiments/raasw5_test.py
"""
import argparse, difflib, hashlib, importlib.util, json, os, subprocess, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = ["/coc/flash10/ksingara3/robot/openpi/src", ROOT, os.path.join(ROOT, "experiments")]
CK = "/coc/flash10/ksingara3/robot/checkpoints_l90c/pytorch/"
CKPT = CK + "pi05_g8v_raas_l_ae_main6000"
ROWP = os.path.join(ROOT, "ledger", "m3_prereg", "m3_prereg__libero_goal__raasl__s0__b36aa184.json")
SR85_COMMIT = "77772f3d"
ok = True


def sha(p):
    return hashlib.sha256(open(os.path.join(ROOT, p), "rb").read()).hexdigest()


def check(name, cond, extra=""):
    global ok
    ok &= bool(cond)
    print("[test] %-4s %s %s" % ("PASS" if cond else "FAIL", name, extra), flush=True)


def lines(p):
    return open(os.path.join(ROOT, p), encoding="utf-8").read().splitlines()


def load_old():
    src = subprocess.run(["git", "-C", ROOT, "show", "%s:lifelong_ood/policy/backbone_w.py" % SR85_COMMIT], capture_output=True, check=True).stdout
    assert b"assert mode in (1, 2, 3, 4), mode" in src, "not the SR 85 file"
    path = os.path.join(os.environ.get("TMPDIR", "/tmp"), "bw_sr85_%d.py" % os.getpid())
    open(path, "wb").write(src)
    spec = importlib.util.spec_from_file_location("lifelong_ood.policy._bw_sr85", path)
    m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m; spec.loader.exec_module(m)
    os.remove(path)
    return m


def removed_added(a, b):
    d = [l for l in difflib.unified_diff(lines(a), lines(b), lineterm="", n=0) if l[:1] in "+-" and l[:3] not in ("+++", "---")]
    return [l for l in d if l[0] == "-"], [l for l in d if l[0] == "+"]


def main():
    global ok
    row = json.load(open(ROWP, encoding="utf-8"))["metrics"]["files_sha256"]
    for p in ("experiments/raa.py", "experiments/wa_s.py", "lifelong_ood/policy/backbone.py", "lifelong_ood/policy/raas_model.py", "lifelong_ood/policy/raac_model.py", "lifelong_ood/policy/serve.py"):
        check("file unchanged since the SR 82 row: %s" % p, row[p] == sha(p), sha(p)[:12])
    for new, old in (("experiments/raa_w.py", "experiments/raa.py"), ("experiments/wa_sw.py", "experiments/wa_s.py")):
        mi, ad = removed_added(old, new)
        check("%s differs from %s only by the guarded construction (1 line replaced, rest added)" % (new, old), len(mi) == 1 and "PolicyBackbone(name, checkpoint_dir=a.ckpt" in mi[0], "%d removed %d added" % (len(mi), len(ad)))
    mi, ad = removed_added("experiments/wa_sw.py", "experiments/wa_sw5.py")
    check("wa_sw5.py differs from wa_sw.py by exactly two replaced lines (rg_comp_x import, selftest meta path)", len(mi) == 2 and len(ad) == 2 and "import rg_comp as RGC" in mi[0] and "rg_comp_tasks.json" in mi[1]
          and "import rg_comp_x as RGC" in ad[0] and "rg_comp_x_tasks.json" in ad[1], "%d removed %d added" % (len(mi), len(ad)))
    mi, ad = removed_added("experiments/rg_comp.py", "experiments/rg_comp_x.py")
    want = ("OUT = ", "META = ", "F7_K0, F6_K0", "N_SMOKE, N7_MAIN", "for p in (META, os.path.join(ROOT")
    check("rg_comp_x.py differs from rg_comp.py in the five listed lines (plus the added docstring)", len(mi) == 5 and all(any(m.startswith("-" + w) or ("-    " + w) in m for w in want) for m in mi), "%d removed %d added" % (len(mi), len(ad)))
    import raa_w, wa_sw, wa_sw5
    for mod in (raa_w, wa_sw, wa_sw5):
        calls = []

        class Fake:
            def __init__(self, *a, **k):
                calls.append((a, k))
        bk = mod._make_bk(Fake, "cfg", argparse.Namespace(ckpt="CK", where_own=0, where_log=""))
        check("%s._make_bk(where_own=0) builds PolicyBackbone('cfg', checkpoint_dir='CK', device='cuda') only" % mod.__name__, type(bk) is Fake and calls == [(("cfg",), dict(checkpoint_dir="CK", device="cuda"))], str(calls))
    import torch
    from lifelong_ood.policy import backbone as BB
    from lifelong_ood.policy import backbone_w as BW
    from lifelong_ood.policy import raas_model as RS
    import raas_b_test as RT
    OLD = load_old()
    gq = BW._grip_norm()
    check("CLEAR, D_SAT, K_w, REL_RELEASE unchanged; HAND_R = 0.03", BW.CLEAR == OLD.CLEAR == 0.03 and BW.D_SAT == OLD.D_SAT == 0.02 and BW.k_w(0.9375) == OLD.k_w(0.9375) == 46.875 and BW.REL_RELEASE == OLD.REL_RELEASE == 0.02 and BW.HAND_R == 0.03)
    H, A = 10, 32
    cfg = RS.Pi0ConfigS(**RS.servo_fields())
    q01, q99 = np.array(cfg.st_q01, np.float32), np.array(cfg.st_q99, np.float32)

    def norm_state(raw, b=1):
        s = np.zeros((b, 32), np.float32); s[:, :4] = (np.asarray(raw, np.float32) - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
        return torch.as_tensor(s)
    # (3a) modes 1-4: SR 85 vs current where_own_apply on random chunks / states
    g = torch.Generator().manual_seed(0)
    neq = {m: 0 for m in (1, 2, 3, 4)}
    ineq = 0
    for trial in range(400):
        n = 1 if trial % 2 == 0 else 3
        x = torch.randn(n, H * A, generator=g)
        raw = [float(torch.randn(1, generator=g) * .08), float(torch.randn(1, generator=g) * .08), float(torch.randn(1, generator=g) * .08), [0.0, 1.0, 2.0][trial % 3]]
        st = norm_state(raw)
        for mode in (1, 2, 3, 4):
            a, ia = OLD.where_own_apply(x, st, cfg, mode, gq, A, H, return_info=True)
            b, ib = BW.where_own_apply(x, st, cfg, mode, gq, A, H, return_info=True)
            neq[mode] += int(not torch.equal(a, b))
            ineq += int(any(ia[k] != ib[k] for k in ia))
    check("modes 1-4: where_own_apply torch.equal to the SR 85 function on 1600 random calls", all(v == 0 for v in neq.values()), "mismatches %s" % neq)
    check("modes 1-4: info fields (m, rel_xy, gate, c, own) equal to SR 85's", ineq == 0, "mismatches %d" % ineq)
    # (4) modes 5 / 6 / 7 crafted
    g_open = (-1.0 - gq[0]) / (gq[1] - gq[0] + 1e-6) * 2.0 - 1.0
    g_closed = (1.0 - gq[0]) / (gq[1] - gq[0] + 1e-6) * 2.0 - 1.0
    x0 = torch.randn(1, H * A, generator=g)
    x0.reshape(1, H, A)[:, :, 6] = g_open
    X0 = x0.reshape(1, H, A)
    kw = BW.k_w(cfg.servo_max)
    amx = (np.array(cfg.act_q99[:2]) - np.array(cfg.act_q01[:2]) + 1e-6)

    def servo(e):
        return np.clip(kw * np.asarray(e, np.float64), -cfg.servo_max, cfg.servo_max) * 2.0 / amx
    mu = np.array([-0.0894, 0.0813])                                         # the rack's order of magnitude
    HELD = 2.0
    # (name, raw channel [dx, dy, dz, role], expected: own5, own6, own7, gate5, gate6, gate7)  with e = rel - mu for modes 5 / 7, e = rel for mode 6
    def craft(tag, rel_xy, dz, role=HELD):
        return tag, [rel_xy[0], rel_xy[1], dz, role]
    cases = [craft("at mu + 5 cm (module far from aim), clear", mu + np.array([0.05, 0.0]), -0.05),
             craft("at mu + 3.2 cm, clear", mu + np.array([0.032, 0.0]), -0.05),
             craft("at mu + 2.9 cm, clear (inside R)", mu + np.array([0.029, 0.0]), -0.05),
             craft("at mu + 2.5 cm, clear (inside R, between 2 and 3 cm)", mu + np.array([0.0, -0.025]), -0.05),
             craft("at mu + 1.5 cm, clear", mu + np.array([0.0, 0.015]), -0.05),
             craft("at mu exactly, clear", mu, -0.05),
             craft("at mu + 5 cm, held but not clear (c = 2.9 cm)", mu + np.array([0.05, 0.0]), -0.029),
             craft("at mu + 5 cm, target above the object", mu + np.array([0.05, 0.0]), 0.12),
             craft("at the reference (rel = 0), clear (|rel - mu| = 12 cm)", np.zeros(2), -0.05),
             craft("rel small (2.5 cm), clear, far from mu", np.array([0.025, 0.0]), -0.05),
             craft("object role, clear", mu + np.array([0.05, 0.0]), -0.05, 1.0), craft("no role", [0.0, 0.0], -0.05, 0.0)]
    for tag, raw in cases:
        st = norm_state(raw)
        held = raw[3] > 1.5
        clear = held and (-raw[2]) >= BW.CLEAR - 1e-7
        rel = np.array(raw[:2], np.float64)
        # float32 round trip of the state: use the module's own decoded values for the expectations
        dec = (st[0, :4].float().numpy() + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
        rel = dec[:2].astype(np.float64)
        e5 = rel - mu.astype(np.float32); e6 = rel
        for mode, e, R_own, G in ((5, e5, True, BW.HAND_R), (6, e6, True, BW.HAND_R), (7, e5, False, BW.REL_RELEASE)):
            y, inf = BW.where_own_apply(x0, st, cfg, mode, gq, A, H, return_info=True, mu=torch.as_tensor(mu, dtype=torch.float32))
            y = y.reshape(1, H, A)
            en = float(np.linalg.norm(e))
            own = held and clear and (en > 0.03 if R_own else True)
            gate = held and en > G
            lab = "W5" if mode == 5 else ("W5c2" if mode == 6 else "W5c1")
            if own:
                check("%s xy = saturating servo of %s [%s]" % (lab, "rel - mu" if mode != 6 else "rel", tag), np.allclose(y[0, :, :2].numpy(), servo(e)[None], atol=1e-5), str(np.round(servo(e), 3).tolist()))
                check("%s other non-gripper dims bit-equal to the input [%s]" % (lab, tag), torch.equal(y[:, :, 2:6], X0[:, :, 2:6]) and torch.equal(y[:, :, 7:], X0[:, :, 7:]))
            else:
                check("%s xy bit-equal to the input (not owned) [%s]" % (lab, tag), torch.equal(y[:, :, :2], X0[:, :, :2]) and torch.equal(y[:, :, 2:6], X0[:, :, 2:6]))
            check("%s gate %s [%s]" % (lab, "fires" if gate else "silent", tag), (torch.allclose(y[:, :, 6], torch.full((1, H), g_closed), atol=1e-5) if gate else torch.equal(y[:, :, 6], X0[:, :, 6])) and inf["gate"] == gate)
            check("%s info own / c / e_xy [%s]" % (lab, tag), inf["own"] == own and abs(inf["c"] + raw[2]) < 1e-5 and abs(inf["e_xy"] - en) < 1e-5, "e %.2f cm" % (100 * en))
            if not held:
                check("%s output == input bit for bit when no target is held [%s]" % (lab, tag), torch.equal(y.reshape(1, -1), x0))
    for tag, raw in cases:
        st = norm_state(raw)
        a = BW.where_own_apply(x0, st, cfg, 5, gq, A, H, mu=torch.zeros(2))
        b = BW.where_own_apply(x0, st, cfg, 6, gq, A, H)
        c = BW.where_own_apply(x0, st, cfg, 5, gq, A, H)                       # mu None = unknown class
        check("W5 with mu = 0 == W5c2 == W5 with no mu [%s]" % tag, torch.equal(a, b) and torch.equal(a, c))
        d = BW.where_own_apply(x0, st, cfg, 7, gq, A, H)                        # unknown class
        e4 = OLD.where_own_apply(x0, st, cfg, 4, gq, A, H)
        check("W5c1 with no mu == SR 85's W4 bit for bit [%s]" % tag, torch.equal(d, e4))
    # (5) checkpoint
    obs = RT.raac_obs(4)
    extra = ([0.05, -0.03, 0.02, 2.0], [0.012, -0.004, 0.01, 2.0], [-0.20, 0.20, 0.0, 2.0], [0.05, -0.03, -0.05, 2.0], [0.20, 0.10, -0.031, 2.0], [0.05, -0.03, -0.029, 2.0], [0.0, 0.0, -0.06, 2.0])
    obs = obs + [dict(obs[0], **{"observation/state": np.asarray(s, np.float32)}) for s in extra]
    print("[test] prompts of the frames: %s" % [o["prompt"] for o in obs], flush=True)
    tab = json.load(open(BW.MU_JSON, encoding="utf-8"))["mu_m"]
    res = {}
    for w in (1, 2, 3, 4, 5, 6, 7):
        bk = BW.PolicyBackboneW("pi05_g8v_raas_l_ae", checkpoint_dir=CKPT, device="cuda", where_own=w)
        old = OLD.PolicyBackboneW("pi05_g8v_raas_l_ae", checkpoint_dir=CKPT, device="cuda", where_own=(w if w <= 4 else 4)) if w in (1, 2, 3, 4, 7) else None
        for j, o in enumerate(obs):
            ob = bk.observation(o); pre = bk.prefix(ob)
            cls = o["prompt"] if o["prompt"] in tab else None
            if w in (5, 7):
                check("w%d frame %d class lookup %r -> mu %s" % (w, j, o["prompt"], cls), bk._mu_cls == cls and (bk._mu == (tab[cls] if cls else None)))
            raw = (pre.state[0, :4].float().cpu().numpy() + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
            nz = np.random.default_rng(300 + j).standard_normal((1, bk.dim_full)).astype(np.float32)
            for kw_ in (dict(), dict(full=True), dict(num_steps=4)):
                new = bk.sample(1, pre, seed=0, noise=nz, **kw_)
                if w <= 4:
                    check("w%d frame %d [%s] np.array_equal to the SR 85 PolicyBackboneW" % (w, j, list(kw_)), np.array_equal(new, old.sample(1, pre, seed=0, noise=nz, **kw_)))
                else:
                    base = BB.PolicyBackbone.sample(bk, 1, pre, seed=0, noise=nz, full=True)
                    m_ = None if w == 6 or cls is None else tab[cls]
                    exp = BW.where_own_apply(torch.as_tensor(base, dtype=torch.float32, device=bk.device), pre.state, bk.model.config, w, bk._gq, bk.A, bk.H, mu=m_ if w == 5 else (m_ if w == 7 else None))
                    exp = exp.cpu().numpy().astype(base.dtype)
                    exp = exp if kw_.get("full") else exp[:, bk.real]
                    if kw_.get("num_steps"):
                        continue
                    check("w%d frame %d [%s] = where_own_apply(base sample, mu of the class)" % (w, j, list(kw_)), np.array_equal(new, exp))
                    if w == 7 and cls is None:
                        check("w7 frame %d [%s] (unknown class) == SR 85 W4 bit for bit" % (j, list(kw_)), np.array_equal(new, old.sample(1, pre, seed=0, noise=nz, **kw_)))
            res[(w, j)] = bk.sample(1, pre, seed=0, noise=nz, full=True)
        del bk
        if old is not None:
            del old
        torch.cuda.empty_cache()
    bk = BW.PolicyBackboneW("pi05_g8v_raas_l_ae", checkpoint_dir=CKPT, device="cuda", where_own=5)
    for ph, want in (("target: the rack", "target: the rack"), ("target: the rack. rel: +1 -2 +3 cm, role target", "target: the rack"), ("target: the tray", None), ("object: the bowl", None), ("put the bowl on the plate", None)):
        o = dict(obs[0], prompt=ph); bk.observation(o)
        check("w5 class lookup %r -> %s" % (ph, want), bk._mu_cls == want)
    print("[test] OVERALL %s" % ("PASS" if ok else "FAIL"), flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
