"""SR 90 serving test (sbatch only; one GPU; openpi venv).  Bit-identity of where_own modes 0-7 against the SR 85 / SR 87 code and semantics of the new modes 8 / 9 (W6 / W6nl) and the target latch.  Not a result.
(1) FILES: raa.py, wa_s.py, backbone.py, raas_model.py, raac_model.py, serve.py equal the SR 82 prereg row's hashes; raa_w.py, wa_sw.py, wa_sw5.py, rg_comp_x.py, raasw5_mu.json equal the SR 87 prereg row's hashes;
    raa_w6.py differs from raa_w.py, wa_sw6.py from wa_sw5.py, rg_comp_y.py from rg_comp_x.py only in the listed lines.
(2) DEFAULT PATH (mode 0): _make_bk(where_own=0) builds exactly PolicyBackbone(name, checkpoint_dir=ckpt, device="cuda") in raa_w6 and wa_sw6.
(3) MODES 1-7 UNCHANGED: SR 85 backbone_w.py (git 77772f3d; modes 1-4) and SR 87 backbone_w.py (git 7d094a1e; modes 1-7) vs the current one: torch.equal outputs (and equal info fields) of where_own_apply on random
    chunks / states with random mu (modes 5 / 7) and np.array_equal outputs of PolicyBackboneW.sample on the RAA-sL checkpoint, same noise, for modes 1-7 (default / full / 4 steps) on 13 frames.
(4) MODES 8 / 9 (crafted chunks, CPU): the aim point (rel - mu), ownership (held, c >= CLEAR, c > h_class or |rel - mu| > r_class), the gate (held and outside the envelope), boundaries, mode 8 == mode 9 chunkwise,
    unknown class (env None) == SR 85's mode 4 bit for bit.
(5) TARGET LATCH: carry start, accept below tau, reject above, release / episode reset.
(6) MODEL-LEVEL: the class lookup (mu and envelope) from obs["prompt"]; modes 8 / 9 on the checkpoint equal where_own_apply(base sample, mu, env of the class); unknown class == mode 4.
    python experiments/raasw6_test.py
"""
import argparse, difflib, hashlib, importlib.util, json, os, subprocess, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = ["/coc/flash10/ksingara3/robot/openpi/src", ROOT, os.path.join(ROOT, "experiments")]
CK = "/coc/flash10/ksingara3/robot/checkpoints_l90c/pytorch/"
CKPT = CK + "pi05_g8v_raas_l_ae_main6000"
ROW82 = os.path.join(ROOT, "ledger", "m3_prereg", "m3_prereg__libero_goal__raasl__s0__b36aa184.json")
ROW87 = os.path.join(ROOT, "ledger", "m3_prereg", "m3_prereg__libero_goal__raasw5__s0__7d094a1e.json")
SR85, SR87 = "77772f3d", "7d094a1e"
ok = True


def sha(p):
    return hashlib.sha256(open(os.path.join(ROOT, p), "rb").read()).hexdigest()


def check(name, cond, extra=""):
    global ok
    ok &= bool(cond)
    print("[test] %-4s %s %s" % ("PASS" if cond else "FAIL", name, extra), flush=True)


def lines(p):
    return open(os.path.join(ROOT, p), encoding="utf-8").read().splitlines()


def load_old(commit, marker, tag):
    src = subprocess.run(["git", "-C", ROOT, "show", "%s:lifelong_ood/policy/backbone_w.py" % commit], capture_output=True, check=True).stdout
    assert marker in src, "not the %s file" % tag
    path = os.path.join(os.environ.get("TMPDIR", "/tmp"), "bw_%s_%d.py" % (tag, os.getpid()))
    open(path, "wb").write(src)
    spec = importlib.util.spec_from_file_location("lifelong_ood.policy._bw_" + tag, path)
    m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m; spec.loader.exec_module(m)
    os.remove(path)
    if hasattr(m, "MU_JSON"):
        m.MU_JSON = os.path.join(ROOT, "lifelong_ood", "policy", "raasw5_mu.json")
    return m


def removed_added(a, b):
    d = [l for l in difflib.unified_diff(lines(a), lines(b), lineterm="", n=0) if l[:1] in "+-" and l[:3] not in ("+++", "---")]
    return [l for l in d if l[0] == "-"], [l for l in d if l[0] == "+"]


def main():
    global ok
    r82 = json.load(open(ROW82, encoding="utf-8"))["metrics"]["files_sha256"]
    r87 = json.load(open(ROW87, encoding="utf-8"))["metrics"]["files_sha256"]
    for p in ("experiments/raa.py", "experiments/wa_s.py", "lifelong_ood/policy/backbone.py", "lifelong_ood/policy/raas_model.py", "lifelong_ood/policy/raac_model.py", "lifelong_ood/policy/serve.py"):
        check("file unchanged since the SR 82 row: %s" % p, r82[p] == sha(p), sha(p)[:12])
    for p in ("experiments/raa_w.py", "experiments/wa_sw.py", "experiments/wa_sw5.py", "experiments/rg_comp_x.py", "lifelong_ood/policy/raasw5_mu.json", "experiments/fc_n2_wx.py"):
        check("file unchanged since the SR 87 row: %s" % p, r87[p] == sha(p), sha(p)[:12])
    mi, ad = removed_added("experiments/raa_w.py", "experiments/raa_w6.py")
    check("raa_w6.py differs from raa_w.py by 4 replaced lines (RAAServer init, diag record, serve construction, argparse) and additions", len(mi) == 4 and sum("latch" in l.lower() or "choices=(0, 1, 2, 3, 4, 5, 6, 7)" in l or "roles=r.get" in l or "srv = RAAServer" in l for l in mi) >= 3, "%d removed %d added" % (len(mi), len(ad)))
    mi, ad = removed_added("experiments/wa_sw5.py", "experiments/wa_sw6.py")
    check("wa_sw6.py differs from wa_sw5.py by 6 replaced lines (init, record, serve, rg_comp_y import / meta path, argparse)", len(mi) == 6 and any("import rg_comp_x as RGC" in l for l in mi) and any("rg_comp_x_tasks.json" in l for l in mi), "%d removed %d added" % (len(mi), len(ad)))
    mi, ad = removed_added("experiments/rg_comp_x.py", "experiments/rg_comp_y.py")
    want = ("OUT = ", "META = ", "F7_K0, F6_K0", "for p in (META")
    check("rg_comp_y.py differs from rg_comp_x.py in the listed lines only (+ docstring)", len(mi) >= 4 and all(any(w in m for w in want) or m.startswith('-"""SR 87') or m.startswith("-and per F6") for m in mi), "%d removed %d added: %s" % (len(mi), len(ad), [m[:40] for m in mi]))
    env = json.load(open(os.path.join(ROOT, "lifelong_ood", "policy", "raasw6_env.json"), encoding="utf-8"))
    mu5 = json.load(open(os.path.join(ROOT, "lifelong_ood", "policy", "raasw5_mu.json"), encoding="utf-8"))["mu_m"]
    check("raasw6_env.json mu == raasw5_mu.json", all(np.allclose(env["mu_m"][k], mu5[k], atol=1e-12, rtol=0) for k in mu5) and set(env["mu_m"]) == set(mu5))
    import raa_w, wa_sw5, raa_w6, wa_sw6
    for mod in (raa_w6, wa_sw6):
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
    from lifelong_ood.policy.w6_latch import TargetLatch
    import raas_b_test as RT
    OLD85 = load_old(SR85, b"assert mode in (1, 2, 3, 4), mode", "sr85")
    OLD87 = load_old(SR87, b"assert mode in (1, 2, 3, 4, 5, 6, 7), mode", "sr87")
    gq = BW._grip_norm()
    check("CLEAR, D_SAT, K_w, REL_RELEASE, HAND_R unchanged", BW.CLEAR == OLD87.CLEAR == 0.03 and BW.D_SAT == OLD87.D_SAT == 0.02 and BW.k_w(0.9375) == OLD87.k_w(0.9375) == 46.875 and BW.REL_RELEASE == OLD87.REL_RELEASE == 0.02 and BW.HAND_R == OLD87.HAND_R == 0.03)
    H, A = 10, 32
    cfg = RS.Pi0ConfigS(**RS.servo_fields())
    q01, q99 = np.array(cfg.st_q01, np.float32), np.array(cfg.st_q99, np.float32)

    def norm_state(raw, b=1):
        s = np.zeros((b, 32), np.float32); s[:, :4] = (np.asarray(raw, np.float32) - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
        return torch.as_tensor(s)
    # (3a) modes 1-7: old vs current where_own_apply on random chunks / states, random mu
    g = torch.Generator().manual_seed(0)
    neq85 = {m: 0 for m in (1, 2, 3, 4)}; neq87 = {m: 0 for m in range(1, 8)}
    ineq = 0
    for trial in range(400):
        n = 1 if trial % 2 == 0 else 3
        x = torch.randn(n, H * A, generator=g)
        raw = [float(torch.randn(1, generator=g) * .08), float(torch.randn(1, generator=g) * .08), float(torch.randn(1, generator=g) * .08), [0.0, 1.0, 2.0][trial % 3]]
        st = norm_state(raw)
        mu_r = [float(torch.randn(1, generator=g) * .05), float(torch.randn(1, generator=g) * .05)]
        for mode in range(1, 8):
            kw = dict(mu=torch.as_tensor(mu_r, dtype=torch.float32)) if mode in (5, 7) else {}
            b, ib = BW.where_own_apply(x, st, cfg, mode, gq, A, H, return_info=True, **kw)
            a7, ia7 = OLD87.where_own_apply(x, st, cfg, mode, gq, A, H, return_info=True, **kw)
            neq87[mode] += int(not torch.equal(a7, b)); ineq += int(any(ia7[k] != ib[k] for k in ia7))
            if mode <= 4:
                a5, ia5 = OLD85.where_own_apply(x, st, cfg, mode, gq, A, H, return_info=True)
                neq85[mode] += int(not torch.equal(a5, b)); ineq += int(any(ia5[k] != ib[k] for k in ia5))
    check("modes 1-4: where_own_apply torch.equal to the SR 85 function on 1600 random calls", all(v == 0 for v in neq85.values()), "mismatches %s" % neq85)
    check("modes 1-7: where_own_apply torch.equal to the SR 87 function on 2800 random calls (random mu for 5 / 7)", all(v == 0 for v in neq87.values()), "mismatches %s" % neq87)
    check("modes 1-7: info fields equal to the old functions'", ineq == 0, "mismatches %d" % ineq)
    # (4) modes 8 / 9 crafted
    g_open = (-1.0 - gq[0]) / (gq[1] - gq[0] + 1e-6) * 2.0 - 1.0
    g_closed = (1.0 - gq[0]) / (gq[1] - gq[0] + 1e-6) * 2.0 - 1.0
    x0 = torch.randn(1, H * A, generator=g)
    x0.reshape(1, H, A)[:, :, 6] = g_open
    X0 = x0.reshape(1, H, A)
    kw_ = BW.k_w(cfg.servo_max)
    amx = (np.array(cfg.act_q99[:2]) - np.array(cfg.act_q01[:2]) + 1e-6)

    def servo(e):
        return np.clip(kw_ * np.asarray(e, np.float64), -cfg.servo_max, cfg.servo_max) * 2.0 / amx
    mu = np.array(env["mu_m"]["target: the bowl"]); h_, r_ = env["h_class_m"]["target: the bowl"], env["r_class_m"]["target: the bowl"]
    HELD = 2.0
    cases = []

    def craft(tag, rel_xy, c, role=HELD):
        cases.append((tag, [rel_xy[0], rel_xy[1], -c, role]))
    craft("far xy, high", mu + np.array([0.08, 0.0]), 0.15)
    craft("far xy, inside height (c < h)", mu + np.array([0.0, 0.06]), 0.05)
    craft("inside xy, above h (c = h + 1.5 cm)", mu + np.array([0.01, 0.0]), h_ + 0.015)
    craft("inside xy, below h, clear (inside envelope)", mu + np.array([0.01, 0.0]), h_ - 0.015)
    craft("xy just inside r, below h", mu + np.array([r_ - 0.0015, 0.0]), h_ - 0.02)
    craft("xy just outside r, below h", mu + np.array([0.0, r_ + 0.0015]), h_ - 0.02)
    craft("c just below h, xy at mu", mu, h_ - 0.0015)
    craft("c just above h, xy at mu", mu, h_ + 0.0015)
    craft("outside xy, not clear (c = 2.9 cm)", mu + np.array([0.06, 0.0]), 0.029)
    craft("inside, low (c = 1 cm)", mu, 0.01)
    craft("target above the object (c = -5 cm), xy far", mu + np.array([0.06, 0.0]), -0.05)
    craft("object role, far", mu + np.array([0.06, 0.0]), 0.15, 1.0)
    craft("no role", [0.0, 0.0], 0.15, 0.0)
    envt = (h_, r_)
    for tag, raw in cases:
        st = norm_state(raw)
        held = raw[3] > 1.5
        dec = (st[0, :4].float().numpy() + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
        rel = dec[:2].astype(np.float64); c = float(-dec[2])
        e = rel - mu.astype(np.float32); en = float(np.linalg.norm(e))
        outside = (c > h_) or (en > r_)
        clear = c >= BW.CLEAR - 1e-7
        own = held and clear and outside
        gate = held and outside
        ys = {}
        for mode in (8, 9):
            y, inf = BW.where_own_apply(x0, st, cfg, mode, gq, A, H, return_info=True, mu=torch.as_tensor(mu, dtype=torch.float32), env=envt)
            ys[mode] = y
            y = y.reshape(1, H, A)
            lab = "W6" if mode == 8 else "W6nl"
            if own:
                check("%s xy = saturating servo of rel - mu [%s]" % (lab, tag), np.allclose(y[0, :, :2].numpy(), servo(e)[None], atol=1e-5))
            else:
                check("%s xy bit-equal to the input (not owned) [%s]" % (lab, tag), torch.equal(y[:, :, :2], X0[:, :, :2]))
            check("%s other non-gripper dims bit-equal [%s]" % (lab, tag), torch.equal(y[:, :, 2:6], X0[:, :, 2:6]) and torch.equal(y[:, :, 7:], X0[:, :, 7:]))
            check("%s gate %s [%s]" % (lab, "fires" if gate else "silent", tag), (torch.allclose(y[:, :, 6], torch.full((1, H), g_closed), atol=1e-5) if gate else torch.equal(y[:, :, 6], X0[:, :, 6])) and inf["gate"] == gate)
            check("%s info own / outside / c / e_xy [%s]" % (lab, tag), inf["own"] == own and inf["outside"] == (outside if True else False) and abs(inf["c"] - c) < 1e-5 and abs(inf["e_xy"] - en) < 1e-5, "e %.2f cm c %.2f cm h %.2f r %.2f" % (100 * en, 100 * c, 100 * h_, 100 * r_))
            if not held:
                check("%s output == input bit for bit when no target is held [%s]" % (lab, tag), torch.equal(y.reshape(1, -1), x0))
        check("W6 == W6nl chunk bit for bit [%s]" % tag, torch.equal(ys[8], ys[9]))
        a8 = BW.where_own_apply(x0, st, cfg, 8, gq, A, H)                       # unknown class: env None, mu None
        a9 = BW.where_own_apply(x0, st, cfg, 9, gq, A, H, mu=torch.as_tensor(mu, dtype=torch.float32))   # env None -> mode 4 even with a mu passed
        e4 = OLD85.where_own_apply(x0, st, cfg, 4, gq, A, H)
        check("W6 / W6nl with an unknown class == SR 85's W4 bit for bit [%s]" % tag, torch.equal(a8, e4) and torch.equal(a9, e4))
    # every class: the table values are used (rack: h = 0.4 cm < CLEAR: ownership reduces to held and clear and e > ... ; plate h = 1.4 cm: W6 == W5c1 ownership)
    for ph in env["h_class_m"]:
        mu_c = np.array(env["mu_m"][ph]); hc, rc = env["h_class_m"][ph], env["r_class_m"][ph]
        st = norm_state([mu_c[0] + 0.005, mu_c[1], -0.05, HELD])
        y8 = BW.where_own_apply(x0, st, cfg, 8, gq, A, H, mu=torch.as_tensor(mu_c, dtype=torch.float32), env=(hc, rc))
        y7 = BW.where_own_apply(x0, st, cfg, 7, gq, A, H, mu=torch.as_tensor(mu_c, dtype=torch.float32))
        if hc < 0.05 and rc > 0.005:
            check("W6 == W5c1 where c = 5 cm > h_class and |e| = 0.5 cm < r_class (%s: h %.1f cm): xy equal, gate differs by design" % (ph, 100 * hc), torch.equal(y8.reshape(1, H, A)[:, :, :2], y7.reshape(1, H, A)[:, :, :2]))
    # (5) latch
    L = TargetLatch(tau=0.01)
    t0 = np.array([0.5, 0.1, 0.9])
    r = [L.apply(1, False, t0 + 5.0), L.apply(1, True, t0)]
    check("latch: not held passes the estimate through; carry start latches it", np.allclose(r[0], t0 + 5.0) and np.array_equal(r[1], t0) and L.last["start"])
    r2 = L.apply(1, True, t0 + np.array([0.004, 0.0, 0.0]))
    check("latch: a change <= tau (4 mm) is accepted", np.allclose(r2, t0 + np.array([0.004, 0, 0])) and not L.last["rejected"])
    r3 = L.apply(1, True, t0 + np.array([0.08, 0.0, 0.16]))
    check("latch: a jump > tau (8 cm, 16 cm) is rejected, the held value stays", np.allclose(r3, t0 + np.array([0.004, 0, 0])) and L.last["rejected"])
    r4 = L.apply(1, True, t0 + np.array([0.08, 0.0, 0.16]))
    check("latch: a persistent jump stays rejected", np.allclose(r4, t0 + np.array([0.004, 0, 0])) and L.n_rej == 2)
    r5 = L.apply(1, False, t0 + 1.0); r6 = L.apply(1, True, t0 + 2.0)
    check("latch: release resets, the next carry latches the new estimate", np.allclose(r6, t0 + 2.0) and L.last["start"])
    L.apply(7, True, t0 + 3.0)
    check("latch: a new episode id resets even without a release", np.allclose(L.val, t0 + 3.0) and L.ep == 7)
    check("latch: tau = raasw6_env.json tau_target_m", abs(TargetLatch().tau - env["tau_target_m"]) < 1e-15)
    # (6) checkpoint
    obs = RT.raac_obs(4)
    extra = ([0.05, -0.03, 0.02, 2.0], [0.012, -0.004, 0.01, 2.0], [-0.20, 0.20, 0.0, 2.0], [0.05, -0.03, -0.05, 2.0], [0.20, 0.10, -0.031, 2.0], [0.05, -0.03, -0.029, 2.0], [0.0, 0.0, -0.06, 2.0], [0.01, 0.0, -0.12, 2.0])
    obs = obs + [dict(obs[0], **{"observation/state": np.asarray(s, np.float32)}) for s in extra]
    print("[test] prompts of the frames: %s" % [o["prompt"] for o in obs], flush=True)
    tabm, tabh = env["mu_m"], {k: (env["h_class_m"][k], env["r_class_m"][k]) for k in env["h_class_m"]}
    for w in range(1, 10):
        bk = BW.PolicyBackboneW("pi05_g8v_raas_l_ae", checkpoint_dir=CKPT, device="cuda", where_own=w)
        old87 = OLD87.PolicyBackboneW("pi05_g8v_raas_l_ae", checkpoint_dir=CKPT, device="cuda", where_own=w) if 5 <= w <= 7 else None
        old4 = OLD85.PolicyBackboneW("pi05_g8v_raas_l_ae", checkpoint_dir=CKPT, device="cuda", where_own=4) if w >= 8 else None
        old85 = OLD85.PolicyBackboneW("pi05_g8v_raas_l_ae", checkpoint_dir=CKPT, device="cuda", where_own=w) if w <= 4 else None
        for j, o in enumerate(obs):
            ob = bk.observation(o); pre = bk.prefix(ob)
            if old87 is not None:
                old87.observation(o)
            cls = o["prompt"] if o["prompt"] in tabm else None
            if w in (5, 7, 8, 9):
                check("w%d frame %d class lookup %r -> mu %s" % (w, j, o["prompt"], cls), bk._mu_cls == cls and (bk._mu == ([float(v) for v in tabm[cls]] if cls else None)))
            if w in (8, 9):
                check("w%d frame %d envelope lookup %r" % (w, j, o["prompt"]), bk._env == (tabh[cls] if cls else None))
            nz = np.random.default_rng(300 + j).standard_normal((1, bk.dim_full)).astype(np.float32)
            for kwx in (dict(), dict(full=True), dict(num_steps=4)):
                new = bk.sample(1, pre, seed=0, noise=nz, **kwx)
                if w <= 7:
                    if old87 is not None:
                        check("w%d frame %d [%s] np.array_equal to the SR 87 PolicyBackboneW" % (w, j, list(kwx)), np.array_equal(new, old87.sample(1, pre, seed=0, noise=nz, **kwx)))
                    if w <= 4:
                        if old85 is not None:
                            check("w%d frame %d [%s] np.array_equal to the SR 85 PolicyBackboneW" % (w, j, list(kwx)), np.array_equal(new, old85.sample(1, pre, seed=0, noise=nz, **kwx)))
                else:
                    if kwx.get("num_steps"):
                        continue
                    base = BB.PolicyBackbone.sample(bk, 1, pre, seed=0, noise=nz, full=True)
                    exp = BW.where_own_apply(torch.as_tensor(base, dtype=torch.float32, device=bk.device), pre.state, bk.model.config, w, bk._gq, bk.A, bk.H,
                                             mu=tabm[cls] if cls else None, env=tabh[cls] if cls else None).cpu().numpy().astype(base.dtype)
                    exp = exp if kwx.get("full") else exp[:, bk.real]
                    check("w%d frame %d [%s] = where_own_apply(base sample, mu, env of the class)" % (w, j, list(kwx)), np.array_equal(new, exp))
                    if cls is None:
                        old4.observation(o)
                        check("w%d frame %d [%s] (unknown class) == SR 85 W4 bit for bit" % (w, j, list(kwx)), np.array_equal(new, old4.sample(1, pre, seed=0, noise=nz, **kwx)))
        del bk
        for v in (old87, old4):
            if v is not None:
                del v
        torch.cuda.empty_cache()
    bk = BW.PolicyBackboneW("pi05_g8v_raas_l_ae", checkpoint_dir=CKPT, device="cuda", where_own=8)
    for ph, want in (("target: the rack", "target: the rack"), ("target: the rack. rel: +1 -2 +3 cm, role target", "target: the rack"), ("target: the tray", None), ("object: the bowl", None), ("put the bowl on the plate", None)):
        o = dict(obs[0], prompt=ph); bk.observation(o)
        check("w8 class lookup %r -> %s" % (ph, want), bk._mu_cls == want and (bk._env is not None) == (want is not None))
    print("[test] OVERALL %s" % ("PASS" if ok else "FAIL"), flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
