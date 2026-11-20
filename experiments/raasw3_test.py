"""SR 85 serving test (sbatch only; one GPU; openpi venv).  Bit-identity of where_own modes 0 / 1 / 2 against the SR 84 code and semantics of the new modes 3 / 4 (W3 / W4).  Not a result.
(1) FILES: experiments/raa.py, wa_s.py, lifelong_ood/policy/backbone.py, raas_model.py, raac_model.py, serve.py equal the SR 82 prereg row's hashes; raa_w.py / wa_sw.py differ from raa.py / wa_s.py
    by 1 removed line (the guarded construction) and added lines only.
(2) DEFAULT PATH: with --where_own 0 _make_bk builds exactly PolicyBackbone(name, checkpoint_dir=ckpt, device="cuda").
(3) MODES 1 / 2 UNCHANGED: the SR 84 backbone_w.py (git commit 0bf20468, loaded as a separate module) vs the current one: torch.equal outputs of where_own_apply on random chunks and states
    (all roles, dz of both signs), and np.array_equal outputs of PolicyBackboneW.sample on the RAA-sL checkpoint with the same noise (modes 1, 2; default / full / 4 steps).
(4) MODES 3 / 4 semantics (crafted chunks, CPU): not held or held-not-clear -> W3 output == input bit for bit; held and c >= CLEAR (c = -dz) -> xy = the W1 saturating servo, other dims bit-equal; boundary
    just below / above CLEAR; W4 = W3 except the gripper dim, which is mode 2's gate; W4's gate also fires where the expert owns xy.
(5) MODEL-LEVEL (checkpoint): W3 / W4 at m = 0 frames and held-not-clear frames: np.array_equal to the base sample; at held-clear frames: xy = numpy servo, noise-independent, every other dim bit-equal (W3), W4 - W3 only dim 6.
    python experiments/raasw3_test.py
"""
import argparse, difflib, hashlib, importlib.util, json, os, subprocess, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = ["/coc/flash10/ksingara3/robot/openpi/src", ROOT, os.path.join(ROOT, "experiments")]
CK = "/coc/flash10/ksingara3/robot/checkpoints_l90c/pytorch/"
CKPT = CK + "pi05_g8v_raas_l_ae_main6000"
ROWP = os.path.join(ROOT, "ledger", "m3_prereg", "m3_prereg__libero_goal__raasl__s0__b36aa184.json")
SR84_COMMIT = "0bf20468"
ok = True


def sha(p):
    return hashlib.sha256(open(os.path.join(ROOT, p), "rb").read()).hexdigest()


def check(name, cond, extra=""):
    global ok
    ok &= bool(cond)
    print("[test] %-4s %s %s" % ("PASS" if cond else "FAIL", name, extra), flush=True)


def lines(p):
    return open(os.path.join(ROOT, p), encoding="utf-8").read().splitlines()


def load_sr84():
    src = subprocess.run(["git", "-C", ROOT, "show", "%s:lifelong_ood/policy/backbone_w.py" % SR84_COMMIT], capture_output=True, check=True).stdout
    assert b"assert mode in (1, 2), mode" in src, "not the SR 84 file"
    path = os.path.join(os.environ.get("TMPDIR", "/tmp"), "bw_sr84_%d.py" % os.getpid())
    open(path, "wb").write(src)
    spec = importlib.util.spec_from_file_location("lifelong_ood.policy._bw_sr84", path)
    m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m; spec.loader.exec_module(m)
    os.remove(path)
    return m


def main():
    global ok
    row = json.load(open(ROWP, encoding="utf-8"))["metrics"]["files_sha256"]
    for p in ("experiments/raa.py", "experiments/wa_s.py", "lifelong_ood/policy/backbone.py", "lifelong_ood/policy/raas_model.py", "lifelong_ood/policy/raac_model.py",
              "lifelong_ood/policy/serve.py"):
        check("file unchanged since the SR 82 row: %s" % p, row[p] == sha(p), sha(p)[:12])
    for new, old in (("experiments/raa_w.py", "experiments/raa.py"), ("experiments/wa_sw.py", "experiments/wa_s.py")):
        d = [l for l in difflib.unified_diff(lines(old), lines(new), lineterm="", n=0) if l[:1] in "+-" and l[:3] not in ("+++", "---")]
        minus = [l for l in d if l[0] == "-"]
        print("[test] diff %s vs %s: %d removed, %d added" % (new, old, len(minus), len(d) - len(minus)), flush=True)
        check("%s differs only by the guarded construction (1 line replaced, rest added)" % new, len(minus) == 1 and "PolicyBackbone(name, checkpoint_dir=a.ckpt" in minus[0])
    import raa_w, wa_sw
    for mod in (raa_w, wa_sw):
        calls = []

        class Fake:
            def __init__(self, *a, **k):
                calls.append((a, k))
        bk = mod._make_bk(Fake, "cfg", argparse.Namespace(ckpt="CK", where_own=0, where_log=""))
        check("%s._make_bk(where_own=0) builds PolicyBackbone('cfg', checkpoint_dir='CK', device='cuda') only" % mod.__name__,
              type(bk) is Fake and calls == [(("cfg",), dict(checkpoint_dir="CK", device="cuda"))], str(calls))
    import torch
    from lifelong_ood.policy import backbone as BB
    from lifelong_ood.policy import backbone_w as BW
    from lifelong_ood.policy import raas_model as RS
    import raas_b_test as RT
    OLD = load_sr84()
    gq = BW._grip_norm()
    check("CLEAR = 0.03 m, D_SAT, K_w unchanged", BW.CLEAR == 0.03 and BW.D_SAT == OLD.D_SAT == 0.02 and BW.k_w(0.9375) == OLD.k_w(0.9375) == 46.875 and BW.REL_RELEASE == OLD.REL_RELEASE)
    H, A = 10, 32
    cfg = RS.Pi0ConfigS(**RS.servo_fields())
    q01, q99 = np.array(cfg.st_q01, np.float32), np.array(cfg.st_q99, np.float32)

    def norm_state(raw, b=1):
        s = np.zeros((b, 32), np.float32); s[:, :4] = (np.asarray(raw, np.float32) - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
        return torch.as_tensor(s)
    # (3a) modes 1, 2: SR 84 vs current where_own_apply on random chunks / states
    g = torch.Generator().manual_seed(0)
    neq = 0
    for trial in range(400):
        n = 1 if trial % 2 == 0 else 3
        x = torch.randn(n, H * A, generator=g)
        raw = [float(torch.randn(1, generator=g) * .08), float(torch.randn(1, generator=g) * .08), float(torch.randn(1, generator=g) * .08), [0.0, 1.0, 2.0][trial % 3]]
        st = norm_state(raw)
        for mode in (1, 2):
            a = OLD.where_own_apply(x, st, cfg, mode, gq, A, H)
            b = BW.where_own_apply(x, st, cfg, mode, gq, A, H)
            neq += int(not torch.equal(a, b))
    check("modes 1 and 2: where_own_apply torch.equal to the SR 84 function on 800 random calls", neq == 0, "mismatches %d" % neq)
    # (4) modes 3 / 4 crafted
    g_open = (-1.0 - gq[0]) / (gq[1] - gq[0] + 1e-6) * 2.0 - 1.0
    g_closed = (1.0 - gq[0]) / (gq[1] - gq[0] + 1e-6) * 2.0 - 1.0
    x0 = torch.randn(1, H * A, generator=g)
    x0.reshape(1, H, A)[:, :, 6] = g_open
    kw = BW.k_w(cfg.servo_max)
    cases = (("held, clear 5 cm, far", [0.05, -0.03, -0.05, 2.0], True, True), ("held, clear 3.2 cm", [0.05, -0.03, -0.032, 2.0], True, True),
             ("held, just under CLEAR (2.9 cm)", [0.05, -0.03, -0.029, 2.0], False, True), ("held, at target height (c = 0)", [0.05, -0.03, 0.0, 2.0], False, True),
             ("held, target above object", [0.05, -0.03, 0.15, 2.0], False, True), ("held, clear, near (1 cm xy)", [0.008, -0.006, -0.05, 2.0], True, False),
             ("object role, clear", [0.05, -0.03, -0.05, 1.0], False, False), ("no role", [0.0, 0.0, -0.05, 0.0], False, False))
    for tag, raw, owns, far in cases:
        st = norm_state(raw)
        y3 = BW.where_own_apply(x0, st, cfg, 3, gq, A, H).reshape(1, H, A)
        y4, i4 = BW.where_own_apply(x0, st, cfg, 4, gq, A, H, return_info=True)
        y4 = y4.reshape(1, H, A)
        y1 = BW.where_own_apply(x0, st, cfg, 1, gq, A, H).reshape(1, H, A)
        held = raw[3] > 1.5
        exp = np.clip(kw * np.array(raw[:2]), -cfg.servo_max, cfg.servo_max) * 2.0 / (np.array(cfg.act_q99[:2]) - np.array(cfg.act_q01[:2]) + 1e-6)
        if owns:
            check("W3 xy = saturating servo [%s]" % tag, np.allclose(y3[0, :, :2].numpy(), exp[None], atol=1e-5) and torch.equal(y3, y1), str(np.round(exp, 3).tolist()))
            check("W3 other dims bit-equal to the input [%s]" % tag, torch.equal(y3[:, :, 2:], x0.reshape(1, H, A)[:, :, 2:]))
        else:
            check("W3 output == input bit for bit [%s]" % tag, torch.equal(y3.reshape(1, -1), x0))
        gate_expect = held and far
        check("W4 gate %s [%s]" % ("fires" if gate_expect else "silent", tag),
              (torch.allclose(y4[:, :, 6], torch.full((1, H), g_closed), atol=1e-5) if gate_expect else torch.equal(y4[:, :, 6], x0.reshape(1, H, A)[:, :, 6])) and i4["gate"] == gate_expect)
        check("W4 equals W3 in every dim but the gripper [%s]" % tag, torch.equal(y4[:, :, :6], y3[:, :, :6]) and torch.equal(y4[:, :, 7:], y3[:, :, 7:]))
        check("info own/c [%s]" % tag, i4["own"] == (owns if mode_ok(raw) else False) and abs(i4["c"] + raw[2]) < 1e-5)
    # (3b, 5) checkpoint
    obs = RT.raac_obs(4)
    extra = ([0.05, -0.03, 0.02, 2.0], [0.012, -0.004, 0.01, 2.0], [-0.20, 0.20, 0.0, 2.0], [0.05, -0.03, -0.05, 2.0], [0.20, 0.10, -0.031, 2.0], [0.05, -0.03, -0.029, 2.0], [0.0, 0.0, -0.06, 2.0])
    obs = obs + [dict(obs[0], **{"observation/state": np.asarray(s, np.float32)}) for s in extra]
    base_ck = {}
    res = {}
    for w in (1, 2, 3, 4):
        bk = BW.PolicyBackboneW("pi05_g8v_raas_l_ae", checkpoint_dir=CKPT, device="cuda", where_own=w)
        if w in (1, 2):
            old = OLD.PolicyBackboneW("pi05_g8v_raas_l_ae", checkpoint_dir=CKPT, device="cuda", where_own=w)
        print("[test] where_own %d: servo_k %r servo_max %r" % (w, bk.model.config.servo_k, bk.model.config.servo_max), flush=True)
        for j, o in enumerate(obs):
            ob = bk.observation(o); pre = bk.prefix(ob)
            raw = (pre.state[0, :4].float().cpu().numpy() + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
            held = raw[3] > 1.5
            clear = held and (-raw[2]) >= BW.CLEAR
            nz = np.random.default_rng(300 + j).standard_normal((1, bk.dim_full)).astype(np.float32)
            nz2 = np.random.default_rng(900 + j).standard_normal((1, bk.dim_full)).astype(np.float32)
            for kw_ in (dict(), dict(full=True), dict(num_steps=4)):
                new = bk.sample(1, pre, seed=0, noise=nz, **kw_)
                if w in (1, 2):
                    check("w%d frame %d [%s] np.array_equal to the SR 84 PolicyBackboneW" % (w, j, list(kw_)), np.array_equal(new, old.sample(1, pre, seed=0, noise=nz, **kw_)))
                else:
                    base = BB.PolicyBackbone.sample(bk, 1, pre, seed=0, noise=nz, **kw_)
                    if not clear and w == 3:
                        check("w3 frame %d (%s) [%s] == base sample bit for bit" % (j, "held, not clear" if held else "m = 0", list(kw_)), np.array_equal(base, new))
            if w in (3, 4):
                B = BB.PolicyBackbone.sample(bk, 1, pre, seed=0, noise=nz, full=True).reshape(bk.H, bk.A)
                N = bk.sample(1, pre, seed=0, noise=nz, full=True).reshape(bk.H, bk.A)
                rest = (np.arange(bk.A) != 0) & (np.arange(bk.A) != 1)
                if w == 4:
                    rest &= np.arange(bk.A) != 6
                if clear:
                    N2 = bk.sample(1, pre, seed=0, noise=nz2, full=True).reshape(bk.H, bk.A)
                    exp = np.clip(BW.k_w(bk.model.config.servo_max) * raw[:2], -bk.model.config.servo_max, bk.model.config.servo_max) * 2.0 / (np.array(bk.model.config.act_q99[:2]) - np.array(bk.model.config.act_q01[:2]) + 1e-6)
                    check("w%d frame %d held-clear: xy = numpy servo, noise independent" % (w, j), np.array_equal(N[:, :2], N2[:, :2]) and np.allclose(N[:, :2], exp[None], atol=1e-5),
                          "rel %s cm" % np.round(raw[:3] * 100, 1).tolist())
                    check("w%d frame %d held-clear: other dims bit-equal to base" % (w, j), np.array_equal(B[:, rest], N[:, rest]))
                else:
                    check("w%d frame %d not owned: xy bit-equal to base" % (w, j), np.array_equal(B[:, :2], N[:, :2]))
                    check("w%d frame %d not owned: dims except gripper bit-equal to base" % (w, j), np.array_equal(B[:, rest], N[:, rest]))
            res[(w, j)] = bk.sample(1, pre, seed=0, noise=nz, full=True)
        del bk
        if w in (1, 2):
            del old
        torch.cuda.empty_cache()
    for j in range(len(obs)):
        d = np.abs(res[(3, j)] - res[(4, j)]).reshape(10, 32)
        check("frame %d: W4 - W3 differs only in the gripper dim" % j, d[:, [i for i in range(32) if i != 6]].max() == 0.0, "max grip diff %.3g" % d[:, 6].max())
    print("[test] OVERALL %s" % ("PASS" if ok else "FAIL"), flush=True)
    sys.exit(0 if ok else 1)


def mode_ok(raw):
    return raw[3] > 1.5


if __name__ == "__main__":
    main()
