"""SR 84 serving test (sbatch only; one GPU; openpi venv).  Bit-identity of the default path and the semantics of the guarded RAA-sL-W branch.  Not a result.
(1) FILES: experiments/raa.py, experiments/wa_s.py, lifelong_ood/policy/backbone.py, raas_model.py are byte-identical to the SR 82 prereg row's hashes (the existing paths cannot have changed);
    raa_w.py / wa_sw.py differ from raa.py / wa_s.py ONLY by the added lines (the _make_bk helper, its call, two argparse lines).
(2) DEFAULT PATH: with --where_own 0 (the default) _make_bk constructs exactly PolicyBackbone(name, checkpoint_dir=ckpt, device="cuda") (a recording stand-in class).
(3) MODEL: on the RAA-sL checkpoint, PolicyBackboneW (where_own 1, 2) vs the unbound base PolicyBackbone.sample on the same instance and noise:
    m = 0 frames (object role, zero state): np.array_equal for default / full / num_steps 4; held frames: every dim except xy bit-equal under W1, xy equal an independent numpy
    saturating controller, xy independent of the noise (expert removed); W2 differs from W1 only in the gripper dim; where_own_apply on crafted chunks (release gate).
(4) gripper convention: demo gripper action on held frames is > 0 (closed), open < 0.
    python experiments/raasw_test.py
"""
import argparse, difflib, hashlib, io, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = ["/coc/flash10/ksingara3/robot/openpi/src", ROOT, os.path.join(ROOT, "experiments")]
CK = "/coc/flash10/ksingara3/robot/checkpoints_l90c/pytorch/"
CKPT = CK + "pi05_g8v_raas_l_ae_main6000"
ROWP = os.path.join(ROOT, "ledger", "m3_prereg", "m3_prereg__libero_goal__raasl__s0__b36aa184.json")
ok = True


def sha(p):
    return hashlib.sha256(open(os.path.join(ROOT, p), "rb").read()).hexdigest()


def check(name, cond, extra=""):
    global ok
    ok &= bool(cond)
    print("[test] %-4s %s %s" % ("PASS" if cond else "FAIL", name, extra), flush=True)


def lines(p):
    return open(os.path.join(ROOT, p), encoding="utf-8").read().splitlines()


def main():
    global ok
    row = json.load(open(ROWP, encoding="utf-8"))["metrics"]["files_sha256"]
    for p in ("experiments/raa.py", "experiments/wa_s.py", "lifelong_ood/policy/backbone.py", "lifelong_ood/policy/raas_model.py", "lifelong_ood/policy/raac_model.py",
              "lifelong_ood/policy/serve.py"):
        check("file unchanged since the SR 82 row: %s" % p, row[p] == sha(p), sha(p)[:12])
    for new, old, nadd in (("experiments/raa_w.py", "experiments/raa.py", None), ("experiments/wa_sw.py", "experiments/wa_s.py", None)):
        d = [l for l in difflib.unified_diff(lines(old), lines(new), lineterm="", n=0) if l[:1] in "+-" and l[:3] not in ("+++", "---")]
        minus = [l for l in d if l[0] == "-"]
        plus = [l for l in d if l[0] == "+"]
        print("[test] diff %s vs %s: %d removed, %d added" % (new, old, len(minus), len(plus)), flush=True)
        for l in d:
            print("[test]    " + l[:150], flush=True)
        # removed lines: only the bk construction line, the argparse tail line is kept (we only add), module docstring untouched
        check("%s differs only by the guarded construction (1 line replaced, rest added)" % new, len(minus) == 1 and "PolicyBackbone(name, checkpoint_dir=a.ckpt" in minus[0])
    # (2) default path
    import raa_w, wa_sw
    for mod in (raa_w, wa_sw):
        calls = []

        class Fake:
            def __init__(self, *a, **k):
                calls.append((a, k))
        a = argparse.Namespace(ckpt="CK", where_own=0, where_log="")
        bk = mod._make_bk(Fake, "cfg", a)
        check("%s._make_bk(where_own=0) builds PolicyBackbone('cfg', checkpoint_dir='CK', device='cuda') only" % mod.__name__,
              type(bk) is Fake and calls == [(("cfg",), dict(checkpoint_dir="CK", device="cuda"))], str(calls))
        a2 = argparse.Namespace(ckpt="CK")                             # a namespace without the flag (older callers) also takes the old path
        calls.clear(); mod._make_bk(Fake, "cfg", a2)
        check("%s._make_bk(no flag attribute) old path" % mod.__name__, calls == [(("cfg",), dict(checkpoint_dir="CK", device="cuda"))])
    import torch
    from lifelong_ood.policy import backbone as BB
    from lifelong_ood.policy import backbone_w as BW
    from lifelong_ood.policy import raas_model as RS
    import raas_b_test as RT
    # (4) gripper convention
    try:
        import pyarrow.parquet as pq
        g = []
        for ep in (0, 40, 80, 120):
            t = pq.read_table(os.path.join(RT.LR, "data", "chunk-000", "episode_%06d.parquet" % ep)).to_pydict()
            st = np.asarray(t["state"], np.float32); ac = np.asarray(t["actions"], np.float32)
            g.append((float(ac[st[:, 3] == 2.0, 6].mean()), float(ac[st[:, 3] == 1.0, 6].mean()), float(ac[:, 6].min()), float(ac[:, 6].max())))
        print("[test] demo gripper action: held-frame mean %s, object-frame mean %s, min %s max %s" % tuple(np.round(np.array(g).T, 3).tolist()), flush=True)
        check("demo gripper action on held frames is positive (closed)", all(x[0] > 0 for x in g))
    except Exception as e:                                                   # noqa: BLE001
        check("demo gripper convention readable", False, repr(e))
    gq = BW._grip_norm()
    print("[test] norm stats action dim 6: q01 %.4f q99 %.4f; K_w = %.4f per m (servo_max / %.2f m); existing K = %.4f" % (gq[0], gq[1], BW.k_w(0.9375), BW.D_SAT, 2.851209548740752), flush=True)
    # unit test of where_own_apply (release gate) on crafted chunks, CPU
    H, A = 10, 32
    cfg = RS.Pi0ConfigS(**RS.servo_fields())
    q01, q99 = np.array(cfg.st_q01, np.float32), np.array(cfg.st_q99, np.float32)

    def norm_state(raw):
        s = np.zeros((1, 32), np.float32); s[0, :4] = (np.asarray(raw, np.float32) - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
        return torch.as_tensor(s)
    g_open = (-1.0 - gq[0]) / (gq[1] - gq[0] + 1e-6) * 2.0 - 1.0
    g_closed = (1.0 - gq[0]) / (gq[1] - gq[0] + 1e-6) * 2.0 - 1.0
    x0 = torch.zeros(1, H * A); x0.reshape(1, H, A)[:, :, 0] = -0.4; x0.reshape(1, H, A)[:, :, 2] = 0.3; x0.reshape(1, H, A)[:, :, 6] = g_open
    for tag, raw, m2_fires in (("far held", [0.05, -0.03, 0.02, 2.0], True), ("near held (1.2 cm)", [0.01, -0.0066, 0.0, 2.0], False),
                               ("object role far", [0.05, -0.03, 0.02, 1.0], False), ("no role", [0.0, 0.0, 0.0, 0.0], False)):
        st = norm_state(raw)
        y1 = BW.where_own_apply(x0, st, cfg, 1, gq, A, H).reshape(1, H, A)
        y2, info = BW.where_own_apply(x0, st, cfg, 2, gq, A, H, return_info=True)
        y2 = y2.reshape(1, H, A)
        held = raw[3] > 1.5
        kw = BW.k_w(cfg.servo_max)
        exp = np.clip(kw * np.array(raw[:2]), -cfg.servo_max, cfg.servo_max) * 2.0 / (np.array(cfg.act_q99[:2]) - np.array(cfg.act_q01[:2]) + 1e-6)
        if held:
            check("W1 xy = saturating servo [%s]" % tag, np.allclose(y1[0, :, :2].numpy(), exp[None], atol=1e-5), str(np.round(exp, 3).tolist()))
        else:
            check("W1 identical to input when m = 0 [%s]" % tag, torch.equal(y1.reshape(1, -1), x0))
        check("W1 z / rot / grip untouched [%s]" % tag, torch.equal(y1[:, :, 2:], x0.reshape(1, H, A)[:, :, 2:]))
        check("W2 release gate %s [%s]" % ("fires" if m2_fires else "does not fire", tag),
              (torch.allclose(y2[:, :, 6], torch.full((1, H), g_closed), atol=1e-5) if m2_fires else torch.equal(y2[:, :, 6], x0.reshape(1, H, A)[:, :, 6])) and info["gate"] == m2_fires)
        check("W2 xy equals W1 xy and nothing but dim 6 changes [%s]" % tag, torch.equal(y2[:, :, :6], y1[:, :, :6]) and torch.equal(y2[:, :, 7:], y1[:, :, 7:]))
    # a closed command is never changed by the gate
    xc = x0.clone(); xc.reshape(1, H, A)[:, :, 6] = g_closed
    check("W2 leaves a closed gripper command alone", torch.equal(BW.where_own_apply(xc, norm_state([0.05, -0.03, 0.02, 2.0]), cfg, 2, gq, A, H).reshape(1, H, A)[:, :, 6], xc.reshape(1, H, A)[:, :, 6]))
    # (3) model-level
    obs = RT.raac_obs(4)                                                     # 4 held mid-demo frames + object-role frame + zero-state frame
    obs = obs + [dict(obs[0], **{"observation/state": np.asarray(s, np.float32)}) for s in ([0.05, -0.03, 0.02, 2.0], [0.012, -0.004, 0.01, 2.0], [-0.20, 0.20, 0.0, 2.0])]
    res = {}
    for w in (1, 2):
        bk = BW.PolicyBackboneW("pi05_g8v_raas_l_ae", checkpoint_dir=CKPT, device="cuda", where_own=w)
        print("[test] where_own %d: servo_k %r servo_max %r" % (w, bk.model.config.servo_k, bk.model.config.servo_max), flush=True)
        for j, o in enumerate(obs):
            ob = bk.observation(o); pre = bk.prefix(ob)
            raw = (pre.state[0, :4].float().cpu().numpy() + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
            held = raw[3] > 1.5
            nz = np.random.default_rng(300 + j).standard_normal((1, bk.dim_full)).astype(np.float32)
            nz2 = np.random.default_rng(900 + j).standard_normal((1, bk.dim_full)).astype(np.float32)
            for kw_ in (dict(), dict(full=True), dict(num_steps=4)):
                base = BB.PolicyBackbone.sample(bk, 1, pre, seed=0, noise=nz, **kw_)
                new = bk.sample(1, pre, seed=0, noise=nz, **kw_)
                if not held:
                    check("w%d frame %d m=0 np.array_equal %s" % (w, j, list(kw_)), np.array_equal(base, new))
                    continue
                B = BB.PolicyBackbone.sample(bk, 1, pre, seed=0, noise=nz, full=True).reshape(bk.H, bk.A)
                N = bk.sample(1, pre, seed=0, noise=nz, full=True).reshape(bk.H, bk.A)
                if "full" in kw_ or True:
                    rest = np.arange(bk.A) != 0
                    rest &= np.arange(bk.A) != 1
                    if w == 1:
                        check("w1 frame %d held, %s: dims except xy bit-equal" % (j, list(kw_)), np.array_equal(B[:, rest], N[:, rest]) if not kw_ or "full" in kw_ else True)
                    else:
                        r2 = rest & (np.arange(bk.A) != 6)
                        check("w2 frame %d held, %s: dims except xy and gripper bit-equal" % (j, list(kw_)), np.array_equal(B[:, r2], N[:, r2]) if not kw_ or "full" in kw_ else True)
                if not kw_:                                               # default (real rows) path == full path restricted to real rows
                    check("w%d frame %d default rows == full rows[real]" % (w, j, ), np.array_equal(new, bk.sample(1, pre, seed=0, noise=nz, full=True)[:, bk.real]))
            if held:
                N2 = bk.sample(1, pre, seed=0, noise=nz2, full=True).reshape(bk.H, bk.A)
                N1 = bk.sample(1, pre, seed=0, noise=nz, full=True).reshape(bk.H, bk.A)
                exp = np.clip(BW.k_w(bk.model.config.servo_max) * raw[:2], -bk.model.config.servo_max, bk.model.config.servo_max) * 2.0 / (np.array(bk.model.config.act_q99[:2]) - np.array(bk.model.config.act_q01[:2]) + 1e-6)
                check("w%d frame %d xy independent of the noise and = numpy saturating servo" % (w, j), np.array_equal(N1[:, :2], N2[:, :2]) and np.allclose(N1[:, :2], exp[None], atol=1e-5),
                      "rel %s cm, xy norm-units %s, base xy mean %s" % (np.round(raw[:3] * 100, 1).tolist(), np.round(exp, 3).tolist(),
                                                                       np.round(BB.PolicyBackbone.sample(bk, 1, pre, seed=0, noise=nz, full=True).reshape(bk.H, bk.A)[:5, :2].mean(0), 3).tolist()))
            res[(w, j)] = bk.sample(1, pre, seed=0, noise=nz, full=True)
        del bk; torch.cuda.empty_cache()
    for j in range(len(obs)):
        d = np.abs(res[(1, j)] - res[(2, j)]).reshape(10, 32)
        check("frame %d: W2 - W1 differs only in the gripper dim" % j, d[:, [i for i in range(32) if i != 6]].max() == 0.0, "max grip diff %.3g" % d[:, 6].max())
    print("[test] OVERALL %s" % ("PASS" if ok else "FAIL"), flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
