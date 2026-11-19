"""SR 78b serving-fix test (sbatch only; one GPU; openpi venv).  PolicyBackbone.sample now adds the fixed servo of a config with a nonzero servo_k.
(1) BIT-IDENTITY: for configs WITHOUT servo_k (frozen pi0.5 pi05_libero, plain g8v, RAA-c) the pre-change sample (git HEAD copy, BEFORE) and the patched one give
    torch.equal-identical rows on fixed noise (default, full=True, 4 steps).
(2) RAA-s: patched backbone.sample == PI0PytorchS.sample_actions on the same noise (the model's own path, the one raas_check check 4 tested), BEFORE = patched - servo,
    and the end-to-end GuidedPolicy.infer chunk equals unnormalize(sample_actions).
(3) RAA-c state_proj: the served path's v_full uses the state token: patched sample on raac == PI0PytorchC.sample_actions on the same noise, and zeroing the prefix state changes the sample.
    python experiments/raas_b_test.py --before PATH_TO_HEAD_COPY
"""
import argparse, dataclasses, importlib.util, io, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = ["/coc/flash10/ksingara3/robot/openpi/src", ROOT]
LR = "/coc/flash10/ksingara3/robot/data/hf/lerobot/lol/libero_g8v_raac"
CK = "/coc/flash10/ksingara3/robot/checkpoints_l90c/pytorch/"


def load_before(path):
    spec = importlib.util.spec_from_file_location("lifelong_ood.policy._bk_before", path)
    m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m; spec.loader.exec_module(m); return m


def raac_obs(n=4):
    import pyarrow.parquet as pq
    from PIL import Image
    tasks = {}
    for l in open(os.path.join(LR, "meta", "tasks.jsonl"), encoding="utf-8"):
        d = json.loads(l); tasks[d["task_index"]] = d["task"]
    out = []
    for ep in (0, 40, 80, 120):
        t = pq.read_table(os.path.join(LR, "data", "chunk-000", "episode_%06d.parquet" % ep)).to_pydict()
        st = np.asarray(t["state"], np.float32)
        idx = np.where(st[:, 3] == 2.0)[0]
        i = int(idx[len(idx) // 2])
        dec = lambda c: np.asarray(Image.open(io.BytesIO(t[c][i]["bytes"])).convert("RGB"))
        out.append({"observation/image": dec("image"), "observation/wrist_image": dec("wrist_image"), "observation/state": st[i], "prompt": tasks[t["task_index"][i]]})
    # plus one object-role frame and one zero-state frame (servo gate m = 0)
    o2 = dict(out[0]); s = np.asarray(o2["observation/state"]).copy(); s[3] = 1.0; o2["observation/state"] = s
    o3 = dict(out[0]); o3["observation/state"] = np.zeros(4, np.float32)
    return out[:n] + [o2, o3]


def libero_obs(k=3):
    rng = np.random.default_rng(0)
    return [{"observation/state": rng.random(8).astype(np.float32), "observation/image": rng.integers(256, size=(224, 224, 3), dtype=np.uint8),
             "observation/wrist_image": rng.integers(256, size=(224, 224, 3), dtype=np.uint8), "prompt": "pick up the black bowl and place it on the plate"} for _ in range(k)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--before", required=True)
    a = ap.parse_args()
    import torch
    from lifelong_ood.policy import backbone as NEW
    OLD = load_before(a.before)
    ok = True
    # old = pre-change function bound to the same instance (plain duck typing; no second 3B model in memory)
    for cfgname, ck, obs_fn in (("pi05_libero", None, libero_obs), ("pi05_g8v_plain_ae", CK + "pi05_g8v_plain_ae_tl6000", libero_obs),
                                ("pi05_g8v_raac_ae", CK + "pi05_g8v_raac_ae_tl6000", raac_obs)):
        bk = NEW.PolicyBackbone(cfgname, checkpoint_dir=ck, device="cuda")
        print("[test] %s: servo_k attr %r" % (cfgname, getattr(bk.model.config, "servo_k", None)), flush=True)
        worst = 0.0; same = True; n = 0
        for j, o in enumerate(obs_fn()):
            ob = bk.observation(o); pre = bk.prefix(ob)
            nz = np.random.default_rng(100 + j).standard_normal((1, bk.dim_full)).astype(np.float32)
            for kw in (dict(), dict(full=True), dict(num_steps=4)):
                new = NEW.PolicyBackbone.sample(bk, 1, pre, seed=0, noise=nz, **kw)
                old = OLD.PolicyBackbone.sample(bk, 1, pre, seed=0, noise=nz, **kw)
                same &= bool(np.array_equal(new, old)); worst = max(worst, float(np.abs(new - old).max())); n += 1
        print("[test] BIT-IDENTITY %s: %d samples, np.array_equal all %s, max|diff| %.3g" % (cfgname, n, same, worst), flush=True)
        ok &= same
        if cfgname == "pi05_g8v_raac_ae":
            # RAA-c state_proj on the served path
            worst = 0.0; sens = []
            for j, o in enumerate(obs_fn()[:4]):
                ob = bk.observation(o); pre = bk.prefix(ob)
                nz = np.random.default_rng(100 + j).standard_normal((1, bk.dim_full)).astype(np.float32)
                srv = NEW.PolicyBackbone.sample(bk, 1, pre, seed=0, noise=nz, full=True).reshape(bk.H, bk.A)
                with torch.no_grad():
                    ref = bk.model.sample_actions(bk.device, ob, noise=torch.as_tensor(nz, device=bk.device).reshape(1, bk.H, bk.A), num_steps=10)[0].float().cpu().numpy()
                worst = max(worst, float(np.abs(srv - ref).max()))
                z = dataclasses.replace(pre, state=torch.zeros_like(pre.state))
                zs = NEW.PolicyBackbone.sample(bk, 1, z, seed=0, noise=nz, full=True).reshape(bk.H, bk.A)
                sens.append(float(np.abs(srv - zs)[:, :7].max()))
            print("[test] RAA-c served path (backbone.sample) vs PI0PytorchC.sample_actions (uses state_proj via embed_suffix): max|diff| %.3g over 4 frames;"
                  " zeroing the prefix state changes the served chunk by max %s (so state_proj IS read on the served path)" % (worst, np.round(sens, 3).tolist()), flush=True)
            ok &= worst < 5e-3 and min(sens) > 1e-3
        del bk; torch.cuda.empty_cache()
    # RAA-s
    from lifelong_ood.policy import raas_model as RS
    from lifelong_ood.policy.serve import GuidedPolicy
    bk = NEW.PolicyBackbone("pi05_g8v_raas_ae", checkpoint_dir=CK + "pi05_g8v_raas_ae_tl6000", device="cuda")
    print("[test] raas servo_k %r servo_max %r" % (bk.model.config.servo_k, bk.model.config.servo_max), flush=True)
    pol = GuidedPolicy(bk, hooks=None, num_steps=10, seed=13)
    w1 = w2 = w3 = 0.0; mins = 1e9; gate = []
    for j, o in enumerate(raac_obs()):
        ob = bk.observation(o); pre = bk.prefix(ob)
        nz = np.random.default_rng(200 + j).standard_normal((1, bk.dim_full)).astype(np.float32)
        new = NEW.PolicyBackbone.sample(bk, 1, pre, seed=0, noise=nz, full=True).reshape(bk.H, bk.A)
        old = OLD.PolicyBackbone.sample(bk, 1, pre, seed=0, noise=nz, full=True).reshape(bk.H, bk.A)
        sv = RS.servo_norm_torch(bk.model.config, pre.state, bk.A)[0, 0].cpu().numpy()
        with torch.no_grad():
            ref = bk.model.sample_actions(bk.device, ob, noise=torch.as_tensor(nz, device=bk.device).reshape(1, bk.H, bk.A), num_steps=10)[0].float().cpu().numpy()
        w1 = max(w1, float(np.abs(new - ref).max())); w2 = max(w2, float(np.abs(new - (old + sv[None, :])).max()))
        gate.append(float(np.abs(sv).max()))
        if np.abs(sv).max() > 0: mins = min(mins, float(np.abs(new - old)[:, :3].max()))
        # end to end: infer == unnormalize(model's own sample_actions real rows)
        want = bk.unnormalize(ref[:, :7].reshape(1, -1), state=ob.state[0].float().cpu().numpy())[0]
        got = pol.infer(o, noise=nz)["actions"]
        w3 = max(w3, float(np.abs(np.asarray(got) - want).max()))
    print("[test] RAA-s: patched backbone.sample vs PI0PytorchS.sample_actions (same noise) max|diff| %.3g; patched vs (old + servo) %.3g; |servo| per frame (norm units) %s;"
          " min patched-old change on held frames %.3g" % (w1, w2, np.round(gate, 3).tolist(), mins), flush=True)
    print("[test] RAA-s end-to-end GuidedPolicy.infer (noise passed) vs unnormalize(PI0PytorchS.sample_actions): max|diff| %.3g (action units)" % w3, flush=True)
    ok &= w1 < 5e-3 and w3 < 5e-2
    print("[test] OVERALL %s" % ("PASS" if ok else "FAIL"), flush=True)


if __name__ == "__main__":
    main()
