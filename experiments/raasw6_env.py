"""SR 90 (docs/notes/raasw6_plan.md): the release ENVELOPE per target class and the target-latch tolerance, computed ONCE from the TRAINING demos only and committed
(lifelong_ood/policy/raasw6_env.json) before any W6 rollout.  Same pass as experiments/raasw5_mu.py (lol/libero_g8v_raac, 300 held episodes; no evaluation state, no rollout read).
For each target phrase ("target: the plate" ..): over the episode's held frames (state[3] == 2.0) the release frame r = the first frame whose gripper command < 0 (open; the last held frame if none);
  mu            = per-axis median over episodes of the channel rel xy at r (asserted equal to raasw5_mu.json);
  c_rel         = -state[2] at r (channel height of the held object above the target's reference, m);  c_med = median, c_dev = p90 of |c_rel - c_med|;
  h_class       = c_med + c_dev   ("demo median release height + demo p90 spread");
  r_class       = p90 over episodes of |rel_xy(r) - mu|   ("demo p90 xy scatter around mu").
Latch tolerance tau = p99 of the frame-to-frame change of the TARGET's pose in the training demos: per K1-K6 demo (data/fa1/rv/<K>/chan_*.npz: the simulator's slot positions at every rendered frame),
|pos[t + 5, target slot] - pos[t, target slot]| (3-D norm, metres), stride 5 = the serving replan interval (the module sees one measurement per 5 steps), pooled over all demos / frames / tasks.
    python experiments/raasw6_env.py        -> lifelong_ood/policy/raasw6_env.json (refuses to overwrite)
"""
import collections, glob, json, os, sys
import numpy as np
import pyarrow.parquet as pq
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LR = "/coc/flash10/ksingara3/robot/data/hf/lerobot/lol/libero_g8v_raac"
RV = "/coc/flash10/ksingara3/robot/data/fa1/rv"
OUT = os.path.join(ROOT, "lifelong_ood", "policy", "raasw6_env.json")
MU5 = os.path.join(ROOT, "lifelong_ood", "policy", "raasw5_mu.json")
STRIDE = 5


def main():
    if os.path.exists(OUT):
        sys.exit("refusing to overwrite " + OUT)
    eps = [json.loads(l) for l in open(os.path.join(LR, "meta", "episodes.jsonl"), encoding="utf-8")]
    rel, ch = collections.defaultdict(list), collections.defaultdict(list)
    skipped = 0
    for e in eps:
        tg = [t for t in e["tasks"] if t.startswith("target:")]
        if not tg:
            continue
        f = os.path.join(LR, "data", "chunk-000", "episode_%06d.parquet" % e["episode_index"])
        if not os.path.exists(f):
            skipped += 1
            continue
        t = pq.read_table(f).to_pydict()
        st, ac = np.asarray(t["state"], float), np.asarray(t["actions"], float)
        h = np.where(np.isclose(st[:, 3], 2.0))[0]
        if len(h) < 5:
            skipped += 1
            continue
        op = np.where(ac[h, 6] < 0)[0]
        r = int(op[0]) if len(op) else len(h) - 1
        rel[tg[0]].append(st[h[r], :2].tolist())
        ch[tg[0]].append(float(-st[h[r], 2]))
    mu5 = json.load(open(MU5, encoding="utf-8"))["mu_m"]
    mu, cmed, cdev, hcl, rcl, n = {}, {}, {}, {}, {}, {}
    for k in sorted(rel):
        a = np.asarray(rel[k], float)
        mu[k] = [float(np.median(a[:, 0])), float(np.median(a[:, 1]))]
        assert np.allclose(mu[k], mu5[k], atol=1e-12, rtol=0), ("mu differs from raasw5_mu.json", k, mu[k], mu5[k])
        c = np.asarray(ch[k], float)
        cmed[k] = float(np.median(c))
        cdev[k] = float(np.percentile(np.abs(c - cmed[k]), 90))
        hcl[k] = cmed[k] + cdev[k]
        rcl[k] = float(np.percentile(np.linalg.norm(a - np.asarray(mu[k]), axis=1), 90))
        n[k] = int(len(a))
    # latch tolerance: the target slot's frame-to-frame change in the training demos
    nt = {t["compound"]: t["target"] for t in json.load(open(os.path.join(ROOT, "results", "flawed", "n2_tasks.json"), encoding="utf-8"))["tasks"]}
    d, per = [], {}
    for c in ["K%d" % i for i in range(1, 7)]:
        for f in sorted(glob.glob(os.path.join(RV, c, "chan_*.npz"))):
            z = np.load(f, allow_pickle=True)
            ent = [str(x) for x in z["ent"]]
            ti = ent.index(nt[c])
            p = np.asarray(z["pos"], float)[:, ti]
            if len(p) > STRIDE:
                v = np.linalg.norm(p[STRIDE:] - p[:-STRIDE], axis=1)
                d.append(v)
                per.setdefault(c, []).append(float(v.max()))
    d = np.concatenate(d)
    tau = float(np.percentile(d, 99))
    res = dict(mu_m=mu, c_med_m=cmed, c_dev_p90_m=cdev, h_class_m=hcl, r_class_m=rcl, n_demos=n, skipped_episodes=skipped,
               tau_target_m=tau, tau_stats=dict(n=int(len(d)), p50=float(np.percentile(d, 50)), p90=float(np.percentile(d, 90)), p99=tau, max=float(d.max()), stride=STRIDE,
                                                max_per_task={k: float(max(v)) for k, v in per.items()}, demos=int(sum(len(v) for v in per.values()))),
               dataset=LR, chan_root=RV, written_by="experiments/raasw6_env.py",
               definition="h_class = median(c_rel) + p90|c_rel - median|, c_rel = -dz of the channel at the first open command while the role id = target; r_class = p90 |rel_xy - mu| at the same frame; "
                          "tau = p99 of the 5-step change of the target slot position over the K1-K6 training demos; unknown phrase -> W4 behaviour")
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
