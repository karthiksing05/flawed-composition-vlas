"""SR 89 step 2 (docs/notes/scene2_plan.md): the release point mu per plate class, computed ONCE from the 100 scene-2 TRAINING cycles only (episodes >= 400 of
lol/libero_g8v_raacs2, the RAA-sL-S2 training repo; target phrases "target: the left plate" / "target: the right plate"), before any SR 89 step-2 rollout.  The definition is
raasw5_mu.py's (SR 87): over each cycle's held frames (state[3] == 2.0) the release frame r = the first frame whose gripper command < 0 (open; the last held frame if none); mu = the median over
cycles of the channel rel xy (state[0:2], target minus object, m, base-frame axes) at r, per axis.  No evaluation state, no rollout is read.
    python experiments/raasls2_mu.py        -> lifelong_ood/policy/raasls2_mu.json (refuses to overwrite)
"""
import collections, json, os, sys
import numpy as np
import pyarrow.parquet as pq
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LR = "/coc/flash10/ksingara3/robot/data/hf/lerobot/lol/libero_g8v_raacs2"
OUT = os.path.join(ROOT, "lifelong_ood", "policy", "raasls2_mu.json")
FIRST_S2 = 400


def main():
    if os.path.exists(OUT):
        sys.exit("refusing to overwrite " + OUT)
    eps = [json.loads(l) for l in open(os.path.join(LR, "meta", "episodes.jsonl"), encoding="utf-8")]
    cls = collections.defaultdict(list)
    skipped = 0
    for e in eps:
        if e["episode_index"] < FIRST_S2:
            continue
        tg = [t for t in e["tasks"] if t.startswith("target:")]
        assert tg and tg[0] in ("target: the left plate", "target: the right plate"), e
        f = os.path.join(LR, "data", "chunk-000", "episode_%06d.parquet" % e["episode_index"])
        t = pq.read_table(f).to_pydict()
        st, ac = np.asarray(t["state"], float), np.asarray(t["actions"], float)
        h = np.where(np.isclose(st[:, 3], 2.0))[0]
        if len(h) < 5:
            skipped += 1
            continue
        op = np.where(ac[h, 6] < 0)[0]
        r = int(op[0]) if len(op) else len(h) - 1
        cls[tg[0]].append(st[h[r], :2].tolist())
    mu = {k: [float(np.median([v[0] for v in vs])), float(np.median([v[1] for v in vs]))] for k, vs in sorted(cls.items())}
    sd = {k: [float(np.std([v[0] for v in vs])), float(np.std([v[1] for v in vs]))] for k, vs in sorted(cls.items())}
    n = {k: len(vs) for k, vs in sorted(cls.items())}
    res = dict(mu_m=mu, sd_m=sd, n_demos=n, skipped_episodes=skipped, dataset=LR, definition="median over the 100 scene-2 training cycles of the channel rel xy (target minus object, m) at the first open command while the role id = target; "
               "key = the role phrase the channel carries (obs['prompt'] while held)", written_by="experiments/raasls2_mu.py")
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
