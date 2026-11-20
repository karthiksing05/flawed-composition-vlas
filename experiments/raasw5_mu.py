"""SR 87 (docs/notes/raasw5_plan.md): the release point mu per target class, computed ONCE from the TRAINING demos only and committed before any W5 rollout is read.
Training demos = the episodes of lol/libero_g8v_raac (the RAA-sL checkpoint's dataset, tasks "object: ..." / "target: ..." role phrases).  For each target phrase ("target: the plate" ..): over the
episodes' held frames (state[3] == 2.0, the role id) the release frame r = the first frame whose gripper command < 0 (open; the last held frame if none); mu = the median over episodes of the channel
rel xy (state[0:2], target minus object, metres, base-frame axes) at r, per axis.  The same quantity as sr86_q1d.py's table; no evaluation state, no rollout is read.
    python experiments/raasw5_mu.py        -> lifelong_ood/policy/raasw5_mu.json (refuses to overwrite)
"""
import collections, json, os, sys
import numpy as np
import pyarrow.parquet as pq
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LR = "/coc/flash10/ksingara3/robot/data/hf/lerobot/lol/libero_g8v_raac"
OUT = os.path.join(ROOT, "lifelong_ood", "policy", "raasw5_mu.json")


def main():
    if os.path.exists(OUT):
        sys.exit("refusing to overwrite " + OUT)
    eps = [json.loads(l) for l in open(os.path.join(LR, "meta", "episodes.jsonl"), encoding="utf-8")]
    cls = collections.defaultdict(list)
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
        cls[tg[0]].append(st[h[r], :2].tolist())
    mu = {k: [float(np.median([v[0] for v in vs])), float(np.median([v[1] for v in vs]))] for k, vs in sorted(cls.items())}
    n = {k: len(vs) for k, vs in sorted(cls.items())}
    res = dict(mu_m=mu, n_demos=n, skipped_episodes=skipped, dataset=LR, definition="median over training demos of the channel rel xy (target minus object, m) at the first open command while the role id = target; "
               "key = the role phrase the channel carries (obs['prompt'] while held); unknown phrase -> 0", written_by="experiments/raasw5_mu.py")
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
