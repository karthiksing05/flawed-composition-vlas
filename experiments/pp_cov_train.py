"""Coverage arm (docs/COVERAGE_RESULTS.md section 1.5; PLAN_PARTS 10.11): P_min's build with W's GENERATED rows added.  Labelled ablation.

pp_p_train.py is not edited.  This script replaces the module function pp_p_train.rows with a wrapper that appends the generated rows
(DATA/gen/pp_cov_<task>.npz, K1-K6) to the human demo rows BEFORE every derived field is computed, then runs pp_p_train.main
unchanged.  pp_p_train's own task filter (--exclude) drops a held-out task's generated rows with its human rows.  Every fitted
quantity of the build (psupport radii, start radii, role radius, grasp-mode inventory) is fitted on all of the build's rows.  The
input centre stays the human demos' (P_min's).  Creation is NOT run here: pp_p_train.py create, unchanged, on the cov builds, fits
the created part from its k human demos only.

    python experiments/pp_cov_train.py --gen K1+K2+K3+K4+K5+K6 build --tag cov_full --rank 32 --psupport --tscale --decouple \
        --imode --rtransport --bbnom --bbnosize --seed 10
    python experiments/pp_cov_train.py --gen none build ...          (zero generated rows: P_min's build)
    python experiments/pp_cov_train.py smoke                          (batch checks, CPU)
    python experiments/pp_cov_train.py cmp A.npz B.npz                (bitwise checkpoint comparison)
"""
import hashlib
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
import fa1_common as FC                                                     # noqa: E402
import pp_p_train as PT                                                     # noqa: E402

OUT = os.path.join(ROOT, "results", "flawed")
PP = ["K1", "K2", "K3", "K4", "K5", "K6"]
DEMOS, DEMOS_B = os.path.join(FC.DATA, "demos.npz"), os.path.join(FC.DATA, "demos_b.npz")
ROWCOLS = ("task", "demo", "step", "action", "ee_pos", "ee_rot", "grip", "pos", "yaw", "drawer", "prev_grip")
_orig_rows = PT.rows
INFO = {}


class _NP:
    """numpy, except that load() of the two demo files returns the merged (human + generated) arrays."""

    def __init__(self, over):
        self._o = over

    def __getattr__(self, k):
        return getattr(np, k)

    def load(self, p, *a, **kw):
        return self._o[p] if p in self._o else np.load(p, *a, **kw)


def merged(gen):
    """-> ({path: dict}) overrides for demos.npz / demos_b.npz with the generated rows of tasks `gen` appended; {} if none."""
    if not gen:
        return {}
    D, Db = dict(np.load(DEMOS)), dict(np.load(DEMOS_B))
    G = [dict(np.load(os.path.join(FC.DATA, "gen", f"pp_cov_{t}.npz"))) for t in gen]
    for t, g in zip(gen, G):
        assert (g["task"] == list(FC.KNOWN).index(t)).all() and (g["demo"] >= 100).all() and (g["gen"] == 1).all(), t
    # generated rows go FIRST: fa1_train.chunks clips its look-ahead at the file's last row, so the file's last demo (human K8 demo
    # 49 in P_min) pads with its last action; with the generated rows in front, every human row's fields stay bitwise P_min's
    D2 = {k: v for k, v in D.items() if k not in ROWCOLS}
    for k in ROWCOLS:
        D2[k] = np.concatenate([g[k].astype(D[k].dtype) for g in G] + [D[k]])
    Db2 = {k: np.concatenate([g[k].astype(Db[k].dtype) for g in G] + [Db[k]]) for k in ("held", "off")}
    return {DEMOS: D2, DEMOS_B: Db2}


def rows_cov(dev, tasks=None, k=None, gen=None, **kw):
    """pp_p_train.rows over the human rows plus the generated rows of `gen` (default: INFO['gen']); centre from the human demos.
    Other keywords (pgate, ladder L1; off in this arm) pass through to pp_p_train.rows unchanged."""
    assert k is None, "creation uses pp_p_train.rows unchanged (k human demos only)"
    assert not kw.get("pgate"), "the coverage arm is P_min's build: --pgate stays off"
    gen = INFO.get("gen", []) if gen is None else gen
    center_h = np.load(DEMOS)["ee_pos"].mean(0).tolist()
    PT.np = _NP(merged(gen))
    try:
        b, X, center, size, n, demos = _orig_rows(dev, tasks, k, **kw)
    finally:
        PT.np = np
    g = (b["dkey"].cpu().numpy() % 1000) >= 100
    INFO.update(n_rows=int(n), n_generated=int(g.sum()), n_human=int((~g).sum()), generated_share=float(g.mean()),
                generated_episodes=int(len(np.unique(b["dkey"].cpu().numpy()[g]))), center_merged=center)
    return b, X, center_h, size, n, demos


def smoke():
    """COVERAGE_RESULTS 1.7 checks 1 (batch part) and 3 (CPU)."""
    dev = torch.device("cpu")
    res = {}
    for tasks in (list(FC.KNOWN), [c for c in FC.KNOWN if c not in ("K4", "K5")], [c for c in FC.KNOWN if c != "K1"]):
        ref = _orig_rows(dev, tasks)
        z = rows_cov(dev, tasks, gen=[])
        same = all(torch.equal(ref[0][x], z[0][x]) for x in ref[0]) and torch.equal(ref[1], z[1]) and ref[2] == z[2] and ref[4] == z[4]
        g = rows_cov(dev, tasks, gen=PP)
        nh = ref[4]
        differ = [x for x in ref[0] if not torch.equal(ref[0][x], g[0][x][-nh:])] + ([] if torch.equal(ref[1], g[1][-nh:]) else ["chunks"])
        human_prefix = not differ
        key = g[0]["dkey"].numpy()
        starts = key[np.r_[True, key[1:] != key[:-1]]]
        tix = {int(x) for x in np.unique(key[(key % 1000) >= 100] // 1000)}
        res["+".join(tasks)] = dict(zero_gen_bitwise=bool(same), human_rows_unchanged=bool(human_prefix), human_fields_differ=differ,
                                    episodes_contiguous_unique=bool(len(starts) == len(set(starts.tolist()))),
                                    held_01=bool(set(torch.unique(g[0]["held"]).tolist()) <= {0.0, 1.0}),
                                    generated_tasks=sorted(f"K{t + 1}" for t in tix), rows=int(g[4]), generated=INFO["n_generated"],
                                    generated_share=INFO["generated_share"], center_unchanged=bool(g[2] == ref[2]))
        print("[pp_cov smoke]", json.dumps(res["+".join(tasks)]), flush=True)
    ok = all(v["zero_gen_bitwise"] and v["human_rows_unchanged"] and v["episodes_contiguous_unique"] and v["held_01"] and v["center_unchanged"]
             for v in res.values())
    print("[pp_cov smoke] ALL", ok, flush=True)


def cmp(a, b, prefix=""):
    """Bitwise comparison of two checkpoints' arrays (optionally only names starting with `prefix`, e.g. 'obj.' or 'tgt.')."""
    A, B = np.load(a, allow_pickle=False), np.load(b, allow_pickle=False)
    ka, kb = sorted(k for k in A.files if k != "__meta__"), sorted(k for k in B.files if k != "__meta__")
    assert ka == kb, (set(ka) ^ set(kb))
    ka = [k for k in ka if k.startswith(prefix)]
    diff = [k for k in ka if not (A[k].shape == B[k].shape and np.array_equal(A[k], B[k]))]
    print(f"[pp_cov cmp] {os.path.basename(os.path.dirname(a))} vs {os.path.basename(os.path.dirname(b))} prefix '{prefix}': "
          f"{len(ka)} arrays, {len(diff)} differ: {diff}", flush=True)
    print("[pp_cov cmp] BITWISE", not diff, flush=True)


def main():
    if sys.argv[1] == "smoke":
        return smoke()
    if sys.argv[1] == "cmp":
        return cmp(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "")
    assert sys.argv[1] == "--gen", "usage: --gen K1+...|none build ..."
    gen = [] if sys.argv[2] == "none" else sys.argv[2].split("+")
    assert set(gen) <= set(PP)
    rest = sys.argv[3:]
    assert rest[0] == "build" and "--reuse" not in rest, "builds only, both roles trained"
    INFO["gen"] = gen
    PT.rows = rows_cov
    sys.argv = [os.path.join(HERE, "pp_p_train.py")] + rest
    PT.main()
    tag, seed = rest[rest.index("--tag") + 1], int(rest[rest.index("--seed") + 1]) if "--seed" in rest else 0
    rep = dict(tag=tag, seed=seed, gen_tasks=gen, args=rest, generated=bool(gen),
               gen_sha256={t: hashlib.sha256(open(os.path.join(FC.DATA, "gen", f"pp_cov_{t}.npz"), "rb").read()).hexdigest() for t in gen},
               **{k: v for k, v in INFO.items() if k != "gen"})
    with open(os.path.join(OUT, f"pp_cov_train_{tag}_s{seed}.json"), "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1)
    print("[pp_cov train]", json.dumps({k: v for k, v in rep.items() if k not in ("gen_sha256", "center_merged")}), flush=True)


if __name__ == "__main__":
    main()
