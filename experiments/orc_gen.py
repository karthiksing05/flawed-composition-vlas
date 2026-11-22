"""ORC data generator (docs/notes/plan_oracle_pi0.md 2.6): planner table-spot demos with a PAIRED spot.  LIBERO venv, CPU.  sbatch only.

Imports experiments/raafp_gen.py unchanged and monkeypatches its module-level `spot`: the patched spot draws from its own stream default_rng([seed, 7, s, r])
(s = spot index, r = retry), over the same free mask (rule h_j), and keeps only cells >= SEP (15 cm, xy) from every spot already kept for the layout (+ a 4 mm margin
for the sub-cell jitter, so the final distance is >= 15 cm; asserted).  It consumes r extra normals from the attempt's rng, so each retry gets a fresh jitter.
At most 1 + MAXR attempts per (layout, spot).

  pairs  for each of the 100 kept A+ demos (BASE 94,000,000, /coc/.../gen_aplus/<obj>/demo_<k>.npz) regenerate the SAME layout (seed reproduction: raafp_gen.attempt
         derives seed = BASE + 1000*oix + k + 50e6*j, so k = stored seed - BASE - 1000*oix, j = 0) and add one new spot.   Out: <out>/<obj>/demo_<k>.npz (+ pair_of, src_spot,
         init_maxdiff_vs_source), pairs_summary.json.
  test   fresh layouts, BASE 96,000,000: a layout is kept only when BOTH spots yield kept demos; until 20 complete pairs per object (cap 250 attempts per object).
         Out: <out>/<obj>/demo_<2p>.npz (spot 0), demo_<2p+1>.npz (spot 1) + pairs.json.

    python experiments/orc_gen.py pairs --out DIR
    python experiments/orc_gen.py test  --out DIR
"""
import argparse
import glob
import json
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
import fa1_common as FC                                                      # noqa: E402
import fa1g_plan as FP                                                       # noqa: E402
import raafp_gen as G                                                        # noqa: E402

RULE = "h_j"
SEP = 0.15
MARGIN = 0.004
MAXR = 6
REJ = ("layout_rejected", "no_spot", "no_layout")
APLUS = "/coc/flash10/ksingara3/.claude_tmp/raaf/gen_aplus"
CUR = {}


def spot_patched(n2, sc, oe, rule, rng):
    free, _, (lo, hi, step, xs, ys) = G.free_mask(n2, sc, oe, rule, step=0.005)
    ix = np.argwhere(free)
    if len(ix) == 0:
        return None
    P = np.stack([xs[ix[:, 0]], ys[ix[:, 1]]], 1)
    for q in CUR["kept"]:
        P = P[np.linalg.norm(P - np.asarray(q)[:2], axis=1) >= SEP + MARGIN]
    if len(P) == 0:
        return None
    srng = np.random.default_rng([CUR["seed"], 7, CUR["s"], CUR["r"]])
    rng.normal(size=CUR["r"])                                                # consume r normals of the attempt's rng (a fresh jitter per retry)
    i = int(srng.integers(len(P)))
    sp = P[i] + srng.uniform(-step / 2, step / 2, 2) * 0.99
    for q in CUR["kept"]:
        assert np.linalg.norm(sp - np.asarray(q)[:2]) >= SEP, (sp, q)
    return sp


G.spot = spot_patched


def setup():
    n2, _ = FP.lang_roles()
    kps = json.load(open(os.path.join(FP.OUT, "a1g_keypoints.json"), encoding="utf-8"))["tasks"]
    return n2, kps


def try_spot(n2, kps, bddl, lay_env, oe, tgt, task, oix, k, j, s, kept, tmp):
    """Up to 1 + MAXR attempts for one (layout, spot).  Returns (record of the kept demo or the last record, retries used)."""
    rec = None
    for r in range(MAXR + 1):
        CUR.update(seed=G.BASE + 1000 * oix + k + 50_000_000 * j, s=s, r=r, kept=list(kept))
        rec = G.attempt(n2, kps[task], bddl, lay_env, oe, tgt, k, oix, RULE, tmp, "obj", j)
        rec["retry"] = r
        if rec["success"] or rec["cls"] in REJ:
            return rec, r
    return rec, MAXR


def pairs(a):
    n2, kps = setup()
    G.BASE = 94_000_000
    summ = {}
    t0 = time.time()
    for oix, (name, oe, task) in enumerate(G.OBJ[:2]):
        t = next(x for x in n2["tasks"] if x["compound"] == task and x["layout"] == "familiar")
        bddl = os.path.join(FP.ROOT, t["bddl"])
        lay_env = FC.make_env(bddl)
        tgt = t["target"]
        os.makedirs(os.path.join(a.out, name), exist_ok=True)
        rows = []
        for f in sorted(glob.glob(os.path.join(APLUS, name, "demo_*.npz")), key=lambda p: int(os.path.basename(p)[5:-4])):
            srck = int(os.path.basename(f)[5:-4])
            fin = os.path.join(a.out, name, "demo_%d.npz" % srck)
            if os.path.exists(fin):
                continue
            z = np.load(f)
            seed = int(z["seed"])
            k = seed - G.BASE - 1000 * oix
            assert k >= 0 and G.BASE + 1000 * oix + k == seed
            tmp = os.path.join(a.out, "_tmp")
            rec, retries = try_spot(n2, kps, bddl, lay_env, oe, tgt, task, oix, k, 0, 1, [z["spot"]], tmp)
            row = dict(src=os.path.basename(f), seed=seed, k=k, ok=bool(rec["success"]), cls=rec["cls"], retries=retries)
            if rec["success"]:
                tf = os.path.join(tmp, "obj", "demo_%d.npz" % k)
                n = dict(np.load(tf))
                sep = float(np.linalg.norm(n["spot"] - z["spot"]))
                dif = float(np.abs(n["init_state"] - z["init_state"]).max())
                assert sep >= SEP, sep
                n.update(pair_of=os.path.basename(f), src_spot=z["spot"], init_maxdiff_vs_source=dif)
                np.savez_compressed(fin, **n)
                os.remove(tf)
                row.update(sep_m=sep, init_maxdiff_vs_source=dif, spot=n["spot"].tolist())
            rows.append(row)
            print("[orc_gen] pairs %s %s: %s retries %d (%.0fs)" % (name, row["src"], rec["cls"], retries, time.time() - t0), flush=True)
        lay_env.close()
        ok = [r for r in rows if r["ok"]]
        summ[name] = dict(task=task, attempted=len(rows), kept=len(ok), existing=len(glob.glob(os.path.join(a.out, name, "demo_*.npz"))), rows=rows,
                          max_init_diff=max((r["init_maxdiff_vs_source"] for r in ok), default=None),
                          min_sep_m=min((r["sep_m"] for r in ok), default=None))
        print("[orc_gen] pairs %s: kept %d of %d (files now %d)" % (name, len(ok), len(rows), summ[name]["existing"]), flush=True)
    shutil.rmtree(os.path.join(a.out, "_tmp"), ignore_errors=True)
    p = os.path.join(a.out, "pairs_summary_%d.json" % int(time.time()))
    with open(p, "x", encoding="utf-8") as fh:
        json.dump(dict(rule=RULE, sep=SEP, base=G.BASE, summary=summ), fh, indent=1)


def test(a):
    n2, kps = setup()
    G.BASE = 96_000_000
    pairs_all = []
    t0 = time.time()
    for oix, (name, oe, task) in enumerate(G.OBJ[:2]):
        t = next(x for x in n2["tasks"] if x["compound"] == task and x["layout"] == "familiar")
        bddl = os.path.join(FP.ROOT, t["bddl"])
        lay_env = FC.make_env(bddl)
        tgt = t["target"]
        os.makedirs(os.path.join(a.out, name), exist_ok=True)
        done_p, calls, k = 0, 0, 0
        while done_p < a.npairs and calls < a.cap:
            j, r0 = 0, None
            while True:
                tmp0 = os.path.join(a.out, "_tmp0")
                r0, rt0 = try_spot(n2, kps, bddl, lay_env, oe, tgt, task, oix, k, j, 0, [], tmp0)
                calls += 1 + rt0
                if r0["cls"] in REJ and j < 40 and calls < a.cap:
                    j += 1
                    continue
                break
            if r0["success"]:
                z0 = dict(np.load(os.path.join(tmp0, "obj", "demo_%d.npz" % k)))
                tmp1 = os.path.join(a.out, "_tmp1")
                r1, rt1 = try_spot(n2, kps, bddl, lay_env, oe, tgt, task, oix, k, j, 1, [z0["spot"]], tmp1)
                calls += 1 + rt1
                print("[orc_gen] test %s k %d j %d: spot0 ok, spot1 %s (calls %d, %.0fs)" % (name, k, j, r1["cls"], calls, time.time() - t0), flush=True)
                if r1["success"]:
                    z1 = dict(np.load(os.path.join(tmp1, "obj", "demo_%d.npz" % k)))
                    sep = float(np.linalg.norm(z0["spot"] - z1["spot"]))
                    assert sep >= SEP
                    f0, f1 = "demo_%d.npz" % (2 * done_p), "demo_%d.npz" % (2 * done_p + 1)
                    z0.update(pair_id=done_p, spot_idx=0)
                    z1.update(pair_id=done_p, spot_idx=1)
                    np.savez_compressed(os.path.join(a.out, name, f0), **z0)
                    np.savez_compressed(os.path.join(a.out, name, f1), **z1)
                    pairs_all.append(dict(obj=name, pair=done_p, k=k, j=j, seed=int(z0["seed"]), files=[f0, f1], spots=[z0["spot"].tolist(), z1["spot"].tolist()], sep_m=sep))
                    done_p += 1
            else:
                print("[orc_gen] test %s k %d j %d: spot0 %s (calls %d)" % (name, k, j, r0["cls"], calls), flush=True)
            k += 1
        lay_env.close()
        print("[orc_gen] test %s: %d complete pairs, %d attempts (%.0fs)" % (name, done_p, calls, time.time() - t0), flush=True)
    for d in ("_tmp0", "_tmp1"):
        shutil.rmtree(os.path.join(a.out, d), ignore_errors=True)
    with open(os.path.join(a.out, "pairs.json"), "x", encoding="utf-8") as fh:
        json.dump(dict(rule=RULE, sep=SEP, base=G.BASE, pairs=pairs_all), fh, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("pairs", "test"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--npairs", type=int, default=20)
    ap.add_argument("--cap", type=int, default=250)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pairs(a) if a.mode == "pairs" else test(a)
