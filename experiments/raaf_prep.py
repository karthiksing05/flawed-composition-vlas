"""RAA-F Step 0 (docs/notes/raaf_plan.md 2.1-2.2, 5): per K1-K6 rendered demo, the quantities the relabel rule needs.  openpi venv, CPU (sbatch only).

Per demo: the hold rows H (sc_t16.held_rows, the convert's own rule), the hold onset h0, the release r (first row at or after h0 whose gripper
command is open, actions[:, 6] < 0), the end e of the d' episode (min(r, first non-held row after h0): d' = rows h0 .. e - 1), the demo's own
target-role channel rel_t (sc_rv.demo_rel + held_target_rel: the exact numbers convert writes into the state token), the grasped / release slot
names and indices, the actions, the eef and every slot position of the last frame.  Nothing is rendered; the images are not read.

    python experiments/raaf_prep.py --out /coc/flash10/ksingara3/.claude_tmp/raaf/step0/prep.pkl

Writes only a new file (refuses to overwrite).
"""
import argparse
import json
import os
import pickle
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
import sc_rv as RV                                                           # noqa: E402

K = 2.851209548740752
M = 0.9375


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a0 = ap.parse_args()
    assert not os.path.exists(a0.out), a0.out
    os.makedirs(os.path.dirname(a0.out), exist_ok=True)
    ns = types.SimpleNamespace(suffix="", root=RV.RV_ROOT, arm="raac", ndemo=0)
    T16 = RV.t16()
    X = RV.lex(ns)
    by_c, _ = RV.roles_of()
    import fa1_common as FC
    info = dict(demos=0, fallback=[], g_ne_given=0, r_ne_given=0)
    demos, stats = [], dict(no_release=[], held_gap=[], no_hold=[])
    for c in ("K1", "K2", "K3", "K4", "K5", "K6"):
        for f in RV.demo_files(ns.root, c):
            zz = np.load(f)
            z = {k: zz[k] for k in ("state", "actions", "language", "task", "demo", "kept")}
            if T16.given(str(z["language"])) is None:
                continue
            rel = RV.demo_rel(z, c, f, ns, X, T16, by_c, info)              # [n, 2, 3]; arm raac: [:, 1] = held_target_rel
            H = np.asarray(T16.held_rows(z["state"], z["actions"], X["tw"], X["lift"]), bool)
            act = np.asarray(z["actions"], np.float64)
            n = len(act)
            ch = np.load(RV.chan_path(ns.root, c, int(os.path.basename(f)[5:-4])))
            pos = np.asarray(ch["pos"], np.float64)
            ee = np.asarray(ch["ee"], np.float64)
            tag = "%s:%s" % (c, os.path.basename(f))
            if not H.any():
                stats["no_hold"].append(tag)
                continue
            h0 = int(np.nonzero(H)[0][0])
            op = np.nonzero(act[h0:, 6] < 0)[0]
            if len(op) == 0:
                stats["no_release"].append(tag)
                continue
            r = h0 + int(op[0])
            nh = np.nonzero(~H[h0:])[0]
            e_run = h0 + int(nh[0]) if len(nh) else n
            e = min(r, e_run)
            if e < r:
                stats["held_gap"].append(tag)
            # which slots g (grasped) and rr (release): the demo's own contact events, as demo_rel (g_ne_given = r_ne_given = 0 in the raac manifest)
            import pp_p_train as PP
            D = dict(task=np.zeros(n, np.int32), demo=np.zeros(n, np.int32), pos=pos)
            s = PP.demo_sites(D, H.astype(np.float64)).get((0, 0))
            g, rs = (s["g"], s["r"]) if s is not None else by_c[c]
            demos.append(dict(task=c, tid=int(c[1:]) - 1, demo=int(z["demo"]), file=f, n=n, H=H, h0=h0, r=r, e=e, g=g, rslot=rs,
                              gi=FC.ENT.index(g), ri=FC.ENT.index(rs), rel=rel[:, 1].astype(np.float64), actions=act, ee=ee,
                              pos_last=pos[-1], pos_h0=pos[h0], kept=np.asarray(z["kept"]), language=str(z["language"])))
    out = dict(demos=demos, stats=stats, chan_info={k: v for k, v in info.items() if k != "offset_cm"}, lex=dict(tw=X["tw"], lift=X["lift"]), K=K, M=M)
    tmp = a0.out + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(out, fh)
    os.rename(tmp, a0.out)
    per = {}
    for d in demos:
        per.setdefault(d["task"], []).append(d)
    print("[raaf_prep] demos %d; no_hold %d no_release %d held_gap %d" % (len(demos), len(stats["no_hold"]), len(stats["no_release"]), len(stats["held_gap"])))
    for c, v in sorted(per.items()):
        nh = sum(int(d["H"].sum()) for d in v)
        nd = sum(d["e"] - d["h0"] for d in v)
        print("[raaf_prep] %s: demos %d, held rows %d, rows h0..e-1 %d, mean h0 %.1f mean e-h0 %.1f" % (
            c, len(v), nh, nd, np.mean([d["h0"] for d in v]), np.mean([d["e"] - d["h0"] for d in v])))
    print("[raaf_prep] chan_info %s" % json.dumps(out["chan_info"]))
    print("[raaf_prep] held_gap %s" % stats["held_gap"][:10])


if __name__ == "__main__":
    main()
