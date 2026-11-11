"""Q2-N (SCAFFOLD_RESULTS 42): the prototype operator parses ONE demonstration of each NEW pairing (N1-N6, recorded by
sc_pu_record.py from extra state 24, familiar layout; robot/data/fa1/pu/rec_new) into a rung's role phrases, at that rung's picked
Q2 setting and at setting 0, with the flow-loss baseline on the same rows.  sc_q2.py's own parse_demo, unchanged; truth from the tid
(scoring only).  One A40.

    python experiments/sc_q2_demo.py --rung R --ckpt DIR        -> results/flawed/sc_q2/<R>/demo_new.json (refuses to overwrite)
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sc_q2 as Q2                                                          # noqa: E402

REC = "/coc/flash10/ksingara3/robot/data/fa1/pu/rec_new"
DEMOS = [("N%d" % k, 16 + 2 * (k - 1)) for k in range(1, 7)]                # (compound, familiar tid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", required=True, choices=list(Q2.RUNGS))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="")
    ap.add_argument("--cands", default="")
    a = ap.parse_args()
    fn = os.path.join(Q2.od(a), "demo_new.json")
    assert not os.path.exists(fn), "exists: " + fn
    p = Q2.picked(a)
    hs = [p] if p == 0 else [p, 0]
    bk = Q2.backbone(a)
    out = dict(rung=a.rung, mode=Q2.mode_of(a), ckpt=a.ckpt, picked=p, demos={}, summary={})
    for c, tid in DEMOS:
        path = os.path.join(REC, "%s_familiar_s10_st24.npz" % c)
        ex = Q2.load_demo(path)
        assert ex["task"] == c, (path, ex["task"])
        for h in hs:
            rec = Q2.parse_demo(bk, ex, Q2.settings()[h], a.rung, Q2.mode_of(a), tid=tid)
            out["demos"]["%s|h%d" % (c, h)] = rec
            print("[q2n] %s h%d pred %s ok %s | flow loss %s ok %s | truth %s %s" % (
                c, h, json.dumps(rec["pred"]), json.dumps(rec["ok"]), json.dumps(rec["fm_pred"]), json.dumps(rec["fm_ok"]),
                rec["true_obj"], rec["true_tgt"]), flush=True)
    for h in hs:
        rs = [out["demos"]["%s|h%d" % (c, h)] for c, _ in DEMOS]
        out["summary"]["h%d" % h] = {fam: {k: sum(float(bool(r[fam][k])) for r in rs) / len(rs) for k in rs[0][fam]}
                                     for fam in ("ok", "fm_ok")}
    print("[q2n] summary", json.dumps(out["summary"]), flush=True)
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    with open(fn + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=str)
    os.replace(fn + ".tmp", fn)


if __name__ == "__main__":
    main()
