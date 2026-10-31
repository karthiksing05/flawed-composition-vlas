"""FLAWED_COMPOSITION prototype A, stage 1h read (section 5.14; m3_prereg tag pa1h).  CPU.
A = Ah_rot (the rotation-since-onset route only; seeds 0-2, tag h, demos + 1g's generated carries), ablation Ah (+ the grasp offset,
seed 0); rollouts results/flawed/a1h_roll.  (The offset was moved to the ablation before any rollout: the identity probe on training
rows, results/flawed/a1h_probe.json, decodes the held object from it at 0.97, as much as 1d's cut proxies.)
CRITERION CHANGE approved by the user after the 1g read (5.13, 5.14): the floor and the C1 reference are the demo-only Hb's known-task
success on the familiar layout, measured in stage 1b before any generated data existed: results/flawed/a1b_read.json
cells.Hb.s_known.est = 0.672 (file sha256 74fcd712...; commit 3ae1e9fc), a FIXED number, the same for both layouts:
  floor  REF < 0.5 -> "uninformative"       C1  s_A(known) - REF >= -0.10 (point, as before)
C2, C3, C4 as in 5.6, with C4 against Hg and Rg (1g's baselines on the same data, rollouts a1g_roll); 3F as in 5.11.  Per layout
(familiar: even tids, new: odd tids) the 5.11 mapping; verdict = "familiar: <v> | new layout: <v>"; the guarantee iff both read
"compositional; three-factor law holds".  Ah_rot, Ag (1g) and Ae_nosize (1f) shown beside; K5 / K6 per arm reported.
-> results/flawed/a1h_read.json (+ a1h_read_familiar.json, a1h_read_new.json, the engine's per-layout reads)

    python experiments/fa1h_read.py
"""
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fa1x_read as RX                                                       # noqa: E402
from fa1g_read import three_factor                                           # noqa: E402

A, ABL, H, R = "Ah_rot", "Ah", "Hg", "Rg"
REF_FILE, REF_SHA = "a1b_read.json", "74fcd712eef3c263b1b2c797d06de20edb103f5c0682f3dec2c1c631ec256760"


def verdict(v, s_known, tf, ref):
    v = dict(v)
    v["engine_verdict"] = v.pop("verdict")
    v["engine_floor_Hg"], v["engine_C1_vs_Hg"] = v.pop("floor"), v.pop("C1")
    v["floor"] = ref < 0.5
    v["C1_value"] = s_known - ref
    v["C1"] = v["C1_value"] >= -0.10
    v["3F"] = all(d["D3_ci"][0] <= 0 <= d["D3_ci"][1] for d in tf.values())
    v["3F_pairings_in_ci"] = sorted(c for c, d in tf.items() if d["D3_ci"][0] <= 0 <= d["D3_ci"][1])
    order = ("C1", "C2", "C3", "C4", "3F")
    if v["floor"]:
        v["verdict"] = "uninformative"
    elif all(v[k] for k in order):
        v["verdict"] = "compositional; three-factor law holds"
    elif all(v[k] for k in order[:4]):
        v["verdict"] = "compositional by C1-C4; three-factor law fails"
    else:
        v["verdict"] = "not compositional (fails: " + ", ".join(k for k in order if not v[k]) + ")"
    return v


if __name__ == "__main__":
    p = os.path.join(RX.OUT, REF_FILE)
    assert hashlib.sha256(open(p, "rb").read()).hexdigest() == REF_SHA, "the 1b reference file changed"
    ref = float(json.load(open(p, encoding="utf-8"))["cells"]["Hb"]["s_known"]["est"])
    RX.ROLLDIR.update({A: "a1h_roll", ABL: "a1h_roll", H: "a1g_roll", R: "a1g_roll", "Ag": "a1g_roll", "Ae_nosize": "a1f_roll"})
    out = {}
    for lay in ("familiar", "new"):
        refs = [("Ag", (0, 1, 2))] + ([("Ae_nosize", (0, 1, 2))] if lay == "familiar" else [])
        res = RX.run(RX.Read(A, (ABL,), refs, H=H, R=R, layout=lay), f"a1h_read_{lay}.json")
        res["three_factor"] = {m: three_factor(m, lay) for m in (A, "Ag")}
        res["verdict"] = verdict(res["verdict"], res["cells"][A]["s_known"]["est"], res["three_factor"][A], ref)
        res["K5_K6"] = {c: {f"{m}_s{s}": res["per_compound"][c][f"{m}_s{s}"]["s"] for m, ss in [(A, (0, 1, 2)), (ABL, (0,)), ("Ag", (0, 1, 2))]
                            for s in ss} for c in ("K5", "K6")}
        out[lay] = res
    v = dict(familiar=out["familiar"]["verdict"]["verdict"], new=out["new"]["verdict"]["verdict"], reference=ref, reference_file=REF_FILE,
             reference_sha256=REF_SHA)
    v["verdict"] = f"familiar: {v['familiar']} | new layout: {v['new']}"
    v["guarantee"] = all(x == "compositional; three-factor law holds" for x in (v["familiar"], v["new"]))
    json.dump(dict(A=A, H=H, R=R, verdict=v, layouts=out), open(os.path.join(RX.OUT, "a1h_read.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(dict(verdict=v, per_layout={k: x["verdict"] for k, x in out.items()}), indent=1))
    for lay in out:
        print("K5K6", lay, json.dumps(out[lay]["K5_K6"]))
        for c, d in out[lay]["three_factor"][A].items():
            print("3F", lay, c, json.dumps({k: (round(x, 3) if isinstance(x, float) else [round(y, 3) for y in x]) for k, x in d.items()}))
