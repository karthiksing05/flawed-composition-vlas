"""FLAWED_COMPOSITION prototype A, stage 1k read (section 5.19; m3_prereg tag pa1k): Ai, architecture unchanged, trained on the demos
plus EQUAL per-object generated support (DATA/gen_k.npz: 200 generated relocated carries per carried object), fresh seeds 6, 7, 8
(tag k; rollouts named Ak in results/flawed/a1k_roll).  CPU.
FROZEN criteria (user, 2026-10-31), per layout, exactly fa1j_read's:
  familiar  C1  s_A(known) - 0.672 >= -0.10 (point; 0.672 = stage 1b's demo-only Hb, a1b_read.json, hash checked)
  new       C1  layout invariance L = s_A(known, new) / s_A(known, familiar) >= 0.8 (point) and CI lower bound >= 0.6
  both      C2  (engine: ratio >= 0.8 point, CI lower >= 0.6), 3F (every N1-N6 CI of D3 contains 0; N6 in),
            C4  (vs Hg and Rg, 1g's runs: both differences > 0 with CI excluding 0)
  C3 DESCRIPTIVE.  per layout: floor -> "uninformative"; all four -> "compositional"; else "not compositional (fails: ...)"
  verdict = "familiar: <v> | new layout: <v>"; CONFIRMED iff both layouts read "compositional".
Descriptive only, not part of the verdict: Aj (1j, seeds 3-5) beside; Hk / Rk (Hb / Rb architectures on the same gen_k data, seed 6);
the object-matched C2 (POST HOC, labelled so): per object, mean new-pairing success over mean known-task success of that object.
-> results/flawed/a1k_read.json (+ a1k_read_familiar.json, a1k_read_new.json)

    python experiments/fa1k_read.py
"""
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fa1j_read as J                                                        # noqa: E402
import fa1x_read as RX                                                       # noqa: E402
from fa1g_read import three_factor                                           # noqa: E402

A, H, R, SEEDS_A = "Ak", "Hg", "Rg", (6, 7, 8)
REF_FILE, REF_SHA = J.REF_FILE, J.REF_SHA
OBJ = {"cheese": (("K6",), ("N1", "N2", "N3")), "bottle": (("K4", "K5"), ("N4", "N5", "N6"))}


def object_matched(res, m, seeds):
    pc = res["per_compound"]
    out = {}
    for ob, (kn, nw) in OBJ.items():
        k = float(np.mean([pc[c][f"{m}_s{s}"]["s"] for c in kn for s in seeds]))
        n = float(np.mean([pc[c][f"{m}_s{s}"]["s"] for c in nw for s in seeds]))
        out[ob] = dict(known=k, new=n, ratio=(n / k if k > 0 else None))
    return out


if __name__ == "__main__":
    p = os.path.join(RX.OUT, REF_FILE)
    assert hashlib.sha256(open(p, "rb").read()).hexdigest() == REF_SHA, "the 1b reference file changed"
    ref = float(json.load(open(p, encoding="utf-8"))["cells"]["Hb"]["s_known"]["est"])
    RX.ROLLDIR.update({A: "a1k_roll", H: "a1g_roll", R: "a1g_roll", "Aj": "a1j_roll", "Hk": "a1k_roll", "Rk": "a1k_roll"})
    J.A, J.H, J.R, J.SEEDS_A = A, H, R, SEEDS_A
    L = J.layout_invariance()
    out = {}
    for lay in ("familiar", "new"):
        res = RX.run(RX.Read(A, (), [("Aj", (3, 4, 5)), ("Hk", (6,)), ("Rk", (6,))], H=H, R=R, layout=lay, a_seeds=SEEDS_A),
                     f"a1k_read_{lay}.json")
        res["three_factor"] = {A: three_factor(A, lay, seeds=SEEDS_A), "Aj": three_factor("Aj", lay, seeds=(3, 4, 5))}
        res["verdict"] = J.verdict(res["verdict"], lay, res["cells"][A]["s_known"]["est"], res["three_factor"][A], ref, L)
        res["K5_K6"] = {c: {f"{m}_s{s}": res["per_compound"][c][f"{m}_s{s}"]["s"] for m, ss in ((A, SEEDS_A), ("Aj", (3, 4, 5)))
                            for s in ss} for c in ("K5", "K6")}
        res["post_hoc_object_matched_C2_descriptive"] = {A: object_matched(res, A, SEEDS_A), "Aj": object_matched(res, "Aj", (3, 4, 5))}
        res["descriptive_same_data_baselines"] = {b: dict(s_known=res["cells"][b]["s_known"]["est"], s_new=res["cells"][b]["s_new"]["est"])
                                                  for b in ("Hk", "Rk") if b in res["cells"]}
        out[lay] = res
    v = dict(familiar=out["familiar"]["verdict"]["verdict"], new=out["new"]["verdict"]["verdict"], reference=ref, reference_file=REF_FILE,
             reference_sha256=REF_SHA, layout_invariance=L)
    v["verdict"] = f"familiar: {v['familiar']} | new layout: {v['new']}"
    v["confirmed"] = v["familiar"] == "compositional" and v["new"] == "compositional"
    json.dump(dict(A=A, arm="Ai", seeds=SEEDS_A, H=H, R=R, gen="gen_k.npz", verdict=v, layouts=out),
              open(os.path.join(RX.OUT, "a1k_read.json"), "w", encoding="utf-8"), indent=1)
    print("VERDICT_FINAL", json.dumps(v))
    for lay in out:
        print("PERLAYOUT", lay, json.dumps(out[lay]["verdict"]))
        print("K5K6", lay, json.dumps(out[lay]["K5_K6"]))
        print("OBJ", lay, json.dumps(out[lay]["post_hoc_object_matched_C2_descriptive"]), json.dumps(out[lay]["descriptive_same_data_baselines"]))
        for c, d in out[lay]["three_factor"][A].items():
            print("3F", lay, c, json.dumps({k: (round(x, 3) if isinstance(x, float) else [round(y, 3) for y in x]) for k, x in d.items()}))
