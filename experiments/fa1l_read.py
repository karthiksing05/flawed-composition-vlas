"""FLAWED_COMPOSITION prototype A, stage 1l read (section 5.21; m3_prereg tag pa1l): Ai, architecture unchanged, trained on the demos
plus gen_l.npz (1k's carries plus the grasp segments of every failed generated attempt, role-wise acceptance), fresh seeds 9, 10,
11 (tag l; rollouts named Al in results/flawed/a1l_roll).  CPU.
FROZEN criteria, exactly fa1j_read's / fa1k_read's (C1, C2, 3F, C4 vs Hg / Rg per layout; C3 descriptive).
Descriptive only: Ak (1k) beside; Hl / Rl (Hb / Rb architectures on gen_l, seed 9); the POST-HOC object-matched C2.
-> results/flawed/a1l_read.json (+ a1l_read_familiar.json, a1l_read_new.json)

    python experiments/fa1l_read.py
"""
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fa1j_read as J                                                        # noqa: E402
import fa1k_read as K                                                        # noqa: E402
import fa1x_read as RX                                                       # noqa: E402
from fa1g_read import three_factor                                           # noqa: E402

A, H, R, SEEDS_A = "Al", "Hg", "Rg", (9, 10, 11)
PREV, PREV_SEEDS = "Ak", (6, 7, 8)

if __name__ == "__main__":
    p = os.path.join(RX.OUT, J.REF_FILE)
    assert hashlib.sha256(open(p, "rb").read()).hexdigest() == J.REF_SHA, "the 1b reference file changed"
    ref = float(json.load(open(p, encoding="utf-8"))["cells"]["Hb"]["s_known"]["est"])
    RX.ROLLDIR.update({A: "a1l_roll", H: "a1g_roll", R: "a1g_roll", PREV: "a1k_roll", "Hl": "a1l_roll", "Rl": "a1l_roll"})
    J.A, J.H, J.R, J.SEEDS_A = A, H, R, SEEDS_A
    L = J.layout_invariance()
    out = {}
    for lay in ("familiar", "new"):
        res = RX.run(RX.Read(A, (), [(PREV, PREV_SEEDS), ("Hl", (9,)), ("Rl", (9,))], H=H, R=R, layout=lay, a_seeds=SEEDS_A),
                     f"a1l_read_{lay}.json")
        res["three_factor"] = {A: three_factor(A, lay, seeds=SEEDS_A), PREV: three_factor(PREV, lay, seeds=PREV_SEEDS)}
        res["verdict"] = J.verdict(res["verdict"], lay, res["cells"][A]["s_known"]["est"], res["three_factor"][A], ref, L)
        res["K5_K6"] = {c: {f"{m}_s{s}": res["per_compound"][c][f"{m}_s{s}"]["s"] for m, ss in ((A, SEEDS_A), (PREV, PREV_SEEDS))
                            for s in ss} for c in ("K5", "K6")}
        res["post_hoc_object_matched_C2_descriptive"] = {A: K.object_matched(res, A, SEEDS_A), PREV: K.object_matched(res, PREV, PREV_SEEDS)}
        res["descriptive_same_data_baselines"] = {b: dict(s_known=res["cells"][b]["s_known"]["est"], s_new=res["cells"][b]["s_new"]["est"])
                                                  for b in ("Hl", "Rl") if b in res["cells"]}
        out[lay] = res
    v = dict(familiar=out["familiar"]["verdict"]["verdict"], new=out["new"]["verdict"]["verdict"], reference=ref, reference_file=J.REF_FILE,
             reference_sha256=J.REF_SHA, layout_invariance=L)
    v["verdict"] = f"familiar: {v['familiar']} | new layout: {v['new']}"
    v["confirmed"] = v["familiar"] == "compositional" and v["new"] == "compositional"
    json.dump(dict(A=A, arm="Ai", seeds=SEEDS_A, H=H, R=R, gen="gen_l.npz", verdict=v, layouts=out),
              open(os.path.join(RX.OUT, "a1l_read.json"), "w", encoding="utf-8"), indent=1)
    print("VERDICT_FINAL", json.dumps(v))
    for lay in out:
        print("PERLAYOUT", lay, json.dumps(out[lay]["verdict"]))
        print("K5K6", lay, json.dumps(out[lay]["K5_K6"]))
        print("OBJ", lay, json.dumps(out[lay]["post_hoc_object_matched_C2_descriptive"]), json.dumps(out[lay]["descriptive_same_data_baselines"]))
        for c, d in out[lay]["three_factor"][A].items():
            print("3F", lay, c, json.dumps({k: (round(x, 3) if isinstance(x, float) else [round(y, 3) for y in x]) for k, x in d.items()}))
