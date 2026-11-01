"""FLAWED_COMPOSITION prototype A, stage 1m read (section 5.24; m3_prereg tag pa1m): arm Ap (Ao with the occupancy restricted to local
free space) trained on the demos plus gen_lc.npz, fresh seeds 15, 16, 17 (tag m; rollouts named Am in results/flawed/a1m_roll).  CPU.
FROZEN criteria, exactly fa1j_read's (C1, C2, 3F, C4 vs Hg / Rg per layout; C3 descriptive).
Descriptive only: Al (1l) and Ak (1k) beside; Hl / Rl (Hb / Rb on gen_lc, seed 12, 1l's runs); the POST-HOC object-matched C2;
the cheese / bottle hold factors.
-> results/flawed/a1m_read.json (+ a1m_read_familiar.json, a1m_read_new.json)

    python experiments/fa1m_read.py
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

A, H, R, SEEDS_A = "Am", "Hg", "Rg", (15, 16, 17)
SIDE = {"Al": (12, 13, 14), "Ak": (6, 7, 8)}

if __name__ == "__main__":
    p = os.path.join(RX.OUT, J.REF_FILE)
    assert hashlib.sha256(open(p, "rb").read()).hexdigest() == J.REF_SHA, "the 1b reference file changed"
    ref = float(json.load(open(p, encoding="utf-8"))["cells"]["Hb"]["s_known"]["est"])
    RX.ROLLDIR.update({A: "a1m_roll", H: "a1g_roll", R: "a1g_roll", "Al": "a1l_roll", "Ak": "a1k_roll", "Hl": "a1l_roll", "Rl": "a1l_roll"})
    J.A, J.H, J.R, J.SEEDS_A = A, H, R, SEEDS_A
    L = J.layout_invariance()
    out = {}
    for lay in ("familiar", "new"):
        refs = [(m, ss) for m, ss in SIDE.items()] + [("Hl", (12,)), ("Rl", (12,))]
        res = RX.run(RX.Read(A, (), refs, H=H, R=R, layout=lay, a_seeds=SEEDS_A), f"a1m_read_{lay}.json")
        res["three_factor"] = {A: three_factor(A, lay, seeds=SEEDS_A), **{m: three_factor(m, lay, seeds=ss) for m, ss in SIDE.items()}}
        res["verdict"] = J.verdict(res["verdict"], lay, res["cells"][A]["s_known"]["est"], res["three_factor"][A], ref, L)
        allm = [(A, SEEDS_A)] + list(SIDE.items())
        res["K5_K6"] = {c: {f"{m}_s{s}": res["per_compound"][c][f"{m}_s{s}"]["s"] for m, ss in allm for s in ss} for c in ("K5", "K6")}
        res["post_hoc_object_matched_C2_descriptive"] = {m: K.object_matched(res, m, ss) for m, ss in allm}
        res["hold_factor_descriptive"] = {m: dict(cheese=res["three_factor"][m]["N1"]["h_o"], bottle=res["three_factor"][m]["N4"]["h_o"])
                                          for m, _ in allm}
        res["descriptive_same_data_baselines"] = {b: dict(s_known=res["cells"][b]["s_known"]["est"], s_new=res["cells"][b]["s_new"]["est"])
                                                  for b in ("Hl", "Rl") if b in res["cells"]}
        out[lay] = res
    v = dict(familiar=out["familiar"]["verdict"]["verdict"], new=out["new"]["verdict"]["verdict"], reference=ref, reference_file=J.REF_FILE,
             reference_sha256=J.REF_SHA, layout_invariance=L)
    v["verdict"] = f"familiar: {v['familiar']} | new layout: {v['new']}"
    v["confirmed"] = v["familiar"] == "compositional" and v["new"] == "compositional"
    json.dump(dict(A=A, arm="Ap", seeds=SEEDS_A, H=H, R=R, gen="gen_lc.npz", verdict=v, layouts=out),
              open(os.path.join(RX.OUT, "a1m_read.json"), "w", encoding="utf-8"), indent=1)
    print("VERDICT_FINAL", json.dumps(v))
    for lay in out:
        print("PERLAYOUT", lay, json.dumps(out[lay]["verdict"]))
        print("K5K6", lay, json.dumps(out[lay]["K5_K6"]))
        print("HOLD", lay, json.dumps(out[lay]["hold_factor_descriptive"]))
        print("OBJ", lay, json.dumps(out[lay]["post_hoc_object_matched_C2_descriptive"]), json.dumps(out[lay]["descriptive_same_data_baselines"]))
        for c, d in out[lay]["three_factor"][A].items():
            print("3F", lay, c, json.dumps({k: (round(x, 3) if isinstance(x, float) else [round(y, 3) for y in x]) for k, x in d.items()}))
