"""PARTS_RESULTS 46 (descriptive): summary of pp_far_diag.py bb files.  Per file (driver x task, new layout), per evaluated model and m
setting (and size probe), per horizontal-distance bin: n calls, median commanded vertical step (vz), share rising (vz > 0.1), median
commanded horizontal step toward the object (prog).  Also the driver's episodes (holds, successes).  CPU.
-> results/flawed/pp_bb_read.json

    python experiments/pp_bb_read.py drtrm0_K4_new+dnom_K4_new+...
"""
import json
import os
import sys

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "flawed")


def main():
    res = {}
    for nm in sys.argv[1].split("+"):
        z = json.load(open(os.path.join(OUT, f"pp_bb_{nm}.json"), encoding="utf-8"))
        E = z["episodes"]
        r = dict(driver=z["driver"], driver_bb_m=z["driver_bb_m"], n=len(E), held=sum(e["held_step"] is not None for e in E),
                 success=sum(e["success"] for e in E), n_calls=len(z["calls"]),
                 m_built=sorted({c["m_built"] for c in z["calls"]}), table={})
        for k, v in z["summary"].items():
            model, setting, b = k.split("|")
            r["table"].setdefault(f"{model}|{setting}", {})[b] = [v["n"], round(v["vz_med"], 2), round(v["rise_share"], 2),
                                                                     round(v["prog_med"], 2)]
        res[nm] = r
        print("==", nm, {q: r[q] for q in ("driver_bb_m", "n", "held", "success", "n_calls", "m_built")}, flush=True)
        for k, v in r["table"].items():
            print("  ", k, "  ".join(f"{b}:{x}" for b, x in v.items()), flush=True)
    out = "pp_bb_read.json" if len(sys.argv) < 3 else f"pp_bb_read_{sys.argv[2]}.json"     # optional output suffix
    assert not os.path.exists(os.path.join(OUT, out)), f"{out} exists"
    json.dump(dict(cols="[n, vz_med, rise_share, prog_med]", runs=res), open(os.path.join(OUT, out), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
