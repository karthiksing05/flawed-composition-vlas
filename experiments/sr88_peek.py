import json
import numpy as np
RW = "/coc/flash10/ksingara3/flawed-composition-vlas/results/flawed/raasw5"
for arm in ("w5", "w4", "w5c1", "w5c2", "raasl"):
    d = json.load(open("%s/b_%s/full_4_wa.json" % (RW, arm)))
    res = {e["trial"]: bool(e["success"]) for e in json.load(open("%s/b_%s/full_4.json" % (RW, arm)))["tasks"]["4"]["episodes"]}
    rows = []
    for e in d["eps"]:
        q = [x for x in e["queries"] if x["held"]]
        if q:
            er = np.array([x["err_obj_cm"] for x in q]); et = np.array([x["err_tgt_cm"] for x in q])
            rows.append((e["trial"], res[e["trial"]], er[0], np.median(er), er.max(), np.median(et), len(q)))
    for r in rows[:0]:
        print(arm, "trial %2d succ=%d held err_obj first %.2f med %.2f max %.2f | err_tgt med %.2f | n=%d" % r)
    a = np.array([[r[2], r[3], r[4], r[5]] for r in rows])
    print(arm, "mean(first)=%.2f mean(med)=%.2f mean(max)=%.2f; frac (max-first)<0.5: %.2f ; err_tgt med mean %.2f" % (a[:, 0].mean(), a[:, 1].mean(), a[:, 2].mean(), np.mean((a[:, 2] - a[:, 0]) < 0.5), a[:, 3].mean()))
