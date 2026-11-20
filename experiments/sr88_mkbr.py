"""SR 88: state selection for the instrumented re-runs (rule fixed here, applied to results/flawed/sr88/pick.json = the SR 87 outcomes).
Per cell, 20 states: (1) W5 failed and W4 or W5c1 (uniL: or RAA-sL) succeeded, lowest trial ids first, at most 14;  (2) all-fail (W5, W4, W5c1) up to 4;
(3) controls = W5, W4, W5c1 all succeeded, lowest ids, to fill 20.  Branch files are subsets of the SR 87 branch files."""
import json
R = "/coc/flash10/ksingara3/flawed-composition-vlas/results/flawed"
P = json.load(open(R + "/sr88/pick.json"))
SRC = {("a", 22): R + "/n2_branches_wx.json", ("a", 26): R + "/n2_branches_wx.json", ("a", 10): R + "/n2_branches_wx.json", ("a", 0): R + "/n2_branches_wx.json",
       ("b", 4): R + "/rg_comp_x/br7_main.json", ("b", 8): R + "/rg_comp_x/br6_main.json"}
sel = {}
import os
os.makedirs(R + "/sr88/br", exist_ok=True)
for cell, d in P.items():
    rows = {r[0]: r[1] for r in d["rows"]}
    cmp = ["w4", "w5c1"] + (["raasl"] if cell == "uniL" else [])
    L = [t for t in sorted(rows) if rows[str(t) if False else t]["w5"] == 0 and any(rows[t][a] == 1 for a in cmp)][:14]
    FF = [t for t in sorted(rows) if rows[t]["w5"] == 0 and t not in L and all(rows[t][a] != 1 for a in cmp)][:4]
    C = [t for t in sorted(rows) if rows[t]["w5"] == 1 and rows[t]["w4"] == 1 and rows[t]["w5c1"] == 1][:20 - len(L) - len(FF)]
    S = sorted(set(L + FF + C))
    sel[cell] = dict(L=L, FF=FF, C=C, all=S, tid=d["tid"], phase=d["phase"])
    br = json.load(open(SRC[(d["phase"], d["tid"])]))
    sub = [b for b in br if b["trial"] in S]
    assert len(sub) == len(S), (cell, len(sub), len(S))
    json.dump(sub, open(R + "/sr88/br/br_%s_%d.json" % (d["phase"], d["tid"]), "w"), indent=1)
    print(cell, "tid", d["tid"], "L", L, "FF", FF, "C", C, flush=True)
json.dump(sel, open(R + "/sr88/sel.json", "w"), indent=1)
