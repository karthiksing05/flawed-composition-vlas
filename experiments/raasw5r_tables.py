"""SR 91: markdown tables from results/flawed/raasw5r_read[_T].json and raasw5r_wa_gated[_T].json (numpy-free; compute node).   python experiments/raasw5r_tables.py [--tag T]"""
import argparse, json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "flawed")
ap = argparse.ArgumentParser(); ap.add_argument("--tag", default=""); a = ap.parse_args()
sfx = ("_" + a.tag.lstrip("_")) if a.tag else ""
A = json.load(open(os.path.join(OUT, "raasw5r_read%s.json" % sfx), encoding="utf-8"))
B = json.load(open(os.path.join(OUT, "raasw5r_wa_gated%s.json" % sfx), encoding="utf-8"))["W5"]
f = lambda v: "-" if v is None else "%.3f" % v
print("## Phase A labels (paired bootstrap B=2000, rng 0)")
print("| label | delta [95% CI] | verdict |\n|---|---|---|")
for k, v in A["verdicts"].items():
    if v.get("label") == "MISSING" or "ci" not in v:
        print("| %s | MISSING | |" % k); continue
    print("| %s | %+.3f [%+.3f, %+.3f] | %s |" % (k, v["delta"], v["ci"][0], v["ci"][1], v.get("label") or "(no label)"))
print("\n## Phase A per cell (60 fresh states)")
arms = A["arms"]
print("| cell | " + " | ".join(arms) + " |\n|---|" + "---|" * len(arms))
for c in ["N1", "N2", "N3", "N4", "N5", "N6", "K1", "K2", "K3", "K4", "K5", "K6"]:
    print("| %s | %s |" % (c, " | ".join(f(A["per_task"]["%s|%s" % (x, c)]["success"]) for x in arms)))
for g in ("N", "K"):
    print("| %s1-%s6 | %s |" % (g, g, " | ".join(f(A["success"]["%s|%s" % (x, g)]) for x in arms)))
print("\n## Phase B gates / D labels (W5, full arm)")
print("gates:", {k: (round(v["value"], 3) if v["value"] is not None else None, v["label"]) for k, v in B["gates"].items()})
print("inputs:", B["inputs"], " wa_read primary:", B["wa_read_primary"], " GATED PRIMARY:", B["primary_gated"])
print("| contrast | delta [95% CI] | label |\n|---|---|---|")
for k, v in list(B["deltas"].items()) + list(B["extra"].items()):
    if v.get("delta") is None:
        print("| %s | %s | |" % (k, v.get("label"))); continue
    ci = v["ci"]
    print("| %s | %+.3f [%s, %s] | %s |" % (k, v["delta"], "-" if ci[0] is None else "%+.3f" % ci[0], "-" if ci[1] is None else "%+.3f" % ci[1], v.get("label") or ""))
print("\n## Phase B per cell (full arm unless noted; success over the cell's 60 fresh states)")
names = ["F7_relL", "F7_relR", "F7_pos", "F7_neg", "F7_uniL", "F7_uniR", "F7_uniP", "K2t", "F6_b", "F6_a", "K1t"]
cols = ["w5", "raasl", "w5noR", "w5noM", "I", "rgp"]
print("| cell | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols))
for t, n in enumerate(names):
    print("| %s | %s |" % (n, " | ".join(f(B["per_cell_success"].get("%s|%d" % (x, t))) for x in cols)))
P = os.path.join(OUT, "raasw5r_pooled%s.json" % sfx)
if os.path.exists(P):
    PV = json.load(open(P, encoding="utf-8"))
    print("\n## SECONDARY pooled over %s (SR 87 / SR 90 data seen before registration)" % PV["sets"])
    print("| contrast | pooled delta [95% CI] | per-set deltas | label |\n|---|---|---|---|")
    for k, v in PV["verdicts"].items():
        print("| %s | %+.3f [%+.3f, %+.3f] | %s | %s |" % (k, v["delta"], v["ci"][0], v["ci"][1], ", ".join("%s %+.3f" % (s, d) for s, d in v["per_set"].items()), v.get("label") or ""))
