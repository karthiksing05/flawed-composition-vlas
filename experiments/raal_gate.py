"""SR 80 post-read: writes raal_wa_gated.json from raal_wa_read.json (raal_read.py readb crashed AFTER raal_wa_read.json was written, in wa_read's print loop,
on a contrast whose arms were not rolled (noMe); same gating code as raal_read.readb with .get for delta) and prints compact tables.  Refuses to overwrite."""
import json, os
OUT = "/coc/flash10/ksingara3/flawed-composition-vlas/results/flawed"
o2 = os.path.join(OUT, "raal_wa_gated.json")
if os.path.exists(o2):
    raise SystemExit("exists")
r = json.load(open(os.path.join(OUT, "raal_wa_read.json"), encoding="utf-8"))
gt = {k: (v["label"] == "TESTABLE") for k, v in r["gates"].items()}
inp = r["primary_inputs"]
allg = all(gt.values()) and len(gt) == 3
shown = inp["D1"] == "SUPPORTED" and inp["D2"] == "SUPPORTED" and inp["D3a"] == "EQUIVALENT" and inp["D3b"] == "EQUIVALENT"
g = dict(wa_read_primary=r["primary"], inputs=inp, gates=r["gates"],
         primary_gated=("DISSOCIATION SHOWN" if (shown and allg) else ("UNTESTABLE (D1, D2, D3a, D3b pass, a gate fails)" if shown else r["primary"])),
         deltas={k: dict(delta=v.get("delta"), ci=v.get("ci"), label=v.get("label")) for k, v in r["contrasts"].items()})
json.dump(g, open(o2, "w", encoding="utf-8"), indent=1)
print("GATED", g["primary_gated"], json.dumps(r["gates"]))
print("INPUTS", json.dumps(inp))
for k, v in r["contrasts"].items():
    print("C", k, v.get("delta"), v.get("ci"), v.get("label"))
for k, v in r["success"].items():
    print("S", k, v)
for k, v in r["channel"].items():
    if k.startswith("full/") or k.startswith("noR/") or k.startswith("noM/"):
        print("CH", k, v["queries"], round(v["omitted_share"], 3), round(v["err_median_cm"], 2))
x = json.load(open(os.path.join(OUT, "raal_read.json"), encoding="utf-8"))
print("A-SUCCESS", json.dumps(x["success"]))
print("A-GRASP", json.dumps(x["grasped_right"]))
print("A-PLACE", json.dumps(x["place_given_grasp"]))
print("A-VERDICTS")
for k, v in x["verdicts"].items():
    print("V", k, json.dumps(v))
print("A-FIT", json.dumps(x["fit_gate"]), x["primary"], x["overall"])
for k, v in x["per_task"].items():
    print("PT", k, json.dumps(v)[:600])
