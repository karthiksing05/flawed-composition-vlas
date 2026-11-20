"""SR 86 Q1 part e (exploratory): which robot / object contacts occur during the hover (module-owned, chan rel_xy < 1.5 cm) of the W3 K5 rollouts; z command during hover."""
import collections, glob, json, os
import numpy as np
D = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "flawed", "sr86", "s86w3")
cf = sorted(glob.glob(os.path.join(D, "chan_*.jsonl")))[0]; ch = [json.loads(l) for l in open(cf, encoding="utf-8")]
wh = [json.loads(l) for l in open(cf.replace("chan_", "where_"), encoding="utf-8")]; wh = wh[len(wh) - len(ch):]
by = collections.defaultdict(list)
for c, w in zip(ch, wh): by[c["ep"]].append((c, w))
R, O, zc, zh, eez = collections.Counter(), collections.Counter(), [], [], []
nh = 0
for f in sorted(glob.glob(os.path.join(D, "instr_raac_8_t*.npz"))):
    t = int(f.rsplit("_t", 1)[1][:-4]); d = np.load(f, allow_pickle=True)
    A = np.asarray(d["a"], float); obj = np.asarray(d["obj"], float); eef = np.asarray(d["eef"], float); cons = [str(s) for s in d["cons"]]
    hov = [int(c["t"]) + 10 for c, w in by[t] if c["role"] == "target" and w["own"] and w["rel_xy"] < 0.015]
    steps = sorted(set(r + k for r in hov for k in range(5) if r + k < len(obj)))
    if not steps: continue
    nh += 1
    sr = collections.Counter(); so = collections.Counter()
    for i in steps:
        o, r = cons[i].split("||") if "||" in cons[i] else (cons[i], "")
        for x in r.split(";"):
            if x: sr[x.split(">")[0] + ">" + x.split(">")[-1].split("_g")[0]] += 1
        for x in o.split(";"):
            if x and "gripper0" not in x: so[x.split("/")[0]] += 1
    R.update({k: 1 for k in sr}); O.update({k: 1 for k in so})
    zc.append(float(np.mean(A[steps, 2]))); zh.append(float(obj[steps[-1], 2] - obj[steps[0], 2])); eez.append(float(eef[steps[-1], 2] - eef[steps[0], 2]))
    print(t, "hover steps", len(steps), "mean z cmd %.2f" % zc[-1], "obj dz over hover %.1f cm" % (100 * zh[-1]), "robot contacts (steps):", dict(sr), "obj non-gripper contacts:", dict(so))
print("episodes with hover", nh, "| episodes with robot-link contact by name", dict(R), "| object contacts", dict(O))
print("median mean-z-cmd during hover %.2f, median obj dz %.1f cm" % (np.median(zc), 100 * np.median(zh)))
