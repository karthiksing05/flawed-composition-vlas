"""XB failure-class shares (S/G/W/O/H/X) from policy-free pose logs, held-out vs partner cells.  CPU, numpy.  Writes results/bench/xb/classes.json (refuses to overwrite).
Operational definitions (logged in docs/notes/xb_status.md; the registration names the classes but not the thresholds):
  S success (the benchmark's checker).  G the instructed object never rose 2 cm above its reset height.  H lifted and, at the last step, still within 8 cm of the end effector
  (MESA/BiMESA, RoboCasa) or still >= 2 cm above its reset height (RoboTwin: no eef track).  W lifted, not held, and its final xy lies within 10 cm of the centre of a receptacle that is not the instructed
  one (MESA: other receptacle-class objects in the scene; RoboCasa: other fixtures, half the fixture size + 5 cm; RoboTwin: n/a, there is no competing receptacle).
  O lifted, not held, on no non-instructed receptacle, not successful.  X anything else (log error, object not identified).
"""
import glob, json, os
import numpy as np

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
XB = ROOT + "/results/bench/xb"
RKEYS = ["tray", "plate", "bowl", "basket", "pan", "cutting_board", "pot", "baking", "drainer", "box", "cabinet", "microwave", "sink", "stove", "oven", "drawer", "fridge", "blender", "toaster", "dishwasher"]
kc = lambda s: s.lower().replace(" ", "_").replace("-", "_")


def cls_mesa(ep, npz, info):
    if ep["success"]:
        return "S"
    cats = ep["objects_categories"]; names = list(npz["names"]); pos = npz["pos"]; eef = npz["eef"]
    ob = [i for i, n in enumerate(names) if info["objects"][0].replace(" ", "_") in kc(cats.get(n, ""))]
    if not ob:
        return "X"
    i = ob[0]; z = pos[:, i, 2]
    if z.max() - z[0] < 0.02:
        return "G"
    e = np.asarray(eef); d = np.linalg.norm(pos[-1, i, :] - (e[-1].reshape(-1, 3)), axis=1).min() if e.size else 9.0
    if d < 0.08:
        return "H"
    rk = info["receptacle"].split(":")[0]
    oth = [j for j, n in enumerate(names) if j != i and any(k in kc(cats.get(n, "")) for k in RKEYS) and rk.replace(" ", "_") not in kc(cats.get(n, ""))]
    if any(np.linalg.norm(pos[-1, i, :2] - pos[-1, j, :2]) < 0.10 for j in oth):
        return "W"
    return "O"


def cls_rc(ep, npz):
    if ep["success"]:
        return "S"
    names = list(npz["names"]); pos = npz["pos"]
    if "obj" not in names or len(pos) == 0:
        return "X"
    i = names.index("obj"); z = pos[:, i, 2]
    if z.max() - z[0] < 0.02:
        return "G"
    if np.linalg.norm(pos[-1, i] - np.asarray(npz["eef"])[-1]) < 0.08:
        return "H"
    fx = ep.get("fixtures", {}); fam_dst = ep["family"].split("->")[1]; dkey = {"microwave": "microwave", "sink": "sink", "cabinet": "cab"}[fam_dst]
    for n, v in fx.items():
        if dkey in n.lower() or dkey in v.get("cls", "").lower():
            continue
        if not any(k in (n + v.get("cls", "")).lower() for k in RKEYS):
            continue
        half = np.maximum(np.asarray(v["size"][:2]) / 2, 0.0) + 0.05
        if np.all(np.abs(pos[-1, i, :2] - np.asarray(v["pos"][:2])) < half):
            return "W"
    return "O"


def cls_rt(ep, npz):
    if ep["success"]:
        return "S"
    t = np.asarray(npz["track"])
    if ep.get("error") or t.size == 0:
        return "X"
    z = t[:, 3]; z0 = ep["reset_pos"]["A"][2]
    if z.max() - z0 < 0.02:
        return "G"
    return "H" if z[-1] - z0 >= 0.02 else "O"


def main():
    out = f"{XB}/classes.json"
    assert not os.path.exists(out), "never overwrite"
    cells = {b: json.load(open(f"{XB}/cells_{b}.json")) for b in ("mesa", "bimesa")}
    res = {}
    for bench in ("mesa", "bimesa", "robocasa", "robotwin"):
        for mdir in sorted(glob.glob(f"{XB}/roll/{bench}/*")):
            m = os.path.basename(mdir); cnt = {"heldout": {}, "partner": {}}
            for jf in glob.glob(f"{mdir}/**/ep*.json", recursive=True):
                ep = json.load(open(jf)); nz = jf[:-5] + ".npz"
                if not os.path.exists(nz):
                    continue
                npz = np.load(nz, allow_pickle=True)
                try:
                    if bench in ("mesa", "bimesa"):
                        info = {**cells[bench]["heldout_info"], **cells[bench]["partner_info"]}[ep["task"]]
                        c = cls_mesa(ep, npz, info)
                    elif bench == "robocasa":
                        c = cls_rc(ep, npz)
                    else:
                        c = cls_rt(ep, npz)
                except Exception:
                    c = "X"
                ct = ep["cell_type"]; cnt[ct][c] = cnt[ct].get(c, 0) + 1
            res[f"{bench}/{m}"] = {ct: dict(counts=d, shares={k: v / max(1, sum(d.values())) for k, v in d.items()}) for ct, d in cnt.items()}
    with open(out, "x") as fh:
        json.dump(res, fh, indent=1)
    for k, v in res.items():
        print(k, {ct: {c: round(s, 3) for c, s in d["shares"].items()} for ct, d in v.items()})


if __name__ == "__main__":
    main()
