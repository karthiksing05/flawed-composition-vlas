"""Language-sensitivity probe client (docs/notes/fp_dev_results.md, v2).  LIBERO venv (py3.8).  No rollout: libero_client.py's own env reset,
init state, settle steps and first-query observation (so the policy sees exactly what the atlas rollouts' first query saw), then, instead of
acting, FOUR kinds of queries to the already running frozen server with the same branch noise (meta/seed_base):
  base   the instructed language
  alt_j  each trained instruction that names a DIFFERENT target for the same object (scene 1: the known K tasks of the object;
         scene 2: the other plate for the same mug)
  floor  the instructed language, a different noise seed (seed_base + 7919)
  rep    the instructed language, the same seed again (determinism check)
and the episode is ended after that first query.  One json line per (task, trial) into $FP_OUT.
    python experiments/fp_probe_client.py --scene 1|2 <libero_client args: --port --suite --tasks fam|<ids> --fresh --branches --out ...>
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
i = sys.argv.index("--scene")
SCENE = int(sys.argv[i + 1])
del sys.argv[i:i + 2]
OUTF = os.environ["FP_OUT"]

if SCENE == 1:
    import fc_n2_client as FN
    LC = FN.LC
    TASKS = FN.N2["tasks"]
    ATLAS = ["K1", "K2", "K3", "K4", "K6", "N1", "N2", "N3", "N4", "N5", "N6"]
    FAM = {t["compound"]: t for t in TASKS if t["layout"] == "familiar"}
    LANG = {FAM[c]["language"]: FAM[c] for c in ATLAS}
    KNOWN = [t for t in TASKS if t["layout"] == "familiar" and t["status"] == "known" and t["compound"] in ("K1", "K2", "K3", "K4", "K5", "K6")]       # K1..K6 as in fp_build.TRAINED
    ALT = {}
    for lg, t in LANG.items():
        ALT[lg] = [k["language"] for k in KNOWN if k["object"] == t["object"] and k["target"] != t["target"]]
    FAMIDS = ",".join(str(FAM[c]["tid"]) for c in ATLAS)
else:
    import s2 as S2
    import libero_client as LC
    M = S2.register_suite(LC)
    T = {t["key"]: t for t in M["tasks"][:4]}
    LANG = {t["language"]: t for t in T.values()}
    ALT = {}
    for lg, t in LANG.items():
        ALT[lg] = [u["language"] for u in T.values() if u["object"] == t["object"] and u["target"] != t["target"]]
    FAMIDS = "0,1,2,3"


class Stop(Exception):
    pass


Base = LC._wcp.WebsocketClientPolicy


class Probe(Base):
    def infer(self, el):
        p = str(el["prompt"])
        sb = int(el["meta/seed_base"])

        def q(prompt, off=0):
            e = dict(el)
            e["prompt"] = prompt
            e["meta/seed_base"] = sb + off
            return np.asarray(Base.infer(self, e)["actions"], np.float64)
        base = q(p)
        rep = q(p)
        floor = q(p, 7919)
        alts = [q(a) for a in ALT.get(p, [])]
        row = dict(task_id=int(el["meta/task_id"]), trial=int(el["meta/episode"]), prompt=p, alts=ALT.get(p, []), shape=list(base.shape),
                   base=base[:10].tolist(), floor=floor[:10].tolist(), rep_maxabs=float(np.abs(rep - base).max()), alt_list=[a[:10].tolist() for a in alts])
        # distances computed offline from the saved first 10 steps and the full-chunk statistics below
        def dist(a, b, sl):
            d = np.linalg.norm(a[sl] - b[sl], axis=-1).mean()
            n = np.linalg.norm(b[sl], axis=-1).mean()
            return float(d / max(n, 1e-8))
        for nm, sl in (("full", slice(None)), ("exec5", slice(0, 5))):
            row["d_" + nm] = [dist(a, base, sl) for a in alts]
            row["floor_" + nm] = dist(floor, base, sl)
        row["d_xyz5"] = [float(np.linalg.norm(a[:5, :3] - base[:5, :3], axis=-1).mean() / max(np.linalg.norm(base[:5, :3], axis=-1).mean(), 1e-8)) for a in alts]
        row["floor_xyz5"] = float(np.linalg.norm(floor[:5, :3] - base[:5, :3], axis=-1).mean() / max(np.linalg.norm(base[:5, :3], axis=-1).mean(), 1e-8))
        with open(OUTF, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        raise Stop()


LC._wcp.WebsocketClientPolicy = Probe
if "--tasks" in sys.argv:
    j = sys.argv.index("--tasks")
    if sys.argv[j + 1] == "fam":
        sys.argv[j + 1] = FAMIDS
if __name__ == "__main__":
    LC.main()
