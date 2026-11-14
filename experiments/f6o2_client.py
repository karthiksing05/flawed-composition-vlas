"""F6O2 (experiments/f6o2_tasks.py): f6o_client.py's harness on the f6o2 suite (results/flawed/f6o2/f6o2_tasks.json; prompt
"put the bowl on the plate"; fx's F6 BDDL; rows 0-23 = fx's disp states, rows 24.. = the new states), one task per process.
experiments/libero_client.py is unchanged in what it sends and how it rolls out.  f6o_client.py executes on import (it parses
its own task), so its F6oEnv is restated here for the f6o2 table; the track is the same.  Additions inside the env object:
  - the covers (f6o_tasks.install) placed per the task's condition by f6o_tasks.CoverEnv (a, b, c, d as f6o);
  - task M: condition b through f6o2_tasks.MemEnv, the memory overlay painted into both cameras from the move on; per episode
    the overlay's record (segmenter mask vs the true plate, centre error) and painted pixels at each query -> <out>_mem.json.
  - the track, as f6o_client's -> <out>_track.npz.
Start states only from --fresh (results/flawed/f6o2/states), episodes only through --branches.  LIBERO venv (py3.8).

    PYTHONPATH=.../third_party/libero .../examples/libero/.venv/bin/python experiments/f6o2_client.py --port P --suite f6o2 --tasks 4 \
        --trials 1 --fresh results/flawed/f6o2/states --branches results/flawed/f6o2/branches_main_s0.json --videos 0 --bind track --out X.json
"""
import collections
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import libero_client as LC                                                 # noqa: E402
import f6o_tasks as OT                                                     # noqa: E402
import f6o2_tasks as O2                                                    # noqa: E402
from libero.libero import benchmark                                        # noqa: E402

F6O2 = json.load(open(O2.META, encoding="utf-8"))
Task = collections.namedtuple("Task", "name language problem_folder bddl_file")


class F6o2Suite:
    def __init__(self):
        self.tasks = [Task(t["key"], t["language"], os.path.dirname(os.path.join(ROOT, t["bddl"])), os.path.basename(t["bddl"]))
                      for t in F6O2["tasks"]]
        self.n_tasks = len(self.tasks)

    def get_task(self, i):
        return self.tasks[i]

    def get_task_init_states(self, i):
        raise RuntimeError("f6o2 runs only from --fresh start states")


_orig = benchmark.get_benchmark_dict


def _bdict(*a, **k):
    d = dict(_orig(*a, **k))
    d[F6O2["suite"]] = F6o2Suite
    return d


benchmark.get_benchmark_dict = _bdict
LC.MAX_STEPS[F6O2["suite"]] = 300


def _arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


TID = int(_arg("--tasks"))
OUTF = _arg("--out")
TASK = F6O2["tasks"][TID]
assert TASK["tid"] == TID
assert int(_arg("--wait", F6O2["wait"])) == OT.WAIT and int(_arg("--replan", 5)) == OT.REPLAN, "f6o2 assumes the client's wait / replan"
CV = F6O2["cover"]
OT.install(CV["radius"], CV["half_h"], CV["n"])
STATES = np.load(os.path.join(O2.ST, f"{O2.SUITE}_{TID}.npy"))
ORIG, RECS, MEMS = LC.OffScreenRenderEnv, [], []
MEM = TASK["arm"] == "M"
BASE = O2.MemEnv if MEM else OT.CoverEnv


class F6o2Env(BASE):
    """CoverEnv (or MemEnv for M) plus f6o_client's track."""

    def __init__(self, env):
        if MEM:
            BASE.__init__(self, env, F6O2["parks"], CV["zc"], STATES, track=True)
        else:
            BASE.__init__(self, env, TASK["cond"], F6O2["parks"], CV["zc"], STATES, track=True, beside=F6O2["beside"])
        self.tr = None

    def _names(self):
        return list(self._e.env.objects_dict.keys())

    def _snap(self):
        inner = self._e.env
        return np.array([np.array(inner.sim.data.body_xpos[inner.obj_body_id[n]]) for n in self._names()], np.float32)

    def set_init_state(self, s):
        obs = super().set_init_state(s)
        self.tr = dict(start=self._snap(), pos=[], ee=[], a6=[])
        return obs

    def step(self, a):
        r = super().step(a)
        if self.tr is not None:
            self.tr["pos"].append(self._snap())
            self.tr["ee"].append(np.asarray(r[0]["robot0_eef_pos"], np.float32))
            self.tr["a6"].append(float(a[6]))
        return r

    def close(self):
        if self.tr is not None and self.n > 0:
            R, C = self.tr, self.rows
            RECS.append(dict(start=R["start"], pos=np.stack(R["pos"]), ee=np.stack(R["ee"]), a6=np.array(R["a6"], np.float32),
                             hidden=np.array(C["covered"], bool), plate_px=np.array(C["px"], np.int32), k=np.array(C["k"]),
                             success=np.array(bool(self._e.env._check_success()))))
            z = dict(n=np.array(len(RECS)), names=np.array(self._names()), wait=np.array(OT.WAIT))
            for i, r in enumerate(RECS):
                z.update({f"e{i}_{k}": v for k, v in r.items()})
            np.savez_compressed(OUTF.replace(".json", "_track.npz"), **z)
            if MEM and self.mem is not None:
                MEMS.append(dict(k=int(C["k"]), steps=int(self.n), **self.mem.rec))
                side = OUTF.replace(".json", "_mem.json")
                with open(side + ".tmp", "w", encoding="utf-8") as fh:
                    json.dump(dict(tid=TID, key=TASK["key"], mem_h=O2.MEM_H, eps=MEMS), fh, indent=1)
                os.replace(side + ".tmp", side)
                rec = self.mem.rec["record"] or {}
                print(f"[f6o2] M row {C['k']}: record ok={rec.get('ok')} seg IoU {rec.get('iou')} centre err {rec.get('centre_xy_err')} "
                      f"painted (agent, wrist) at queries {[t[1:] for t in self.mem.rec['trace'][:6]]}", flush=True)
            self.tr = None
        self._e.close()


LC.OffScreenRenderEnv = lambda *a, **k: F6o2Env(ORIG(*a, **k))

if __name__ == "__main__":
    assert "--fresh" in sys.argv and "--branches" in sys.argv, "f6o2 episodes need --fresh and --branches"
    assert _arg("--suite") == F6O2["suite"] and "," not in _arg("--tasks"), "one f6o2 task per process"
    if "--max-steps" not in sys.argv:
        sys.argv += ["--max-steps", str(TASK["max_steps"])]
    LC.main()
