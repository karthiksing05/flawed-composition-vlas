"""F6O (experiments/f6o_tasks.py): experiments/libero_client.py, unchanged in what it sends and how it rolls out, on the f6o suite
(results/flawed/f6o/f6o_tasks.json; prompt "put the bowl on the plate"; fx's F6 BDDL and start states), one task per process.
Additions, all inside the env object the client builds:
  - the two covers are in the model (f6o_tasks.install, before the client builds any env) and placed per the task's condition by
    f6o_tasks.CoverEnv (a: both parked; b: cover 0 moves onto the plate at env step WAIT + HIDE_AT; c: cover 0 on the plate from
    the start; d: cover 0 moves at b's step to a spot beside the plate that leaves it visible).  Physics and the success test are
    fx's.
  - a track, as fx_client's: every movable object's position, the end effector and the commanded gripper after every env step,
    plus whether cover 0 is on the plate and, at every step whose observation feeds a policy query, the plate's segmentation
    pixels in agentview and wrist (-1 elsewhere) -> <out>_track.npz.
Start states only from --fresh (results/flawed/f6o/states), episodes only through --branches.  LIBERO venv (py3.8).

    PYTHONPATH=.../third_party/libero .../examples/libero/.venv/bin/python experiments/f6o_client.py --port P --suite f6o --tasks 3 \
        --trials 1 --fresh results/flawed/f6o/states --branches results/flawed/fx/fx_branches_main.json --videos 0 --bind track --out X.json
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
from libero.libero import benchmark                                        # noqa: E402

F6O = json.load(open(OT.META, encoding="utf-8"))
Task = collections.namedtuple("Task", "name language problem_folder bddl_file")


class F6oSuite:
    def __init__(self):
        self.tasks = [Task(t["key"], t["language"], os.path.dirname(os.path.join(ROOT, t["bddl"])), os.path.basename(t["bddl"]))
                      for t in F6O["tasks"]]
        self.n_tasks = len(self.tasks)

    def get_task(self, i):
        return self.tasks[i]

    def get_task_init_states(self, i):
        raise RuntimeError("f6o runs only from --fresh start states")


_orig = benchmark.get_benchmark_dict


def _bdict(*a, **k):
    d = dict(_orig(*a, **k))
    d[F6O["suite"]] = F6oSuite
    return d


benchmark.get_benchmark_dict = _bdict
LC.MAX_STEPS[F6O["suite"]] = 300


def _arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


TID = int(_arg("--tasks"))
OUTF = _arg("--out")
TASK = F6O["tasks"][TID]
assert TASK["tid"] == TID
assert int(_arg("--wait", F6O["wait"])) == OT.WAIT and int(_arg("--replan", 5)) == OT.REPLAN, "f6o assumes the client's wait / replan"
CV = F6O["cover"]
OT.install(CV["radius"], CV["half_h"], CV["n"])
STATES = np.load(os.path.join(OT.ST, f"{OT.SUITE}_{TID}.npy"))
ORIG, RECS = LC.OffScreenRenderEnv, []


class F6oEnv(OT.CoverEnv):
    """CoverEnv plus fx_client's track."""

    def __init__(self, env):
        super().__init__(env, TASK["cond"], F6O["parks"][TASK["layout"]], CV["zc"], STATES, track=True,
                         beside=F6O["beside"][TASK["layout"]])
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
            self.tr = None
        self._e.close()


LC.OffScreenRenderEnv = lambda *a, **k: F6oEnv(ORIG(*a, **k))

if __name__ == "__main__":
    assert "--fresh" in sys.argv and "--branches" in sys.argv, "f6o episodes need --fresh and --branches"
    assert _arg("--suite") == F6O["suite"] and "," not in _arg("--tasks"), "one f6o task per process"
    if "--max-steps" not in sys.argv:
        sys.argv += ["--max-steps", str(TASK["max_steps"])]
    LC.main()
