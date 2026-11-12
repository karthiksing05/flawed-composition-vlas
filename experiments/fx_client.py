"""FX diagnostics (experiments/fx_tasks.py): experiments/libero_client.py, unchanged in what it sends and how it rolls out, on the fx
suite (results/flawed/fx/fx_tasks.json; prompt = the task's language; absolute BDDL paths), one task per process.  Two additions,
both inside the env object the client builds:
  - F6's hide: from env step WAIT + hide_at on (the step whose observation feeds policy step hide_at), every geom of the task's
    `hide` object sits in an unrendered geom group, in both cameras.  Physics and the success test are unchanged.
  - a track: every movable object's position, the end effector and the commanded gripper after every env step (settle steps
    included) -> <out>_track.npz, as experiments/pp_atlas_client.py saves its tracks.
Start states only from --fresh (results/flawed/fx/states), episodes only through --branches (a fresh, re-seeded env per episode).
LIBERO venv (py3.8).

    PYTHONPATH=.../third_party/libero .../examples/libero/.venv/bin/python experiments/fx_client.py --port P --suite fx --tasks 3 \
        --trials 1 --fresh results/flawed/fx/states --branches results/flawed/fx/fx_branches_main.json --videos 0 --bind track --out X.json
"""
import collections
import os
import sys
import json

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import libero_client as LC                                                 # noqa: E402
import fx_tasks as FT                                                      # noqa: E402
from libero.libero import benchmark                                        # noqa: E402

FX = json.load(open(FT.META, encoding="utf-8"))
Task = collections.namedtuple("Task", "name language problem_folder bddl_file")


class FxSuite:
    def __init__(self):
        self.tasks = [Task(t["key"], t["language"], os.path.dirname(os.path.join(ROOT, t["bddl"])), os.path.basename(t["bddl"]))
                      for t in FX["tasks"]]
        self.n_tasks = len(self.tasks)

    def get_task(self, i):
        return self.tasks[i]

    def get_task_init_states(self, i):
        raise RuntimeError("fx runs only from --fresh start states")


_orig = benchmark.get_benchmark_dict


def _bdict(*a, **k):
    d = dict(_orig(*a, **k))
    d[FX["suite"]] = FxSuite
    return d


benchmark.get_benchmark_dict = _bdict
LC.MAX_STEPS[FX["suite"]] = 300


def _arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


TID = int(_arg("--tasks"))
OUTF = _arg("--out")
WAIT = int(_arg("--wait", FX["wait"]))
TASK = FX["tasks"][TID]
assert TASK["tid"] == TID
ORIG, RECS = LC.OffScreenRenderEnv, []


class FxEnv:
    """The client's env with F6's hide and the track.  Nothing else the policy sees changes."""

    def __init__(self, env):
        self._e, self.n, self.hidden, self.rows = env, 0, False, None

    def __getattr__(self, k):
        return getattr(self._e, k)

    def _names(self):
        return list(self._e.env.objects_dict.keys())

    def _snap(self):
        inner = self._e.env
        return np.array([np.array(inner.sim.data.body_xpos[inner.obj_body_id[n]]) for n in self._names()], np.float32)

    def set_init_state(self, s):
        obs = self._e.set_init_state(s)
        self.n, self.hidden = 0, False
        self.rows = dict(start=self._snap(), pos=[], ee=[], a6=[], hidden=[])
        return obs

    def step(self, a):
        self.n += 1
        if TASK["hide"] and not self.hidden and self.n >= WAIT + int(TASK["hide_at"]):
            FT.hide_geoms(self._e, TASK["hide"])
            self.hidden = True
        r = self._e.step(a)
        if self.rows is not None:
            R = self.rows
            R["pos"].append(self._snap())
            R["ee"].append(np.asarray(r[0]["robot0_eef_pos"], np.float32))
            R["a6"].append(float(a[6]))
            R["hidden"].append(self.hidden)
        return r

    def close(self):
        if self.rows is not None and self.n > 0:
            R = self.rows
            RECS.append(dict(start=R["start"], pos=np.stack(R["pos"]), ee=np.stack(R["ee"]), a6=np.array(R["a6"], np.float32),
                             hidden=np.array(R["hidden"], bool), success=np.array(bool(self._e.env._check_success()))))
            a = dict(n=np.array(len(RECS)), names=np.array(self._names()), wait=np.array(WAIT))
            for i, r in enumerate(RECS):
                a.update({f"e{i}_{k}": v for k, v in r.items()})
            np.savez_compressed(OUTF.replace(".json", "_track.npz"), **a)
            self.rows = None
        self._e.close()


LC.OffScreenRenderEnv = lambda *a, **k: FxEnv(ORIG(*a, **k))

if __name__ == "__main__":
    assert "--fresh" in sys.argv and "--branches" in sys.argv, "fx episodes need --fresh and --branches"
    assert _arg("--suite") == FX["suite"] and "," not in _arg("--tasks"), "one fx task per process"
    if "--max-steps" not in sys.argv:
        sys.argv += ["--max-steps", str(TASK["max_steps"])]
    LC.main()
