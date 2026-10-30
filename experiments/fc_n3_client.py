"""FLAWED_COMPOSITION N3 (section 4.12): experiments/libero_client.py, unchanged, on the N3 tasks.  As fc_n2_client.py: a suite
named n3_spatial in LIBERO's benchmark dictionary whose tasks are the generated BDDLs of results/flawed/n3_tasks.json, start states
only from --fresh, every episode through the client's branch path (a fresh env, re-seeded, per episode).

One addition, client-side only (nothing is sent to the policy server): the env class the client builds is wrapped so that, when an
env that has stepped is closed (the branch path closes it after every episode), the episode's start and final positions of every
movable object, the final positions of the scene's targets (plate body, stove cook-region site, cabinet top-side site), the final
gripper opening and the number of steps are appended, in episode order, to <out without .json>_pos.json.  fc_n3_read.py aligns these
records with the client's episodes by order (asserted: same count, same step counts).
LIBERO venv (py3.8).

    PYTHONPATH=.../third_party/libero .../examples/libero/.venv/bin/python experiments/fc_n3_client.py --port P --suite n3_spatial \
        --tasks 3 --fresh results/flawed/n3_states --branches results/flawed/n3_branches_main.json --bind track --out X.json
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
from libero.libero import benchmark                                        # noqa: E402

N3 = json.load(open(os.path.join(ROOT, "results", "flawed", "n3_tasks.json"), encoding="utf-8"))
Task = collections.namedtuple("Task", "name language problem_folder bddl_file")
SITES = ("flat_stove_1_cook_region", "wooden_cabinet_1_top_side")
POS_LOG = []


class N3Suite:
    def __init__(self):
        self.tasks = [Task(t["key"], t["language"], os.path.dirname(os.path.join(ROOT, t["bddl"])), os.path.basename(t["bddl"]))
                      for t in N3["tasks"]]
        self.n_tasks = len(self.tasks)

    def get_task(self, i):
        return self.tasks[i]

    def get_task_init_states(self, i):
        raise RuntimeError("N3 runs only from --fresh start states")


_orig = benchmark.get_benchmark_dict


def _bdict(*a, **k):
    d = dict(_orig(*a, **k))
    d[N3["suite"]] = N3Suite
    return d


def _out_path():
    i = sys.argv.index("--out")
    return os.path.splitext(sys.argv[i + 1])[0] + "_pos.json"


class RecEnv(LC.OffScreenRenderEnv):
    """The client's env; records object and target positions at episode end (client side only)."""

    def _snap(self):
        inner = self.env
        sim = inner.sim
        objs = {n: np.array(sim.data.body_xpos[inner.obj_body_id[n]]).round(5).tolist() for n in inner.objects_dict}
        tg = {}
        for s in SITES:
            try:
                tg[s] = np.array(sim.data.site_xpos[sim.model.site_name2id(s)]).round(5).tolist()
            except Exception:                                              # noqa: BLE001
                pass
        return objs, tg

    def set_init_state(self, s):
        o = super().set_init_state(s)
        self._n3_start, _ = self._snap()
        self._n3_steps = 0
        return o

    def step(self, act):
        r = super().step(act)
        self._n3_steps = getattr(self, "_n3_steps", 0) + 1
        return r

    def close(self):
        if getattr(self, "_n3_steps", 0) > 0:
            objs, tg = self._snap()
            try:
                grip = np.array(self.env.sim.data.qpos[self.env.robots[0]._ref_gripper_joint_pos_indexes]).round(5).tolist()
            except Exception:                                              # noqa: BLE001
                grip = None
            POS_LOG.append(dict(start=self._n3_start, final=objs, sites=tg, gripper_qpos=grip, sim_steps=int(self._n3_steps)))
            with open(_out_path(), "w", encoding="utf-8") as fh:
                json.dump(POS_LOG, fh)
            self._n3_steps = 0
        return super().close()


benchmark.get_benchmark_dict = _bdict
LC.OffScreenRenderEnv = RecEnv
LC.MAX_STEPS[N3["suite"]] = int(N3["max_steps"])

if __name__ == "__main__":
    assert "--fresh" in sys.argv and "--branches" in sys.argv and "--out" in sys.argv, "N3 episodes need --fresh, --branches, --out"
    LC.main()
