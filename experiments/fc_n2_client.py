"""FLAWED_COMPOSITION N2 (section 4.11): experiments/libero_client.py, unchanged, on the N2 tasks.  The only addition is a suite
named n2_goal in LIBERO's benchmark dictionary whose tasks are the generated BDDLs of results/flawed/n2_tasks.json (prompt = the
task's LIBERO file-name instruction; the BDDL path is absolute, so the client's bddl_files join returns it).  Start states come only
from --fresh (results/flawed/n2_states); each episode runs through the client's branch path (a fresh env, re-seeded, per episode).
LIBERO venv (py3.8).

    PYTHONPATH=.../third_party/libero .../examples/libero/.venv/bin/python experiments/fc_n2_client.py --port P --suite n2_goal \
        --tasks 3 --fresh results/flawed/n2_states --branches results/flawed/n2_branches_main.json --bind track --out X.json
"""
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import libero_client as LC                                                 # noqa: E402
from libero.libero import benchmark                                        # noqa: E402

N2 = json.load(open(os.path.join(ROOT, "results", "flawed", "n2_tasks.json"), encoding="utf-8"))
Task = collections.namedtuple("Task", "name language problem_folder bddl_file")


class N2Suite:
    def __init__(self):
        self.tasks = [Task(t["key"], t["language"], os.path.dirname(os.path.join(ROOT, t["bddl"])), os.path.basename(t["bddl"]))
                      for t in N2["tasks"]]
        self.n_tasks = len(self.tasks)

    def get_task(self, i):
        return self.tasks[i]

    def get_task_init_states(self, i):
        raise RuntimeError("N2 runs only from --fresh start states")


_orig = benchmark.get_benchmark_dict


def _bdict(*a, **k):
    d = dict(_orig(*a, **k))
    d[N2["suite"]] = N2Suite
    return d


benchmark.get_benchmark_dict = _bdict
LC.MAX_STEPS[N2["suite"]] = int(N2["max_steps"])

if __name__ == "__main__":
    assert "--fresh" in sys.argv and "--branches" in sys.argv, "N2 episodes need --fresh and --branches"
    LC.main()
