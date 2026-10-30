"""FLAWED_COMPOSITION N1 (section 4.9): the task list whose initial scenes fc_n1_render.py renders.  No model, no outcome.  CPU.

  BindVLA blocks: the 76 LIBERO-90-C training tasks and the 29 dev tasks (l90c.load_split; split_sealed.json is never opened)
  D2 blocks:      the 40 LIBERO training tasks (factors.TRAIN_SUITES) and the unsealed LIBERO-90 tasks of results/flawed/d2_frames.json
Sealed LIBERO-90-C tasks are only EXCLUDED (l90c.is_sealed on the salted hashes of split.json); their identities are used for
nothing else.  Li's 20 OOD tasks are rendered separately (fc_n1_render.py --li).
-> results/flawed/n1_tasks.json

    python experiments/fc_n1_tasklist.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments")]


def main():
    from lifelong_ood import l90c
    from lifelong_ood.policy import factors as F
    sp = l90c.load_split()
    dev = [t for b in ("0.1", "0.3", "0.5", "part") for t in sp["dev"][b]]
    fr = json.load(open(os.path.join(ROOT, "results", "flawed", "d2_frames.json"), encoding="utf-8"))
    lib40 = [t.tid for t in F.load(F.TRAIN_SUITES)]
    allt = sorted(set(sp["train"]) | set(dev) | set(lib40) | set(fr["l90"]))
    keep = [t for t in allt if not (t.startswith("libero_90:") and l90c.is_sealed(t))]
    out = dict(tasks=keep, n=len(keep), bv_train=sp["train"], dev=dev, lib40=lib40, l90=sorted(fr["l90"]),
               n_excluded_sealed=len(allt) - len(keep))
    with open(os.path.join(ROOT, "results", "flawed", "n1_tasks.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(out["n"], "tasks;", out["n_excluded_sealed"], "excluded")


if __name__ == "__main__":
    main()
