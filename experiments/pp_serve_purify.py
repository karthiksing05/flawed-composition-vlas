"""PLAN_PARTS F3 closed loop (docs/PARTS_RESULTS.md section 1): serve pi0.5 or GR00T through lifelong_ood/policy/serve.GuidedPolicy
(the sampler and seeding N2's server used, serve_guided.py mode ours), with the located component purified on every query of a new
pairing (experiments/pp_purify.py).  --mode intact serves the frozen policy unchanged (the paired re-run arm).

    python experiments/pp_serve_purify.py --port P --model pi05|groot --mode intact|purify --comp KV_3 --seed 13
"""
import argparse
import logging
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE), "/coc/flash10/ksingara3/robot/openpi/src"]


class PurifyHooks:
    def __init__(self, M, comp):
        from pp_f1_mix import tasks_pp
        self.M, self.comp, self.T = M, comp, tasks_pp()
        self.by_text = {v["text"]: c for c, v in self.T.items()}
        self.obs = None
        self.n = 0

    def rewrite(self, obs, meta):
        self.obs = obs
        return obs

    def transform_prefix(self, pre):
        import pp_purify as PU
        text = str(self.obs["prompt"])
        c = self.by_text.get(text)
        if c is None or c not in PU.NEW:
            return pre
        fr = {"img": [self.obs["observation/image"]], "wimg": [self.obs["observation/wrist_image"]],
              "state8": [self.obs["observation/state"]]}
        _, pp = PU.purified_prefix(self.M, fr, [0], self.T, c, self.comp)
        self.n += 1
        return pp

    def __call__(self, obs, pre, meta):
        return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", required=True, choices=("pi05", "groot"))
    ap.add_argument("--mode", required=True, choices=("intact", "purify"))
    ap.add_argument("--comp", default="")
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    from lifelong_ood.policy.serve import GuidedPolicy, serve, warm
    from pp_f1_mix import Groot, Pi05
    M = Pi05() if a.model == "pi05" else Groot()
    hooks = PurifyHooks(M, a.comp) if a.mode == "purify" else None
    policy = GuidedPolicy(M.bk, hooks=hooks, num_steps=M.bk.num_steps, seed=a.seed)
    logging.info(f"pp_serve_purify: {M.bk.describe()} mode={a.mode} comp={a.comp}")
    warm(policy)
    serve(policy, a.port)


if __name__ == "__main__":
    main()
