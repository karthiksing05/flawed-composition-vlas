"""SR 90 (docs/notes/raasw6_plan.md): the target latch of W6 (where_own 8; W6nl = 9 has none).  Lives in the SERVER, where the channel is built (the held-target channel is
rel = target_estimate - eef - offset; the latch replaces target_estimate, so the expert and the module both read it).
Rule: at carry start (the first query of an episode's carry: role = target after a query that was not) the target estimate is latched.  A later estimate replaces the latched one only if it moves
less than tau from it (tau = the p99 5-step change of the target's pose over the training demos, lifelong_ood/policy/raasw6_env.json "tau_target_m"); a larger jump is rejected and the latched value
stays (the target is static unless pushed).  Not held: nothing latched (the estimate passes through).  No other state; reset at every episode (ep id) and at every release / non-target query.
"""
import json
import os

import numpy as np

ENV_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raasw6_env.json")


class TargetLatch:
    def __init__(self, tau=None):
        self.tau = float(json.load(open(ENV_JSON, encoding="utf-8"))["tau_target_m"]) if tau is None else float(tau)
        self.ep, self.val, self.n_acc, self.n_rej, self.n_start = None, None, 0, 0, 0
        self.last = dict(start=False, rejected=False, jump=0.0)

    def apply(self, ep, on, tgt):
        tgt = np.asarray(tgt, np.float64)
        self.last = dict(start=False, rejected=False, jump=0.0)
        if not on:
            self.ep, self.val = ep, None
            return tgt
        if self.val is None or ep != self.ep:
            self.ep, self.val = ep, tgt.copy()
            self.n_start += 1
            self.last["start"] = True
            return self.val
        d = float(np.linalg.norm(tgt - self.val))
        self.last["jump"] = d
        if d <= self.tau:
            self.val = tgt.copy()
            self.n_acc += 1
        else:
            self.n_rej += 1
            self.last["rejected"] = True
        return self.val
