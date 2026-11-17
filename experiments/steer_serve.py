"""SR 74 steerability probe, server side: runs any of the repo's serve scripts unchanged, with the served policy wrapped so that SIDE queries
(meta/side) never change anything a later real query sees.

    python experiments/steer_serve.py <serve script> <its arguments>
    e.g.  steer_serve.py experiments/pp_serve_purify.py --port P --model pi05 --mode intact --comp= --seed 13

A side query is answered by the unchanged policy on the counterfactual observation, and afterwards every piece of state the query could
have advanced is put back: GuidedPolicy's call counter and query index (its noise seed is seed + calls, or a branch's seed_base + query,
so the real query that follows draws the SAME noise as the side query: a side chunk and the real chunk differ only by the instruction and
what the front end does with it), the role sequencer's ProprioHeld and previous-command state (sc_rv.RoleView, sc_rv_w.WristView, sc_seg
LearnedMasks wrap it), and the wrist gate.  The executed episode is therefore the one the unprobed server would have run.

Role-phrase servers (rsup / rview2 / rbox / rwrist rungs) look the instruction up in their task tables (sc_rv.roles_of, sc_t16.given).  The
probe asks counterfactual instructions that are not tasks, so at start the tables get one entry per (object, target) composition of
steer_text.SCENES[1].  A side query that carries an absent or nonsense word (meta/side_slot, meta/side_phrase) keeps the true
instruction for routing and painting (the front end has no slot for an unknown entity) and the served role phrase of the named slot is
replaced by the word.  Plain servers (frozen policies, R0) get the full counterfactual text from the client and need none of this.
"""
import copy
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, "/coc/flash10/ksingara3/robot/openpi/src"]

CHILD = ("p", "inner")                                  # attribute names that hold the wrapped policy (RoleView.p, LearnedMasks.inner)
STATE = ("det", "prev", "checked", "calls", "ep_q", "branch", "secs")


def chain(policy):
    out, seen, x = [], set(), policy
    while x is not None and id(x) not in seen:
        seen.add(id(x))
        out.append(x)
        nxt = None
        for a in CHILD:
            c = getattr(x, a, None)
            if c is not None and not isinstance(c, (int, float, str, dict, list)):
                nxt = c
                break
        x = nxt
    return out


class Override:
    """Sits just above the GuidedPolicy: replaces the served prompt's slot phrase when a side query asks for it."""

    def __init__(self, inner):
        self.inner, self.slot, self.phrase, self.hit, self.miss = inner, None, None, 0, 0
        self.metadata = getattr(inner, "metadata", {})

    def infer(self, o):
        if self.slot is not None:
            o = dict(o)
            p = str(o["prompt"])
            if p.startswith("object: ") or p.startswith("target: "):          # a role-phrase prompt: keep the slot's prefix
                pre = "object: " if self.slot == "object" else "target: "
                if p.startswith(pre):
                    o["prompt"] = pre + self.phrase
                    self.hit += 1
                else:
                    self.miss += 1                                            # the server's phase is not the client's: not overridden
                import logging
                logging.info(f"[steer] override slot={self.slot} phrase={self.phrase!r} served prompt {p!r} -> {o['prompt']!r} "
                             f"(hit {self.hit} miss {self.miss})")
        return self.inner.infer(o)

    def __getattr__(self, k):
        return getattr(self.inner, k)


class SideWrap:
    def __init__(self, policy):
        self.policy = policy
        self.metadata = getattr(policy, "metadata", {})
        self.nodes = chain(policy)
        self.override = None
        for n in self.nodes:                                                  # role-phrase servers: the composition tables
            if hasattr(n, "roles") and hasattr(n, "T16"):
                self.extend(n)
        for par in self.nodes:                                                # the Override above the GuidedPolicy, if any wrapper sits above it
            for a in CHILD:
                c = getattr(par, a, None)
                if c is not None and hasattr(c, "bk"):
                    self.override = Override(c)
                    setattr(par, a, self.override)
                    break
            if self.override is not None:
                break
        self.n_side = 0

    @staticmethod
    def extend(rv):
        import steer_text as ST
        rv.T16.given("put the bowl on the plate")                             # fills sc_t16._GIVEN
        S = ST.SCENES[1]
        for o in S["obj"]:
            for t in S["tgt"]:
                if o == t:
                    continue
                text = ST.lang(1, o, t)
                rv.roles.setdefault(text, (o, t))
                rv.T16._GIVEN.setdefault(text, ("object: " + S["phrase"][o], "target: " + S["phrase"][t]))

    def snap(self):
        out = []
        for n in self.nodes:
            d = {}
            for k in STATE:
                if k in n.__dict__:
                    d[k] = copy.deepcopy(n.__dict__[k])
            out.append(d)
        return out

    def restore(self, snap):
        for n, d in zip(self.nodes, snap):
            for k, v in d.items():
                n.__dict__[k] = v

    def infer(self, obs):
        if not obs.get("meta/side"):
            return self.policy.infer(obs)
        o = dict(obs)
        o.pop("meta/side", None)
        slot, phrase = o.pop("meta/side_slot", None), o.pop("meta/side_phrase", None)
        if slot is not None:
            slot, phrase = str(slot), str(phrase)
        snap = self.snap()
        if self.override is not None:
            self.override.slot, self.override.phrase = slot, phrase
        try:
            r = self.policy.infer(o)
        finally:
            if self.override is not None:
                self.override.slot, self.override.phrase = None, None
            self.restore(snap)
        self.n_side += 1
        return r

    def __getattr__(self, k):
        if k in ("policy", "nodes"):
            raise AttributeError(k)
        return getattr(self.policy, k)


def main():
    target = sys.argv[1]
    sys.argv = [target] + sys.argv[2:]
    import lifelong_ood.policy.serve as S
    orig = S.serve

    def serve(policy, port, *a, **k):
        return orig(SideWrap(policy), port, *a, **k)
    S.serve = serve
    runpy.run_path(target, run_name="__main__")


if __name__ == "__main__":
    main()
