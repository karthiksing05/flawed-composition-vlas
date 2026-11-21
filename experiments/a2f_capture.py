"""RAA-F probe (docs/notes/raaf_plan.md 6.1, docs/notes/raaf_probe_runbook.md): the capture wrapper for the servo-free state-token models (Route A / A0 / A+ / B).

A NEW copy of a2_capture.py (unchanged).  The only change: the record also stores observation/crop_image (Route B's third image, 224 x 224 x 3 uint8) as the npz key
`crop` when the model input has it, so that a2_probe_f.py can patch the crop image (patch group in.crop).  Everything else, including the seed bookkeeping and the
file names, is a2_capture.py's.  Serve scripts: raa.py (A, A0, A+), raa_b.py (B), as in a2f_capture.sh.

    A2_OUT=DIR [A2_ROW0=1] python experiments/a2f_capture.py <serve script> <its arguments>
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
import steer_conf_serve as SC                                              # noqa: E402,F401  (patches steer_serve; main below)
import steer_serve as SS                                                   # noqa: E402

OUT = os.environ["A2_OUT"]
ROW0 = os.environ.get("A2_ROW0") == "1"
KEYS = ("observation/image", "observation/wrist_image", "observation/state", "prompt", "observation/crop_image")


def install():
    import numpy as np
    import lifelong_ood.policy.serve as S
    GP = S.GuidedPolicy
    orig = GP.infer
    os.makedirs(OUT, exist_ok=True)
    n = [0]

    def infer(self, obs, *, noise=None):
        aid = obs.get("meta/a2_id")
        row0 = ROW0 and bool(obs.get("meta/episode_start"))
        tr = obs.get("meta/episode")
        ob = {k: obs[k] for k in KEYS if k in obs} if (aid is not None or row0) else None
        out = orig(self, obs, noise=noise)
        if ob is not None:
            self.calls -= 1
            self.ep_q -= 1                                                 # the counters orig advanced: the seed this query drew
            sd = int(self._seed())
            self.calls += 1
            self.ep_q += 1
            tag = open(os.environ["A2_TAG_FILE"]).read().strip() if os.environ.get("A2_TAG_FILE") else "x"
            name = str(aid) if aid is not None else "row0_%s_t%d" % (tag, int(tr))
            name = re.sub(r"[^A-Za-z0-9_.+-]", "_", name)
            path = os.path.join(OUT, name + ".npz")
            if not os.path.exists(path):
                extra = {"crop": np.asarray(ob["observation/crop_image"], np.uint8)} if "observation/crop_image" in ob else {}
                np.savez(path, image=np.asarray(ob["observation/image"]), wrist=np.asarray(ob["observation/wrist_image"]),
                         state=np.asarray(ob["observation/state"], np.float64), prompt=np.array(str(ob["prompt"])), seed=np.array(sd),
                         actions=np.asarray(out["actions"], np.float64), **extra)
                n[0] += 1
        return out
    GP.infer = infer


if __name__ == "__main__":
    install()
    SS.main()
