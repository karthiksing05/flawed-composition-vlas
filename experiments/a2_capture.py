"""SR 83 (A2 distributedness probe), server side: the capture wrapper.  Runs any serve script of the repo through steer_conf_serve.py's
patches (steer_serve.SideWrap: a side query never changes what a later real query sees), and records, at the GuidedPolicy (the last
stage before the model, i.e. AFTER the role views, the paint and the channel of the RAA family), the dict the model is given:
    observation/image, observation/wrist_image (uint8), observation/state, prompt, the noise seed GuidedPolicy drew, the returned chunk.
Records are written for
    * every query whose client sent meta/a2_id (the N / K hold-phase queries of a2_cap_client.py: the real query and one side query per candidate);
    * with A2_ROW0=1, the first query of every episode (meta/episode_start), id row0_<tag>_t<meta/episode>, tag = the content of the file A2_TAG_FILE
      (the F7 rows: native rg_comp / wa_s clients; the job script writes tid<k> there before each client).
Files: $A2_OUT/<id>.npz.  Nothing the policy sees or the episode executes changes.

    A2_OUT=DIR [A2_ROW0=1] python experiments/a2_capture.py <serve script> <its arguments>
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
KEYS = ("observation/image", "observation/wrist_image", "observation/state", "prompt")


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
                np.savez(path, image=np.asarray(ob["observation/image"]), wrist=np.asarray(ob["observation/wrist_image"]),
                         state=np.asarray(ob["observation/state"], np.float64), prompt=np.array(str(ob["prompt"])), seed=np.array(sd),
                         actions=np.asarray(out["actions"], np.float64))
                n[0] += 1
        return out
    GP.infer = infer


if __name__ == "__main__":
    install()
    SS.main()
