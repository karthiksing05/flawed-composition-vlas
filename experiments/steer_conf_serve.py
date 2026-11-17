"""STEER-CONF server wrapper for the raa.py servers (Rrel, RAA).  steer_serve.py (frozen by SR 74) runs unchanged; two plumbing patches
are applied around it, both needed because raa.RAAServer holds its role sequencer one level down:

  1. steer_serve.CHILD gets "rv": the chain of wrapped policies (whose det / prev / calls / ep_q are restored after every side query)
     then runs RAAServer -> sc_rv.RoleView -> raa._Hook -> GuidedPolicy.  Without it RoleView's ProprioHeld and the GuidedPolicy counters
     would advance on side queries and change the executed episode.  RAAServer.cur is overwritten by every query before it is read, and
     RAAServer.n / .noch are counters that do not affect outputs, so they need no restore; sc_seg.LearnedMasks (RAA) keeps only
     diagnostics (acc, n, secs; secs is in STATE).
  2. steer_serve.Override replaces the whole served prompt with "object: <word>" / "target: <word>".  RAA's prompt already carries the
     channel suffix when the override sees it, so the registered rule (absent / nonsense words keep the true instruction's channel)
     needs the suffix kept: the patched Override keeps everything from ". rel: " on.

    python experiments/steer_conf_serve.py experiments/raa.py serve --arm rrel|raa --ckpt DIR --port P ...
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE]
import steer_serve as SS                                                   # noqa: E402

SS.CHILD = ("p", "inner", "rv")
SUF = ". rel: "


_Orig = SS.Override
_OrigSW = SS.SideWrap


class KeepChannelOverride(_Orig):
    text = None                                                            # Rrel: the side query's whole instruction (meta/side_text)

    def infer(self, o):
        p = str(o["prompt"])
        if SUF in p and self.text is not None:
            o = dict(o)
            o["prompt"] = self.text + p[p.index(SUF):]
            self.hit += 1
            import logging
            logging.info(f"[steer] override text {self.text!r}: served prompt {p!r} -> {o['prompt']!r}")
            return self.inner.infer(o)
        if self.slot is not None and SUF in p and (p.startswith("object: ") or p.startswith("target: ")):
            pre = "object: " if self.slot == "object" else "target: "
            if p.startswith(pre):
                o = dict(o)
                o["prompt"] = pre + self.phrase + p[p.index(SUF):]
                self.hit += 1
            else:
                self.miss += 1
            import logging
            logging.info(f"[steer] override slot={self.slot} phrase={self.phrase!r} served prompt {p!r} -> {o['prompt']!r} "
                         f"(hit {self.hit} miss {self.miss})")
            return self.inner.infer(o)
        return _Orig.infer(self, o)


class SideWrapText(_OrigSW):
    def infer(self, obs):
        if obs.get("meta/side") and "meta/side_text" in obs:
            o = dict(obs)
            txt = str(o.pop("meta/side_text"))
            self.override.text = txt
            try:
                return _OrigSW.infer(self, o)
            finally:
                self.override.text = None
        return _OrigSW.infer(self, obs)


SS.Override = KeepChannelOverride
SS.SideWrap = SideWrapText

if __name__ == "__main__":
    SS.main()
