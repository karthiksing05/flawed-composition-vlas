"""FLAWED_COMPOSITION N3 (section 4.12): the compound factor checked by rule against the 40 LIBERO training tasks of pi0.5_libero,
ar0s/groot_libero and lerobot/xvla-libero.  No outcome is read.  CPU (SmolVLA venv).

Each N3 BDDL's goal is parsed with lifelong_ood/policy/factors.py's own rules into a slot (pred, object type, target type, region), and
its prompt is split into the referent phrase ("the black bowl <referent>") and the target phrase ("on the plate" / "on the stove" /
"on top of the cabinet").
  KNOWN iff the prompt is a training prompt and the slot is that training task's slot.
  NEW iff (a) the referent phrase occurs in a training prompt, (b) the slot occurs in some training task and its target phrase in a
  training prompt about the bowl (the atoms are trained), (c) no training prompt has both the referent and the target phrase, and (d)
  no training task of this scene (libero_spatial) has the slot (the joints are new).
Asserted to match the design in n3_tasks.json.
-> results/flawed/n3_novelty.json

    python experiments/fc_n3_novelty.py
"""
from __future__ import annotations

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments")]
OUT = os.path.join(ROOT, "results", "flawed")
PHRASE = {"plate": "on the plate", "stove": "on the stove", "cabinet": "on top of the cabinet"}


def slot_of(text):
    from lifelong_ood.policy import factors as F
    inst = dict(F._typed(F._section(text, "objects")), **F._typed(F._section(text, "fixtures")))
    goal = [tuple(a.split()) for a in re.findall(r"\(([A-Z][A-Za-z]* [^()]*)\)", F._section(text, "goal"))]
    assert len(goal) == 1 and goal[0][0] == "On", goal
    _, a, b = goal[0]
    tgt, region = F._target(b, inst)
    return ("On", inst.get(a, a), tgt, region)


def main():
    from lifelong_ood.policy import factors as F
    n3 = json.load(open(os.path.join(OUT, "n3_tasks.json"), encoding="utf-8"))
    train = F.load(F.TRAIN_SUITES)
    assert len(train) == 40
    prompts = {t.language: t for t in train}
    tr_slots = [(t.tid, t.scene, s) for t in train for s in t.slots if s[0] in ("In", "On")]
    out = {}
    for c in sorted({t["compound"] for t in n3["tasks"]}, key=lambda x: (x[0] != "K", int(x[1:]))):
        t = next(x for x in n3["tasks"] if x["compound"] == c)
        text = open(os.path.join(ROOT, t["bddl"]), encoding="utf-8").read()
        s = slot_of(text)
        lang, ref, tph = t["language"], t["referent"], PHRASE[t["target"]]
        assert lang == f"pick up the black bowl {ref} and place it {tph}", lang
        known = lang in prompts and s in prompts[lang].slots
        ref_seen = sorted(k.tid for k in train if f"the black bowl {ref} " in k.language)
        slot_seen = sorted(k for k, _, x in tr_slots if x == s)
        tph_seen = sorted(k.tid for k in train if "bowl" in k.language and k.language.endswith(tph))
        joint_prompt = sorted(k.tid for k in train if f"the black bowl {ref} " in k.language and k.language.endswith(tph))
        slot_in_scene = sorted(k for k, sc, x in tr_slots if x == s and sc == "libero_spatial")
        new = (not known) and bool(ref_seen) and bool(slot_seen) and bool(tph_seen) and not joint_prompt and not slot_in_scene
        status = "known" if known else ("new" if new else "unclassified")
        assert status == t["status"], (c, status, t["status"], s, ref_seen, slot_seen, tph_seen, joint_prompt, slot_in_scene)
        out[c] = dict(status=status, language=lang, slot=list(s), referent=ref, target_phrase=tph, training_with_referent=ref_seen,
                      training_with_slot=slot_seen, training_with_target_phrase=tph_seen, training_with_both=joint_prompt,
                      training_in_scene_with_slot=slot_in_scene)
        print(c, status, s, "| referent in", ref_seen, "| slot in", slot_seen, "| in-scene", slot_in_scene)
    with open(os.path.join(OUT, "n3_novelty.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(compounds=out, parser_hash=F.parser_hash(), training_set="the 40 LIBERO tasks"), fh, indent=1)


if __name__ == "__main__":
    main()
