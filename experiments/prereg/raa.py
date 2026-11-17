"""PRE-REGISTRATION ROW of RAA-min (SCAFFOLD_RESULTS 73; docs/PLAN_PAPER.md 2b): R0's and R2's inputs plus a role channel (the active role
entity's position minus the end effector's, 1 cm, plus a role id, as a prompt suffix), arms Rrel and RAA, against R0, R2, R2l and
ReGuide-perc (rgp), N1-N6 and K1-K6, 20 dev states + 40 new states k = 540-579.  BEFORE any main rollout.

    python experiments/prereg/raa.py
"""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

FL = os.path.join(ROOT, "results", "flawed")
TIDS = (0, 2, 4, 6, 8, 10, 16, 18, 20, 22, 24, 26)
rows = ledger.load("m3_prereg")
assert not any(r.get("tag") == "raa" for r in rows), "already registered"
assert not os.path.exists(os.path.join(FL, "raa")), "raa rollouts exist"
assert not os.path.exists(os.path.join(FL, "raa_read.json")), "raa read exists"


def fsha(p):
    with open(p if os.path.isabs(p) else os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


REG = [r for r in rows if r.get("tag") == "reguide"]
assert len(REG) == 1, len(REG)
H = REG[0]["metrics"]["files_sha256"]
for p in ("experiments/rg.py", "experiments/rg_stats.py", "results/flawed/rg_stats.json", "results/flawed/sc_pu_calib.json"):
    assert H[p] == fsha(p), ("ReGuide file differs from its reguide-row hash", p)
print("[raa] ReGuide files match the reguide row", flush=True)

PP = "/coc/flash10/ksingara3/.claude_tmp/pp/"
STATE_FILES = (["results/flawed/n2_states_rx/states.npy", "results/flawed/n2_states_rx/meta.json"]
               + ["results/flawed/n2_states_rx/n2_goal_%d.npy" % t for t in TIDS]
               + ["results/flawed/n2_branches_rx.json", "results/flawed/n2_branches_main.json", "results/flawed/n2_tasks.json",
                  "results/flawed/rg_stats.json", "results/flawed/sc_pu_calib.json", "results/flawed/sc_rv_lex.json",
                  "results/flawed/sc_lex.json", "results/flawed/sc_rv_g8v_rrel_manifest.json", "results/flawed/sc_rv_g8v_raa_manifest.json"])
CODE = ["experiments/raa.py", "experiments/raa_read.py", "experiments/fc_n2_rx.py", "experiments/sc_rv.py", "experiments/sc_pu.py",
        "experiments/sc_seg.py", "experiments/sc_t16.py", "experiments/fa1_common.py", "experiments/fc_n2_client.py",
        "experiments/libero_client.py", "experiments/rg.py", "experiments/rg_stats.py", "experiments/rg_roll.sh",
        "experiments/c_train_pi05.py", "lifelong_ood/policy/l90c_config.py", "lifelong_ood/policy/parts_flow.py",
        "lifelong_ood/policy/backbone.py", "lifelong_ood/policy/serve.py", "experiments/prereg/raa_pred.txt",
        "experiments/prereg/raa_seen.txt", PP + "raa_job.sh", PP + "raa_train.sh", PP + "raa_conv.sh", PP + "raa_roll.sh"]
PRED = open(os.path.join(ROOT, "experiments", "prereg", "raa_pred.txt"), encoding="utf-8").read()
SEEN = open(os.path.join(ROOT, "experiments", "prereg", "raa_seen.txt"), encoding="utf-8").read()
Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="raa", seed=0,
                    command="pre-registration: RAA-min, R0 / R2 inputs plus a role channel (Rrel, RAA) against R0, R2, R2l and "
                            "ReGuide-perc on N1-N6 and K1-K6 (docs/SCAFFOLD_RESULTS.md 73)",
                    metrics={"status": "PRE-REGISTERED RAA-MIN",
                             "arms": "Rrel = pi05_g8v_rrel_ae (R0's frames and whole instructions + channel); RAA = pi05_g8v_raa_ae "
                                     "(R2's frames, role phrases and painted views + channel).  Frozen VLM, action expert, batch 32, "
                                     "6,000 steps, lr 5e-5, 1,000 warmup, no EMA, one seed, one A40, submitted on overcap (exp main) "
                                     "and tail-lab (exp tl), the first to reach 6,000 exported.  Test: raa.py serve / client; channel "
                                     "from learned masks + depth + calibration (sc_pu.Estimator), pointer from the contact-induced "
                                     "lexicon; R2's views painted from learned masks.  References R0, R2, R2l on dev states from "
                                     "SR 37 / 43 unchanged, rolled anew on the new states; rgp = rg.py client --arm rgp.",
                             "channel": "pi05 here has discrete_state_input=False: the state vector is never read, so the channel is a "
                                        "text suffix '<prompt>. rel: +12 -3 +5 cm, role target' (whole cm, base-frame axes, role "
                                        "object until ProprioHeld, target while held), on routed frames only (K1-K6, N1-N6).",
                             "states": "dev states = harness trials 4-23 of n2_states (n2_branches_main.json); new = k 540-579 "
                                       "(fc_n2_rx.py; harness trials 24-63, k = trial + 516, seed_base 13000000 + 1000 k).",
                             "tasks": "familiar n2 tids %s (K1-K6, N1-N6)" % (TIDS,),
                             "mapping": "experiments/prereg/raa_pred.txt, 'Labels and margins' (read: experiments/raa_read.py)",
                             "prediction": PRED,
                             "seen": SEEN,
                             "files_sha256": {p: fsha(p) for p in CODE + STATE_FILES},
                             "written_before": "any main rollout of Rrel or RAA, and of R0 / R2 / R2l / rgp on the new states"},
                    bytes=Z, config={}, script="experiments/prereg/raa.py"))
