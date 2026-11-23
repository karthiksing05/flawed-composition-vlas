"""PRE-REGISTRATION ROW tag `nph` (experiments/prereg/nph_pred.txt; docs/notes/plan_necessity_power.md 3): a learned, goal-directed separate where-head on pi0 (pi0-H*) vs pi0-A0, pi0-sL, 6th fresh state set.
Written BEFORE any pi0-H* training and BEFORE the 6th set is built (results/flawed/n2_states_w6, n2_branches_w6.json, results/flawed/nph6, jax/pi0_g8v_raah_l_ae/tl must not exist).  Copy of rvw5.py / pzdev.py.  sbatch (CPU) only."""
import hashlib
import json
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

rows = ledger.load("m3_prereg")
assert not any(r.get("tag") == "nph" for r in rows), "already registered"
for p in ("results/flawed/n2_states_w6", "results/flawed/n2_branches_w6.json", "results/flawed/nph6"):
    assert not os.path.exists(os.path.join(ROOT, p)), "exists: " + p
CK = "/coc/flash10/ksingara3/robot/checkpoints_l90c"
assert not os.path.exists(CK + "/jax/pi0_g8v_raah_l_ae/tl"), "main training chain exists"
assert not os.path.exists(CK + "/pytorch/pi0_g8v_raah_l_ae_tl6000"), "export exists"


def fsha(p):
    with open(p if os.path.isabs(p) else os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


PP = "/coc/flash10/ksingara3/.claude_tmp/pp/"
NEC = PP + "necpow/necpow_4847479.json"
nec = json.load(open(NEC, encoding="utf-8"))["seed"]
sig = dict(sigma_s_N=nec["N"]["sig_seed"], sigma_i_N=nec["N"]["sig_int_per_arm"], sigma_s_K=nec["K"]["sig_seed"])
assert round(sig["sigma_s_N"], 3) == 0.033 and round(sig["sigma_i_N"], 3) == 0.070 and round(sig["sigma_s_K"], 3) == 0.049, sig
fit = json.load(open(os.path.join(ROOT, "results", "flawed", "raah0", "fit.json"), encoding="utf-8"))
gs, gh = fit["gates_Hstar"], fit["gates_H_registered"]
assert fit["all_pass"] and fit["npz_sha256"] == fsha("lifelong_ood/policy/raah0_head.npz")
CODE = ["experiments/prereg/nph_pred.txt", "docs/notes/plan_necessity_power.md", "experiments/raah0_fit.py", "lifelong_ood/policy/raah0_head.npz", "results/flawed/raah0/fit.json",
        "lifelong_ood/policy/raah0_model.py", "lifelong_ood/policy/backbone_h0.py", "lifelong_ood/policy/l90c_config.py", "experiments/fc_n2_w6.py", "experiments/nph_read.py",
        "experiments/nph_disjoint.py", "experiments/nph_geom.py", "experiments/nph_e1.py", "experiments/raah0_check.py", "experiments/raah0_cfgcheck.py", "experiments/c_train_nph.py",
        "experiments/raa_wh0.py", NEC, PP + "nph_train.sh", PP + "nph_chain.sh", PP + "nph_export.sh", PP + "nph_check.sh", PP + "nph_roll.sh", PP + "nph_submit.sh", PP + "nph_states.sh"]
PRED = open(os.path.join(ROOT, "experiments", "prereg", "nph_pred.txt"), encoding="utf-8").read()
SEEN = open(os.path.join(ROOT, "experiments", "prereg", "nph_seen.txt"), encoding="utf-8").read().strip()
Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="nph", seed=0,
                    command="pre-registration: NPH: pi0-H* (learned goal-directed where-head, pi0) vs pi0-A0 and pi0-sL on the 6th fresh state set; n1 (primary), n1r, n2, d1'', nK",
                    metrics={"status": "PRE-REGISTERED NPH", "arm": "pi0_g8v_raah_l_ae (new, tl6000), pi0_g8v_raac_l_ae tl6000, pi0_g8v_raas_l_ae tl6000; Phase A N1-N6 / K1-K6 (60 states, k 820-879)",
                             "mapping": "docs/notes/plan_necessity_power.md section 3", "prediction": PRED, "seen": SEEN,
                             "head_gates": dict(Hstar=gs, H_registered=gh, thresholds=fit["thresholds"], npz_sha256=fit["npz_sha256"]), "sigma_constants": sig,
                             "deviation_notes": "scope fixed by the orchestrator: core only (no 5th-set option); pi0-H* K jobs first as the gate; hard cap 23 GPU-h",
                             "files_sha256": {p: fsha(p) for p in CODE}, "written_before": "pi0-H* training and building the 6th set / any rollout on it"},
                    bytes=Z, config={}, script="experiments/prereg/nph.py"))
