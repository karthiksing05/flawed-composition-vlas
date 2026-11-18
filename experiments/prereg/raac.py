"""PRE-REGISTRATION ROW of RAA-c (SCAFFOLD_RESULTS 77): RAA-o with the role channel as a CONTINUOUS state token (one new linear projection, no text suffix),
against RAA-o, RAA and ReGuide-perc (rgp) on N1-N6 and K1-K6, the SAME 60 states as SR 73 / SR 76.  BEFORE any main RAA-c training job and any RAA-c
rollout.  Predictions: experiments/prereg/raac_pred.txt (committed c53e4a70 before this row).

    python experiments/prereg/raac.py
"""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

FL = os.path.join(ROOT, "results", "flawed")
JAX = "/coc/flash10/ksingara3/robot/checkpoints_l90c/jax/pi05_g8v_raac_ae"
TIDS = (0, 2, 4, 6, 8, 10, 16, 18, 20, 22, 24, 26)
rows = ledger.load("m3_prereg")
assert not any(r.get("tag") == "raac" for r in rows), "already registered"
assert not any(f.startswith("raac_") for f in os.listdir(os.path.join(FL, "raa"))), "raac rollouts exist"
assert not os.path.exists(os.path.join(FL, "raac_read.json")), "raac read exists"
assert not any(os.path.isdir(os.path.join(JAX, e)) for e in ("main", "tl")), "main RAA-c training has started"
smoke_dirs = sorted(os.listdir(JAX)) if os.path.isdir(JAX) else []
assert smoke_dirs in ([], ["smoke"]), smoke_dirs                             # the quick plumbing checkpoint (exp smoke) is the only allowed training so far


def fsha(p):
    with open(p if os.path.isabs(p) else os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


RAA = [r for r in rows if r.get("tag") == "raa"]
RAAO = [r for r in rows if r.get("tag") == "raao"]
assert len(RAA) == 1 and len(RAAO) == 1, (len(RAA), len(RAAO))
H = RAA[0]["metrics"]["files_sha256"]
for p in ("results/flawed/n2_branches_rx.json", "results/flawed/n2_branches_main.json", "results/flawed/n2_tasks.json", "results/flawed/rg_stats.json",
          "results/flawed/sc_pu_calib.json", "results/flawed/sc_rv_lex.json", "results/flawed/sc_lex.json", "results/flawed/n2_states_rx/states.npy",
          "results/flawed/n2_states_rx/meta.json"):
    assert H[p] == fsha(p), ("state / calibration file differs from the raa row's hash", p)
print("[raac] states, branches, calibration and lexicon match the raa row", flush=True)

PP = "/coc/flash10/ksingara3/.claude_tmp/pp/"
CODE = ["experiments/raa.py", "experiments/raac_read.py", "experiments/raac_check.py", "experiments/raac_model_check.py", "experiments/raac_sens_check.py",
        "experiments/raac_export.py", "experiments/raac_final_check.py", "experiments/raao_read.py", "experiments/raa_read.py", "experiments/sc_rv.py", "experiments/sc_pu.py",
        "experiments/sc_seg.py", "experiments/sc_t16.py", "experiments/fa1_common.py", "experiments/fc_n2_client.py", "experiments/libero_client.py",
        "experiments/c_train_pi05.py", "experiments/pp_p_train.py", "lifelong_ood/policy/l90c_config.py", "lifelong_ood/policy/raac_model.py",
        "lifelong_ood/policy/parts_flow.py", "lifelong_ood/policy/backbone.py", "lifelong_ood/policy/serve.py", "experiments/prereg/raac_pred.txt",
        "docs/notes/raac_plan.md", "results/flawed/sc_rv_raac_manifest.json", "results/flawed/raao_read.json", PP + "raa_job.sh", PP + "raac_train.sh",
        PP + "raa_conv.sh", PP + "raa_roll.sh", PP + "t16_export.sh"]
PRED = open(os.path.join(ROOT, "experiments", "prereg", "raac_pred.txt"), encoding="utf-8").read()
SEEN = ("Written 2026-11-18.  Known when written: SR 76's read (RAA-o N1-N6 .594, RAA .656, rgp .897, K1-K6 .881; W1 NOT SUPPORTED) and its exploratory release-aim "
        "checks.  Predictions committed (c53e4a70) before this row, any main training job and any RAA-c rollout.  Checks before the row, all plumbing and not results: "
        "channel-tensor unit check (experiments/raac_check.py, K5 and K1 demo 0: the state vector = RAA-o's channel values, whole-cm rounding equals RAA-o's text "
        "numbers on every frame, prompts = role phrase without suffix, images identical); model-shape check (experiments/raac_model_check.py: only state_proj "
        "[32, 1024] + [1024] added, suffix = H + 1 tokens, state_proj trainable and the VLM frozen, loader leaves state_proj at its random init); raac_read selftest "
        "reproduces SR 73's V1a / V1b and SR 76's W1a / W1b / WK exactly (job 4780945); server-side unit check of the held-target / object / refused / unrouted state (raac_final_check.py, job 4781846); conversion job 4780894, model check 4780904, channel check 4780886.  A 20-step quick checkpoint (exp smoke, not exp main / tl; jobs 4780946 train, 4780947 export) was trained, exported and served for "
        "the forward-pass sensitivity check (job 4780948: the chunk changes with the channel, obs.state = normalised channel then zeros) and a 4-episode rollout smoke (job 4780949; the untrained policy never held an object, so the held branch is covered by the unit check) on smoke states 0-1 (not in the 60 read states, output outside the read directory).  "
        "No code, arm, mapping or prediction changed after the smoke.")
Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="raac", seed=0,
                    command="pre-registration: RAA-c, RAA-o with the role channel as a continuous state token (one new linear projection, no text suffix), against RAA-o, "
                            "RAA and ReGuide-perc on N1-N6 and K1-K6, same states as SR 73 / 76 (docs/SCAFFOLD_RESULTS.md 77)",
                    metrics={"status": "PRE-REGISTERED RAA-C",
                             "arm": "RAA-c = pi05_g8v_raac_ae: RAA-o's frames, role phrases, painted views, channel values (target - held object while held), recipe (frozen "
                                    "VLM, action expert, batch 32, 6,001 steps, lr 5e-5, 1,000 warmup, no EMA, one seed, one A40, exp main on overcap and exp tl on tail-lab, "
                                    "the first to reach 6,000 exported); ONE change: the channel [dx, dy, dz in metres, unrounded; role id 1 object / 2 target; zeros when no "
                                    "channel] enters as the model's continuous state vector through ONE new linear layer state_proj (32 -> 1024, random init, trained) into "
                                    "ONE new causal suffix token before the action tokens (pi0's mechanism), and the text suffix is removed; the robot's own state is not given.",
                             "states": "as SR 73 / 76: dev trials 4-23 of n2_states + new k 540-579 (harness trials 24-63)",
                             "tasks": "familiar n2 tids %s (K1-K6, N1-N6)" % (TIDS,),
                             "mapping": "docs/notes/raac_plan.md 'Registered contrasts' (read: experiments/raac_read.py, extends raa_read.py / raao_read.py)",
                             "prediction": PRED, "seen": SEEN,
                             "files_sha256": {p: fsha(p) for p in CODE},
                             "written_before": "any main RAA-c training job (exp main / tl) and any RAA-c rollout in the read directory"},
                    bytes=Z, config={}, script="experiments/prereg/raac.py"))
