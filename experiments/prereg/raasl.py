"""PRE-REGISTRATION ROW of RAA-sL (SR 82; docs/notes/raasl_plan.md): the RAA-s recipe (servo residual, RAA-c state token) initialised from the LIBERO-fine-tuned pi0.5 (openpi
pi05_libero params) instead of pi05_base.  Phase A: N1-N6 / K1-K6 on SR 73's 60 states; phase B: wa_s.py arms full / noR / noM on rg_comp's rgc main states.  BEFORE any main
RAA-sL training job (exp main / tl) and any RAA-sL rollout.  Predictions: experiments/prereg/raasl_pred.txt (committed before this row).

    python experiments/prereg/raasl.py        (login-free: run through sbatch)
"""
import hashlib
import os
import sys

ROOT = "/coc/flash10/ksingara3/flawed-composition-vlas"
sys.path.insert(0, ROOT)
from lifelong_ood import ledger                                              # noqa: E402

FL = os.path.join(ROOT, "results", "flawed")
JAX = "/coc/flash10/ksingara3/robot/checkpoints_l90c/jax/pi05_g8v_raas_l_ae"
rows = ledger.load("m3_prereg")
assert not any(r.get("tag") == "raasl" for r in rows), "already registered"
assert not any(f.startswith("raasl_") for f in os.listdir(os.path.join(FL, "raa"))), "raasl rollouts exist"
assert not os.path.exists(os.path.join(FL, "wasl")), "wasl rollouts exist"
assert not any(os.path.exists(os.path.join(FL, f)) for f in ("raasl_read.json", "raasl_wa_read.json", "raasl_wa_gated.json")), "raasl read exists"
assert not any(os.path.isdir(os.path.join(JAX, e)) for e in ("main", "tl")), "main RAA-sL training has started"
smoke_dirs = sorted(os.listdir(JAX)) if os.path.isdir(JAX) else []
assert set(smoke_dirs) <= {"smoke"}, smoke_dirs                              # the 20-step plumbing checkpoint (exp smoke) is the only allowed training so far


def fsha(p):
    with open(p if os.path.isabs(p) else os.path.join(ROOT, p), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


RAA = [r for r in rows if r.get("tag") == "raa"]
RGC = [r for r in rows if r.get("tag") == "rg_comp"]
WA = [r for r in rows if r.get("tag") == "wa"]
assert len(RAA) == 1 and len(RGC) == 1 and len(WA) == 1, (len(RAA), len(RGC), len(WA))
for t in ("raal", "raas", "raas_b"):
    assert sum(1 for r in rows if r.get("tag") == t) == 1, t
H = RAA[0]["metrics"]["files_sha256"]
for p in ("results/flawed/n2_branches_rx.json", "results/flawed/n2_branches_main.json", "results/flawed/n2_tasks.json", "results/flawed/rg_stats.json",
          "results/flawed/sc_pu_calib.json", "results/flawed/sc_rv_lex.json", "results/flawed/sc_lex.json", "results/flawed/n2_states_rx/states.npy",
          "results/flawed/n2_states_rx/meta.json"):
    assert H[p] == fsha(p), ("phase A state / calibration file differs from the raa row's hash", p)
H0 = RGC[0]["metrics"]["files_sha256"]
STATE = [p for p in H0 if p.startswith("results/flawed/rg_comp/") and (p.endswith(".npy") or "br" in os.path.basename(p) or p.endswith("tasks.json"))]
for p in STATE + ["results/flawed/sc_pu_calib.json", "results/flawed/sc_rv_lex.json", "experiments/fx_r.py", "experiments/rg_comp.py", "experiments/f6o2_tasks.py"]:
    assert H0[p] == fsha(p), ("phase B state / code differs from the rg_comp row", p)
print("[raasl] phase A states / branches / calibration / lexicon match the raa row; phase B rgc states, branches, tasks match the rg_comp row", flush=True)

PP = "/coc/flash10/ksingara3/.claude_tmp/pp/"
CODE = ["experiments/raasl_read.py", "experiments/raal_read.py", "experiments/raas_b_read.py", "experiments/raa_read.py", "experiments/raao_read.py", "experiments/raac_read.py",
        "experiments/raas_read.py", "experiments/wa_read.py", "experiments/wa.py", "experiments/wa_s.py", "experiments/raa.py", "experiments/sc_rv.py", "experiments/c_train_pi05.py",
        "experiments/raac_export.py", "experiments/pp_p_train.py", "experiments/fa1_common.py", "experiments/fc_n2_client.py", "experiments/libero_client.py",
        "experiments/rg_comp.py", "experiments/fx_r.py", "lifelong_ood/policy/l90c_config.py", "lifelong_ood/policy/raas_model.py", "lifelong_ood/policy/raac_model.py",
        "lifelong_ood/policy/parts_flow.py", "lifelong_ood/policy/backbone.py", "lifelong_ood/policy/serve.py", "experiments/prereg/raasl_pred.txt",
        "docs/notes/raasl_plan.md", "results/flawed/raas_gate.json", "results/flawed/sc_lex.json", "results/flawed/n2_tasks.json",
        "/coc/flash10/ksingara3/robot/checkpoints_l90c/assets/pi05_g8v_raac/lol/libero_g8v_raac/norm_stats.json",
        PP + "raasl_train.sh", PP + "raasl_chain.sh", PP + "wasl_roll.sh", PP + "raa_roll.sh", PP + "t16_export.sh"] + STATE
PRED = open(os.path.join(ROOT, "experiments", "prereg", "raasl_pred.txt"), encoding="utf-8").read()
SEEN = ("Written 2026-11-19.  Known when written: SR 73, 76, 77, 78b, 79, 80 (RAA, RAA-o, RAA-c, RAA-s with servo, WA on RAA, RAA-L).  Checks before the row, plumbing and not "
        "results: 20-step training smoke (exp smoke, tail-lab job 4801501: pi05_libero params load, state_proj random-initialised, step-0 loss .0556); wa_s.py (the state-token copy of "
        "wa.py: wa.py carries the channel as text and cannot serve the RAA-s family) smoke on the SR 78 RAA-s checkpoint, outside results/ (job 4801514).  No code, arm, mapping or "
        "prediction changed after these checks.")
Z = dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0)
print(ledger.record("m3_prereg", bench="libero_goal", tag="raasl", seed=0,
                    command="pre-registration: RAA-sL (SR 82), the RAA-s recipe (servo residual + RAA-c state token) initialised from the LIBERO-fine-tuned pi0.5 (pi05_libero params), "
                            "N1-N6 / K1-K6 on SR 73's states and the WA module-removal arms (wa_s.py) on rg_comp's rgc main states (docs/notes/raasl_plan.md)",
                    metrics={"status": "PRE-REGISTERED RAA-sL",
                             "arm": "RAA-sL = pi05_g8v_raas_l_ae: pi05_g8v_raas_ae's dataset (lol/libero_g8v_raac), norm stats, frames, role views, state-token channel, servo residual "
                                    "(K, M from results/flawed/raas_gate.json) and recipe (frozen VLM, action expert + state_proj, batch 32, 6,001 steps, lr 5e-5, 1,000 warmup, no EMA, "
                                    "one seed, one A40, exp main on overcap and exp tl on tail-lab, the first to reach 6,000 exported); ONE change: the initial weights are openpi "
                                    "pi05_libero's params (the frozen I of SR 45 / 52 / 70) instead of pi05_base's (frozen VLM included); state_proj, absent from pi05_libero, "
                                    "stays at the model's random init as in RAA-s",
                             "leakage": "as SR 80: n2 tasks N1-N8 match no task of libero_spatial / object / goal / 10 / 90, K1-K8 are LIBERO-Goal tasks",
                             "gate": "none beyond plumbing: smoke 20 steps ok",
                             "states": "phase A as SR 73 (dev trials 4-23 + new k 540-579); phase B rg_comp rgc main states",
                             "tasks": "phase A familiar n2 tids (0, 2, 4, 6, 8, 10, 16, 18, 20, 22, 24, 26); phase B rgc tids 0-10",
                             "mapping": "docs/notes/raasl_plan.md 'Registered contrasts' (read: experiments/raasl_read.py)",
                             "prediction": PRED, "seen": SEEN,
                             "files_sha256": {p: fsha(p) for p in CODE},
                             "written_before": "any main RAA-sL training job (exp main / tl) and any RAA-sL rollout"},
                    bytes=Z, config={}, script="experiments/prereg/raasl.py"))
