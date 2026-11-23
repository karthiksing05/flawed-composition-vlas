# Plan: one decisive test of "a separate owner of the where beats letting the network learn it" (power analysis + pre-registration)

Written 2026-11-23.  Design only.  Nothing was trained or rolled, and no file in the repo other than this one was written.
The CPU power job (overcap, 2 CPUs) is 4847479.  Its script, output and JSON live outside git:
/coc/flash10/ksingara3/.claude_tmp/pp/necpow/{necpow.py, necpow.sh, necpow_4847479.out, necpow_4847479.json}.
Inputs: docs/notes/review_atlas_results.md (all, incl. section 7 E4), plan_review_atlas.md sections 1, 3, 6, 8, 9, 10, review_atlas_adv.md,
paper/icml2027/main.tex (abstract, sections 6-8, parts/matrix.tex), results/flawed/rvw5/rvw5_read_{d,h}.json, results/flawed/raah/fit.json,
results/flawed/rva1/rva1_w5_H.json (header), the Phase A grids of 9 arms on the 5th set and 7 arms on the dev set (read through
rvw5_read.grid5 / raaf_dev_read.grid, unchanged), experiments/raah_fit.py, lifelong_ood/policy/raah_model.py, l90c_config.py (pi0 and H
entries), experiments/fc_n2_w5.py, results/flawed/n2_bddl/{N7,N8,K7,K8}.bddl (goal lines).

---

## 0. Answer in brief

- **Task variance drives everything on pi0.5, and it comes from one pairing.**  The registered h1 interval (H - A0, about +-.3) is 98-99% between-task variance.
  Within a task, 60 paired states give a half-width of only .04-.05.  The spread comes from N2 (cream cheese -> stove).  There H scores
  .067 (5th set) and A0 scores 1.0, a task difference of -.93 (-.83 on dev).  On N1, N4 and N6 H beats A0 by +.32 to +.63.
- **On pi0.5, no learned-head design within 25 GPU-h can be decisive against A0 under the registered rule.**  Even an arm exactly as good as
  RAA-sL (the servo itself) would pass the registered task+state rule against A0 only 40% of the time, and against A only 20% (Monte Carlo
  below).  The per-task gains of the servo over A0 are themselves uneven (sd .23-.28 across tasks).  Shrinking the task term needs about 15
  pairings, and LIBERO-Goal has 8 at most.
- **On pi0 the same test is powered.**  The servo's gain over pi0-A0 is uniform across tasks: every task difference is positive on both sets,
  with sd .15-.20.  If a learned separate owner recovers the servo's gain, the registered rule detects it 91% of the time and the primary
  rule below 99%.  At 75% of the gain the figures are 70% and 83%; at 50% they are 51% and 59%.
- **Recommended single experiment (about 18.4 GPU-h core, 22-25 with contingency and the option): pi0-H\*.**  This is pi0 with a learned,
  goal-directed where-head in the servo's slot, the expert retrained on the residual.  It is tested against pi0-A0 (the network that reads
  the target, .542, and still fails) on a new sealed 6th state set, with pi0-sL as the reference.  H\* fixes the registered head's failure
  by principle: it is trained on the demo's own remaining displacement to its release point, not on the demo's velocity.  The fix is fixed
  and gated on training demos before any training or held-out read.
- **Primary rule n1.**  Tasks are fixed and states are bootstrapped.  The interval is widened by the measured seed variance.  There is also
  a no-catastrophic-task condition, so an average cannot hide a pairing where the head collapses (the E4 pattern).  The registered
  task+state rule is reported next to it as n1r.
- **Plainly:** within 25 GPU-h the pi0 design is decisive if the learned owner gets at least about 3/4 of the servo's gain (a true
  difference of +.18 or more).  It is not decisive for smaller true effects: at +.12 it has power .59 and at +.06 power .24.
  - A design decisive against the best pi0.5 in-network route (A, a gap of about .09) needs about 5 seeds per arm under fixed tasks, about
    120-140 GPU-h.
  - Under task-level inference it is impossible in this scene at any budget (section 2.4).

---

## 1. What the existing data say (power job 4847479)

### 1.1 Loader check
The job re-ran the registered routine (rvw5_read.boot_diff, B = 2000, rng 0) and reproduced the registered numbers exactly:
d1' -.222 [-.383, -.069]; h1 5th set +.106 [-.353, +.467]; h1 pooled +.089 [-.208, +.346].

### 1.2 Per-task differences on held-out pairings (N1 cheese->plate, N2 cheese->stove, N3 cheese->cabinet top, N4 bottle->plate, N5 bottle->stove, N6 bottle->bowl)
| contrast | set | N1 | N2 | N3 | N4 | N5 | N6 | mean | sd over tasks | half-width, registered (tasks+states) | half-width, tasks fixed (states only) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| H - A0 | dev | +.383 | **-.833** | -.117 | +.317 | +.050 | +.633 | +.072 | .515 | .382 | .051 |
| H - A0 | 5th | +.450 | **-.933** | +.017 | +.467 | +.033 | +.600 | +.106 | .563 | .410 | .040 |
| H - RAA-sL | 5th | +.067 | **-.917** | +.150 | +.133 | .000 | +.200 | -.061 | .425 | .300 | .043 |
| RAA-sL - A0 | dev | +.217 | .000 | -.167 | +.467 | +.067 | +.567 | +.192 | .282 | .217 | .053 |
| RAA-sL - A0 | 5th | +.383 | -.017 | -.133 | +.333 | +.033 | +.400 | +.167 | .233 | .181 | .050 |
| RAA-sL - A | 5th | +.267 | -.017 | -.150 | +.117 | .000 | +.300 | +.086 | .175 | .144 | .050 |
| pi0-sL - pi0-A0 | dev | +.117 | +.300 | +.050 | +.450 | +.333 | +.283 | +.256 | .147 | .124 | .050 |
| pi0-sL - pi0-A0 | 5th | +.017 | +.133 | +.033 | +.483 | +.417 | +.250 | +.222 | .196 | .154 | .049 |
| RAA-sL - RAA-sL s43 (seed) | dev / 5th | | | | | | | +.056 / +.044 | .129 / .097 | .114 / .093 | .049 / .051 |

Reading:
- H's whole deficit is N2 on both sets.  It shows in the E1 split (rva1_w5_H.json): 55 of H's 70 failures on the 5th set are class H
  (held at the target to the step cap), and 51 of its 57 cheese failures are class H.  On the other five pairings H matches or beats
  RAA-sL (+.07, +.15, +.13, .00, +.20).
- The between-task variance (sigma_tau .51-.56 for H - A0) is 98-99% of the registered variance.  60 states per task already give a
  within-task half-width of about .05.  More states do almost nothing for the registered rule (5th set, pi0 profile: half-width .156 at 60
  states, .152 at 120).
- State pairing between arms is weak: rho = .17-.30, where 0 means independent and 1 means comonotone.  Pairing on states helps little.

### 1.3 Seed variance (the one seed pair: RAA-sL vs RAA-sL s43, dev and 5th set)
- N-mean gaps are .056 and .044.  The per-arm seed SD of the task mean is sigma_s = .033 (one degree of freedom; chi-square 95% interval
  .015-1.05, so it is barely constrained).
- On K the gaps are .081 and .061, giving sigma_s = .049.
- The task x seed interaction is .070 per arm on N.
- A contrast between two single-seed arms therefore carries an extra SD of sqrt(2) x .033 = .047 that no registered interval contains.
- Monte Carlo check: a states-only interval without this term gives 19-23% false positives under the null (f = 0).  With the term it gives
  3-5% (the simulation adds the interaction on top, so it is slightly conservative for power).

### 1.4 Monte Carlo power of a new single-seed arm X against a single-seed comparator (300 replicates, B = 800)
Per-task true rates come from the pooled dev + 5th set (60 + 60 states).  X is set to comparator + f x (servo arm - comparator) on every
task, so f = 1 means "the learned owner matches the fixed-form servo".  Each arm gets a seed offset (N(0, .033) on the mean plus N(0, .070)
per task), and outcomes are drawn with the measured pairing rho.  Rules:
- reg: registered tasks+states bootstrap, lower bound > 0.
- prim: the primary rule of section 3.3 (tasks fixed, seed-widened, no flagged task).

| comparator, profile | f = 0 (null) | .25 | .50 | .75 | 1.0 | 1.0 but X fails N2 like H (.07) |
|---|---|---|---|---|---|---|
| pi0: X vs pi0-A0 (full gain +.239) | reg .03 / prim .04 | .24 / .24 | .51 / .59 | .70 / .83 | .91 / .99 | .00 / .01 (flag fires 99%) |
| pi0.5: X vs A0 (full gain +.179) | .05 / .05 | .13 / .14 | .27 / .37 | .37 / .72 | .40 / .92 | .00 / .00 |
| pi0.5: X vs A (full gain +.093) | .07 / .04 | .11 / .09 | .15 / .16 | .19 / .32 | .20 / .48 | .00 / .00 |

Variants on the pi0 profile:

| pi0 variant | reg | prim |
|---|---|---|
| f = .75 with 120 states | .82 | .89 |
| f = .75 with tasks drawn at random from the six profiles | .71 | .80 |
| f = .75 plus two rack pairings with no effect (.45 for both arms) | .51 | .73 (lower power: the extra tasks dilute) |

### 1.5 Analytic half-widths (95%), pi0 profile (5th set), 6 tasks
| rule | 60 states | 120 states |
|---|---|---|
| tasks + states (registered), no seed term | .156 | .152 |
| tasks + states, seed term .033 | .181 | .178 |
| tasks + states, seed term, finite-population correction (6 of 8 pairings) | .128 | .123 |
| tasks fixed, no seed term | .049 | .035 |
| **tasks fixed, seed term .033 (primary)** | **.104** | .098 |
| tasks fixed, seed term .05 (sensitivity) | .147 | .143 |

With the primary rule, power is .47 at a true difference of .10, .81 at .15 and .97 at .20.  With sigma_s = .05 it falls to .27, .52 and
.76.  For pi0.5 against A0 the registered half-width is .18-.21 at any number of states.

### 1.6 Post hoc illustration (NOT a registered read): the existing h1 under the primary rule
- Pooled H - A0 is +.089.  The states-only interval is [+.057, +.121].
- With the seed term the interval is [-.008, +.186].
- N2 is flagged (-.88 pooled).
- So E4 stays NOT SHOWN, and FLAGGED, under the new rule.  The new rule is not a back door for E4, and h1 as registered stands as NOT SHOWN.

---

## 2. Options considered

### 2.1 (a) More held-out pairings or tasks: rejected as the main lever
- LIBERO-Goal's pick-and-place pairings: 3 movables (bowl, cream cheese, wine bottle) x targets (plate, stove, cabinet top, bowl, rack).
  - Trained: bowl -> plate, stove, cabinet; bottle -> cabinet, rack; cheese -> bowl.
  - New: N1-N6 plus cheese -> rack and bowl -> rack.
  - The existing N7 (cheese into the top drawer, a drawer-opening compound) and N8 (push the bowl) carry no channel; the lexicon refuses
    them (raac_plan).
- So at most 2 new pairings exist, both on the rack.
  - The rack's top region is 10.1 x 2.2 cm, tilted 62 degrees, and built for a bottle.
  - It is the weakest target for every arm (K5: RAA-sL .633, A0 .200, H .45, pi0-sL .25).
  - Feasibility (planner >= .80) is untested.
- Monte Carlo: two near-floor pairings with no effect lower power (pi0, f = .75: .83 to .73).
- The finite-population argument (6 of 8 pairings) is real, and it is reported as a sensitivity in 1.5.  It does not replace pairings that
  do not exist.

### 2.2 (b) Within-checkpoint designs
- Every within-checkpoint swap of the module is circular or double-counts:
  - removing the head or servo from its own expert (noW: DEPENDENT, r3 -.503);
  - swapping the servo for a head in RAA-sL's expert;
  - adding a head at serving to A0 or pi0-A0, whose expert already outputs the full action.
- The non-circular pairing already available is the shared start state.  It is weak (rho .17-.30) and already used.
- The real lever in (b) is the estimand.
  - Fix the tasks: the claim is about this scene's pairings, which are nearly exhausted.
  - Put the between-checkpoint (seed) variance into the interval explicitly, from the one measured seed pair.
- That is what the primary rule does.  It narrows the interval from .15 to .10 on pi0 and from .41 to .10 on pi0.5.  It also widens what
  the registered rule ignores: the seed term.

### 2.3 (c) Fixing H's near-target weakness by principle: adopted, inside (d)
**Principle (stated before any new read):** an owner of the where outputs a correction that points to where the object must go and
vanishes only there.  The expert owns the how.

The registered H was fit to E[demo action xyz(t + j) | rel(t)], the demonstrator's velocity.  That quantity belongs to the how:
- It fades where the demonstrator slows or dwells.
- It points along the demonstrated path, not at the goal.

Evidence on training data (raah/fit.json):
- |h| along x at 2 / 5 cm is .017 / .051, against the servo's .057 / .143.
- On the slice rel = -5 cm along x, H's row-0 output is [+.12, -.43, +.22], while the goal lies along -x.  Axis slices may lie off the
  demo manifold, which is why gate G1 below tests this on validation demo rows.

So H never owned the last few centimetres.  The expert, trained on the residual a - E[a | rel], had no reason to own them either.  On N2
neither module finished the approach, and the object stayed held at the target (class H).

**H\* target.** For each held row t of a demo, t_r is the demo's last held row (the frame before the gripper opens).
- The object's remaining displacement is Delta(t) = rel(t) - rel(t_r).  This needs only the channel: rel = target - object, and the target
  does not move.
- Per chunk row j, the target is y_j = clip(K x Delta(min(t + j, t_r)), +-M), with K = 2.851 and M = .9375.  These are the constants that
  behaviour cloning already fixed (raas_gate.json).
- The learned part is where the object is released, as a class-blind function of the token: the learned analogue of W5's mu table.
- The gain is the demo-fitted K.
- Nothing is tuned on held-out data.

What stays chosen by hand, and is stated in the paper: the map from a position error to a translation command is linear with saturation.

### 2.4 (d) The separate-owner test on pi0: adopted
- pi0-A0 is the "network reads the target (.542) and still fails" condition.  A win there cannot be explained as "the pathway was never
  learned", which is the objection to pi0.5's A0 (where-share -.001).
- The servo's gain over pi0-A0 is uniform across tasks (all 12 task differences > 0), so the test is powered (1.4).
- The pi0.5 alternative (H\* vs A0 and A on a 6th set, about 21-24 GPU-h) has power .72 vs A0 and .32 vs A at f = .75 under the primary
  rule, and .37 / .19 under the registered rule.  It is the second choice.

**Smallest decisive design for pi0.5 against A.**
- The gap is about .09, so 80% power needs a standard error of about .033.
- The seed term dominates: 2 x .033^2 / n_seeds + .025^2 <= .033^2 gives n of about 5 seeds per arm.
- Five trainings of H\* (about 11 GPU-h each) plus 4 more seeds of A, plus rollouts: about 120-140 GPU-h.
- Under task-level inference, 80% power at +.18 needs a task SE of about .064, that is (.25 / .064)^2, about 15 pairings.  The scene
  has 8.

---

## 3. The recommended experiment: NPH (tag `nph`)

### 3.1 Arms (all pi0, pi0_libero init, token in state dims 0-3, quantile norm, raw actions, 6,001 steps, batch 32, lr 5e-5, seed 42; as PZ)
| arm | checkpoint | status |
|---|---|---|
| pi0-H\* | pytorch/pi0_g8v_raah_l_ae_tl6000 (new) | pi0-A0's config (servo_k = 0) plus the frozen head H\* in the servo slot; the expert trains on the residual |
| pi0-A0 | pytorch/pi0_g8v_raac_l_ae_tl6000 | exists (comparator: the network learns the where) |
| pi0-sL | pytorch/pi0_g8v_raas_l_ae_tl6000 | exists (reference: fixed-form servo) |

### 3.2 Steps, in order (each new file is a copy; no existing file is edited except l90c_config.py, add-only)
1. **Head fit (CPU sbatch).**  experiments/raah0_fit.py is a copy of raah_fit.py with the target of 2.3.
   - Same loader (raas_gate), same 13,327 held rows, same demo split hash, same MLP 3 -> 64 -> 64 -> 30 (SiLU, M tanh).
   - Same optimiser (AdamW 1e-3, wd 1e-4, full batch, 3000 steps, seed 0, best validation step).
   - NOT centred: its zero is learned.
   - Outputs: lifelong_ood/policy/raah0_head.npz and results/flawed/raah0/fit.json.
2. **Head gate (CPU, validation demos only; thresholds fixed here).**
   - G1 (goal-directed): mean cosine(h_row0(rel(t)), Delta(t)) >= .80 over validation rows with |Delta| >= 2 cm.
   - G2 (no fade): median |h_row0| / min(K |Delta|, M) >= .60 over validation rows with 2 <= |Delta| <= 6 cm.
   - G3 (zero at release): median |h_row0| <= .05 over rows with |Delta| < 1 cm.
   - The same three numbers for the registered head H are reported (descriptive).
   - Any gate fails: STOP.  No refit, no training; report.
3. **Model and config.**
   - lifelong_ood/policy/raah0_model.py: Pi0ZH(RZ.Pi0ZS) whose servo() returns raah_model.head_norm_jax with the npz above, plus the
     PyTorch twin.
   - l90c_config.py add-only entry `pi0_g8v_raah_l_ae`: exactly the `pi0_g8v_raac_l_ae` branch, with model Pi0ConfigZH(head_npz=..., k = 0).
   - Submit the CPU cfgcheck immediately after the edit, and never while a chain that reads the file is about to restart.
4. **Checks (tail-lab, about .4 GPU-h), a copy of raah_check.py retargeted to the pi0 config.**
   - A linear-form npz reproduces servo_norm under pi0's normalisation (< 1e-6).
   - np / jax / torch agree (< 1e-5).
   - The output is zero off role 2.
   - The parameter tree equals pi0-A0's.
   - 20-step smoke: step-0 loss finite and <= 2 x pi0-sL's step-0 loss.
   - Serving smoke folded into the first rollout job: role-2 tokens logged, chunk minus head equals the residual.
5. **Registration.**  experiments/prereg/nph_pred.txt (3.6) and experiments/prereg/nph.py, a copy of rvw5.py.
   - It hashes nph_pred.txt, this plan, raah0_fit.py, raah0_head.npz, fit.json, raah0_model.py, l90c_config.py, fc_n2_w6.py, nph_read.py
     and the job scripts.
   - It asserts that results/flawed/n2_states_w6 and jax/pi0_g8v_raah_l_ae are absent, and records sigma_s = .033 / sigma_i = .070 from
     necpow_4847479.json.
   - Commit at least 60 s before the first sbatch it guards.  The rva1 row landed in the same second as its job, and that was flagged.
6. **Training.**  One chain on tail-lab (c_train_x.py --name pi0_g8v_raah_l_ae; about 4.4 s/step, as the PZ chains), save 250, keep 1000.
   Export tl6000.
7. **6th set (risk 1 of plan_review_atlas).**  experiments/fc_n2_w6.py is a copy of fc_n2_w5.py with K0 = 820 and n2_states_w5 added to
   the refusal list.
   - States k = 820-879, sampler seed 600000 + k, branch seed_base 13000000 + 1000 k.
   - Outputs results/flawed/n2_states_w6 and n2_branches_w6.json.
   - N/K only: no rg_comp set, because Phase B is not used.
   - A disjointness job (copy of rvw5_disjoint.py) against every stored n2 set.
   - Built only after the nph row is committed.
8. **Rollouts (overcap A40, the usual exclude list, raaw5r_roll.sh pattern with FRESH = n2_states_w6).**
   - pi0-H\*'s two K jobs first (tids 0+2+4, 6+8+10).
   - **Gate: pi0-H\* K1-K6 >= .70**, read by nph_read.py --part gate before any N job is submitted.  If it fails: UNTESTABLE, and no N job,
     no comparator job.
   - Then pi0-H\* N (16+18+20, 22+24+26), and pi0-A0 and pi0-sL Phase A (all four groups each).
   - Every server keeps --chan-diag so E1's classifier runs at no cost.
9. **Read.**  experiments/nph_read.py is a copy of rvw5_read.py with BASE results/flawed/nph6, the arms above and the rules of 3.3.
   Outputs are opened with "x".

### 3.3 Registered contrasts and rules
Notation:
- d_ts is the paired difference (task t, state s) on N1-N6 of the 6th set, 60 states.
- d_t is the task mean, and Delta = mean_t d_t.
- v_t = var_s(d_ts) / 60.
- sigma_s = .033 and sigma_i = .070, frozen.

**n1 PRIMARY: pi0-H\* - pi0-A0 on N1-N6.**
- Interval: Delta +- 1.96 sqrt(var_B + 2 sigma_s^2).  var_B is the variance of a states-only bootstrap (tasks fixed; one state index
  shared by all tasks, as the registered routine; B = 2000, rng 0).
- Flag: some task with d_t + 1.96 sqrt(v_t + 2 sigma_s^2 + 2 sigma_i^2) < -.10.  That is, a pairing where H\* is worse by more than seed
  noise explains.
- Labels:
  - **SUPPORTED** iff lower > 0 and no task is flagged.
  - **FLAGGED** iff lower > 0 and some task is flagged.
  - **REVERSED** iff upper < 0.
  - **EQUIVALENT** iff the interval lies inside [-.10, +.10].
  - **NOT SHOWN** otherwise.
- One primary in this row, so no Holm adjustment inside it.  As robustness, Holm over the necessity primaries {d1', r1, h1, n1}, with the
  raw p of n1 from the seed-widened normal, is reported.

Secondary contrasts:
- **n1r** (comparability with h1 and d1'): the same contrast under the registered task+state bootstrap (raa_read.diff scheme).  SUPPORTED
  iff lower > 0.
- **n2:** pi0-H\* - pi0-sL on N, rule as n1.  NON-INFERIOR iff lower > -.10; COST iff upper < -.10.
- **d1'':** pi0-A0 - pi0-sL on N (third replication of "read but not in control").  SUPPORTED iff upper < 0, under both rules.
- **nK:** pi0-H\* - pi0-A0 on K1-K6, NON-INFERIOR iff lower > -.05 (seed-widened with the K value sigma_s = .049).
- Sensitivity: n1 recomputed with sigma_s = .05, and with the finite-population factor (6 of 8 pairings).  Reported, not used for the label.
- E1 classes for pi0-H\*: the class-H share of failures, and the decomposition against pi0-A0.
- Option, if GPU-h allows (3.5): pi0-H\* on the 5th set (comparators exist), giving a pooled 5th + 6th n1p, reported only.

### 3.4 What each outcome means
| n1 | n1r | reading for the claim |
|---|---|---|
| SUPPORTED | SUPPORTED | **A learned separate owner of the where beats the same network learning the same token from the same data, on the backbone where the network demonstrably reads the target, on states no one has seen, consistently over pairings.**  Paper: necessity becomes "separation at matched data", scoped to this recipe (modified pi0, 6k-step fine-tune, LIBERO-Goal, privileged geometry); E4's pi0.5 null is explained by H not being goal-directed, stated as a post hoc diagnosis |
| SUPPORTED | NOT SHOWN | the same, for this scene's pairings; generalisation over pairings is not shown (state the half-widths) |
| FLAGGED | any | average gain, with a pairing where the learned owner collapses that the network handles (the E4 pattern again): no claim of necessity; the paper reports separation as "helps where it works, adds a failure" |
| NOT SHOWN, point >= +.10 | | direction only; the study had power about .5 at +.10; do not upgrade the wording |
| EQUIVALENT | | **refutes necessity of separation on pi0 at this recipe**: a goal-directed learned owner gives no gain over the network that reads the target.  pi0-sL's gain then belongs to the fixed servo (its zero at rel = 0 and its constant chunk), not to separation.  The paper drops "a separate owner is needed" and keeps "a fixed-form controller on exact geometry is sufficient", plus the atlas |
| REVERSED | | refutes: the separate learned owner hurts |
| UNTESTABLE (head gate or K gate fails) | | recipe failure; no claim; the necessity section stays at E2 + E3, scoped as now |

n2 NON-INFERIOR with n1 SUPPORTED adds "the learned owner matches the fixed-form servo".  n2 COST adds "the fixed form adds precision".
d1'' SUPPORTED is a third replication of pi0's "read but not in control".

### 3.5 GPU-h
| item | GPU-h | basis |
|---|---|---|
| head fit + gate, cfgcheck, read, power | 0 (CPU) | |
| checks + 20-step smoke (tail-lab) | .4 | raah_check: .3-.6 |
| pi0-H\* training, 6,001 steps, one chain | 8.0 | PZ chains 4.4 s/step = 7.3 h, plus restarts |
| export | .1 | |
| 6th set N/K build + disjointness | 1.0 | rvw5 build 2.6 included F6/F7 |
| pi0-H\* Phase A, K jobs first (gate), then N | 3.0 | 5th set: .66-.83 per job x 4 |
| pi0-A0 Phase A | 3.0 | |
| pi0-sL Phase A | 3.0 | |
| **core** | **18.5** | |
| contingency (E4 ran about 30% over plan unit costs) | up to +4 | |
| option: pi0-H\* Phase A on the 5th set (pooled n1p) | +3.0 | run only if GPU-h used is <= 19 when the 6th-set read finishes |
| **ceiling** | **25** | |

Ordering saves money on failure:
- If the head gate fails, the spend is 0.
- If the K gate fails, the spend is about 11 (training + checks + build + 2 jobs), and no comparator is rolled.
- Cut first: the option, then pi0-sL's K jobs (-1.5; d1'' and n2 survive on N).
- Never cut: the head gate, pi0-H\*'s K gate, and N for pi0-H\* and pi0-A0.

### 3.6 experiments/prereg/nph_pred.txt (text to write before any training; probabilities are the designer's)
```
NPH (docs/notes/plan_necessity_power.md 3): a learned, goal-directed separate where-head on pi0 vs the network that reads the target.
pi0_g8v_raah_l_ae = pi0_g8v_raac_l_ae exactly (pi0_libero init, token in state dims 0-3, quantile norm, raw actions, data lol/libero_g8v_raac, 6,001 steps,
batch 32, lr 5e-5, seed 42, one chain) with a frozen head H* (raah0_head.npz, sha256 in the row) in the servo slot, servo_k = 0, expert trained on the residual.
H* = MLP 3-64-64-30 fit (raah_fit protocol) to y_j = clip(2.851 * (rel(min(t+j, t_r)) - rel(t_r)), +-.9375) on the 13,327 held rows of K1-K6 (t_r = the demo's last held row);
not centred.  Principle: the owner of the where points to the release point and vanishes only there; the expert owns the how.  Head gate (validation demos):
G1 mean cos(h_row0, Delta) >= .80 (|Delta| >= 2 cm); G2 median |h_row0| / min(K|Delta|, M) >= .60 (2-6 cm); G3 median |h_row0| <= .05 (|Delta| < 1 cm).
States: 6th fresh set n2_states_w6, k = 820-879 (fc_n2_w6.py, raaf_plan 7.1 recipe, N/K only), built after this row.  Arms: pi0-H*, pi0-A0, pi0-sL, Phase A.
Gate: pi0-H* K1-K6 >= .70 on the 6th set, read before any N job is submitted; else UNTESTABLE.
n1 PRIMARY pi0-H* - pi0-A0, N1-N6: Delta +- 1.96 sqrt(var_B + 2 * .033^2) (states-only bootstrap, tasks fixed, B = 2000, rng 0); flag = some task with
d_t + 1.96 sqrt(v_t + 2 * .033^2 + 2 * .070^2) < -.10.  SUPPORTED iff lower > 0 and no flag; FLAGGED iff lower > 0 and a flag; REVERSED iff upper < 0;
EQUIVALENT iff inside +-.10; else NOT SHOWN.  sigma constants from .claude_tmp/pp/necpow/necpow_4847479.json (RAA-sL vs s43), frozen.
Secondary: n1r same contrast, registered tasks+states bootstrap (lower > 0).  n2 pi0-H* - pi0-sL N, NI at -.10 / COST.  d1'' pi0-A0 - pi0-sL N (upper < 0, both rules).
nK pi0-H* - pi0-A0 K, NI at -.05 (sigma_s .049).  Sensitivity sigma_s .05 and FPC 6/8 reported.  E1 classes.  Option n1p pooled 5th + 6th (reported).
STATUS: pi0-A0 and pi0-sL success on dev and the 5th set seen; H's failures (N2 class H) seen and are the reason for the principle; no pi0-H* checkpoint; no 6th-set state exists.
Predictions:
 (p0) head gate G1-G3 all pass: 70%.  The registered head H fails G1: 80%.
 (p1) pi0-H* 6th-set K >= .70: 85%; K point .88.
 (p2) 6th-set N: pi0-A0 .52 (.35-.68); pi0-sL .75 (.60-.88); pi0-H* .72 (.50-.88).
 (p3) n1: SUPPORTED 50%; FLAGGED 7%; NOT SHOWN 30% (point >= .10 in half of these); EQUIVALENT 10%; REVERSED 3%.
 (p4) n1r SUPPORTED 40%.  (p5) n2 NON-INFERIOR 30%; COST 10%.  (p6) d1'' SUPPORTED 80% (fixed rule), 65% (registered rule).
 (p7) class-H share of pi0-H*'s failures <= .30: 65%.  (p8) nK NON-INFERIOR 60%.
Kill: (k1) a head gate fails (no training); (k2) raah0 check fails or step-0 loss NaN / > 2x pi0-sL's; (k3) the disjointness job finds an equal row;
(k4) any rollout job fails on infrastructure (no read until all cells exist).
```

---

## 4. Risks

1. **The seed SD rests on one seed pair (1 df).**  If the true sigma_s is .05, primary power at +.18 falls from about .83 to about .67.
   The rule freezes .033 and reports .05 as a sensitivity.  A second pi0-H\* seed (+11 GPU-h) is the fix, and it is outside 25.
2. **The no-flag condition can turn a real average gain into FLAGGED** if H\* fails one pairing.  That is intended: E4's average (+.089)
   hid a -.93 pairing.  Under the null the flag fires about 1% of the time.
3. **H\* is "the servo law aimed at a learned release point."**  A reviewer may call it the servo plus a table.  The paper must say:
   - what is learned (where the object is released, from demos, class-blind);
   - what is fixed (the linear-saturating map, with gain and clip fitted by behaviour cloning);
   - that the test is separation (a separate goal-directed owner vs the network) and not free form vs fixed form.  E4 already answered
     that question as NOT SHOWN.
4. **The fix was motivated by H's N2 failures on the dev and 5th sets.**  Hence the 6th set, and the gate on training demos only.  The
   paper says the principle came after E4's diagnosis.
5. **pi0's deviations (EE state overwritten, no delta actions; review O2(d)) are shared by all three arms.**  The claim is about this
   modified pi0.
6. **Tasks fixed in the primary.**  n1 makes no claim over pairings beyond this scene; n1r is reported beside it.  One scene; 6 of 8
   possible pairings.
7. **Engineering.**  pi0 uses RZ.Pi0ConfigZS and its own quantile stats.  The head's normalisation must follow the pi0 config.  Check 4's
   identity test catches a mismatch.  PolicyBackboneH must add the head for a pi0 checkpoint; the serving smoke asserts it.
8. **Budget.**  E4 overran plan unit costs by about 30%.  The gate-first order and the option-first cut bound the loss.  The 6th set
   uses the last unclaimed range named in plan_review_atlas risk 1 (k 820-879).  The build asserts distinctness.
9. **Monte Carlo assumptions.**
   - True rates are pooled dev + 5th set.
   - The comparator is re-rolled on the 6th set.
   - Seed noise is added on top of the interaction, which is conservative.
   - The f-scaling assumes the learned owner's gain has the servo's task profile.  If H\* fails differently (a new pairing), the flag
     catches it.

## 5. Hard rules for the executor
- No python on the login node; sbatch / srun only.
- Every job uses explicit -p / -A and the exclude list.
- No training or rollout before the nph row is committed.
- Never open results/l90c/split_sealed.json or key files.
- Outputs are opened with "x".
- No recursive grep / find over the repo or the data directories.
- One capped waiter at a time, stopped before the session ends.
- Never edit a .sh that a queued job will read.
- In --export, use '+' instead of commas.
