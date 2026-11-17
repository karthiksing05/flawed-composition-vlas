RAA-o (SR 76): RAA with the target-role channel measured from the held object.  Written 2026-11-17 BEFORE the prereg row, any training job of
RAA-o, and any RAA-o rollout.  Known when written: SR 73 (RAA read: N1-N6 .656, bottles N4-N6 .467, cheese N1-N3 .844, K1-K6 .839, rgp .897 on N1-N6), the
exploratory place-step diagnostic (docs/notes/raa_instr_results.md, results/flawed/raa_instr.json, commit 88858efb): RAA's remaining failures are
release imprecision; the target-role channel is target - END EFFECTOR but a held bottle sits 6-10 cm off the gripper (N4 hand offset 6.6 cm in
failures vs rgp 1.2), and release xy error tracks the in-hand offset (N4 r = .59) or the aim; rgp registers the in-hand offset at the grasp and releases
within about 1-2 cm.  That diagnostic was exploratory; this is the confirmatory test of the change it points to.

The change (the only one).  RAA-o = RAA (pi05_base, frozen VLM, action expert only, the same 8 known tasks and demos, 6,001 steps, batch 32, lr 5e-5, 1,000
warmup, no EMA, one seed, the same R2 role views, the same text-suffix format '<prompt>. rel: +12 -3 +5 cm, role target', whole cm, base-frame axes, the same
resolution) with ONE difference: while the object is held (role target) the channel is  target - HELD OBJECT  = target - (eef + o), where o = the grasped
object's position minus the eef at the hold's onset (the row ProprioHeld turns on), held fixed until the hold ends (recomputed at a later onset).  The object role
(before the hold) is unchanged.  No release gate, no controller logic; the policy still decides when to release.  It adds no information: both terms are
RAA's own inputs / estimates.
* Training (experiments/sc_rv.py, arm raao, helper held_target_rel): o from the demo's own recorded slot positions and ProprioHeld rows, the grasped slot g
  from pp_p_train.demo_sites exactly as RAA's channel (no simulator information beyond what demo_sites already reads).
* Test (experiments/raa.py --arm raao): o = sc_pu.Estimator's object estimate minus the eef at the query on which the client's ProprioHeld fires, then held
  fixed; the target estimate is the Estimator's.  The simulator's positions are read only for the per-query error log.

Registered contrasts (read: experiments/raao_read.py, same 60 states as SR 73 = dev trials 4-23 + new k 540-579, same bootstrap B = 2000, rng 0, same raa_read
machinery extended not changed; its selftest reproduces SR 73's V1a and V1b):
W1  primary RAA-o - RAA on N1-N6: W1a success, W1b placed given right grasp.  SUPPORTED iff both lower bounds > 0; PARTIAL iff one; else NOT SUPPORTED.
W2  RAA-o - rgp, N1-N6 success, non-inferiority margin -.10 (NON-INFERIOR iff lower bound > -.10), reported either way; W2b = beats.
WK  RAA-o - RAA on K1-K6 success, non-inferior at -.05.   Fit gate as SR 73 (R0 on K1-K6 >= .80).
Reported with no label: RAA-o - R2, cheese vs bottle pairings, per-task success / right-grasp / placed-given-right-grasp, channel error per query.

Mechanism checks (exploratory, the same instrumentation as the diagnostic, tasks N1, N4, N6, K5 on trials 24-43, run after the read):
D1 correlation between release xy error and the in-hand hand offset on failed releases.   D2 median failed-release xy error on N1 and N4.

Predictions (before data; probabilities are mine).
1. W1 primary: SUPPORTED 45%, PARTIAL 12%, NOT SUPPORTED 43%.  Point estimate of W1a about +.10 (RAA-o N1-N6 about .75, range .62-.88); bottle pairings gain
   more than cheese (X3 > X2 in the point estimate 70%): the bottle's offset is large, the cheese's small.
2. Mechanism D1: the correlation of release xy error with hand offset on failed releases (N4) drops below .2 in absolute value (the diagnostic's r = .59): 45%.
3. D2: the median failed-release xy error is at most 2.5 cm on both N1 and N4: 40%.  (If the gain is smaller than predicted, failures are few and this is noisy.)
4. Success: RAA-o N1-N6 >= .78 with N4 >= .55 (RAA: N4 about .25 in the diagnostic's 20 states): 35%.
5. K5 (wine bottle on the rack, a known task whose failures are the rack's `under` term and not the offset): RAA-o K5 success stays <= .5 (the channel change does not
   fix K5): 60%.  I read the diagnostic's 'K5 <= .5' this way (K5 success, RAA 7/20 there); counted on the pooled 60 states of the read.
6. WK: K1-K6 non-inferior to RAA at -.05 (lower bound > -.05): 65%.  The target channel while held changes in K tasks too; trained on these tasks it should fit.
7. W2 (RAA-o - rgp): NON-INFERIOR (lower bound > -.10) 20%; RAA-o beats rgp (W2b) 2%.  rgp is about .90 on N1-N6.
8. Channel error at test (smoke and main): the held-object offset estimate error (estimate vs simulator, at onset) has median under 2 cm; the target channel error while
   held under 3 cm median (the offset's error adds to the target's).
What would count against the claim: W1a lower bound <= 0 and RAA-o's failed-release xy error still correlated with the hand offset (|r| >= .4): then the in-hand offset
is not the release imprecision's cause at this model's resolution (1 cm digits in a text prompt), and the aim is.  A W1 gain with D1 unchanged would mean the channel
helps for another reason.
