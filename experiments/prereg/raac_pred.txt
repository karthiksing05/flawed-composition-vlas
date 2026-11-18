RAA-c (SR 77): RAA-o with the role channel entering as a CONTINUOUS input, not text.  Written 2026-11-18 BEFORE the prereg row, any main training job of
RAA-c, and any RAA-c rollout.  Known when written: SR 76 (RAA-o N1-N6 .594 against RAA .656, W1 NOT SUPPORTED; K1-K6 .881; rgp .897 on N1-N6), its exploratory
mechanism checks (docs/SCAFFOLD_RESULTS.md 76, results/flawed/raao_instr.json): the in-hand offset stopped explaining the failed-release xy error (N4 r .14, N6 .03
against .57 and .51) but the error did not shrink (median N4 6.5 cm, N6 7.0 cm, RAA 7.0 and 7.9) although the target channel at test has median error 1.15 cm.
So the remaining failure is aim: a release 3-7 cm off a target the channel locates to about 1 cm.

Hypothesis (one).  A number written as text ("+12 -3 +5 cm") is read imprecisely by the action expert: the frozen VLM turns digits into token embeddings and the
expert reads them through the prefix KV cache, a path with no pressure to be metric.  Giving the same four numbers as a continuous input should let the action
expert use them at their real resolution.

The change (the only one).  RAA-c = RAA-o (pi05_base, frozen VLM, action expert only, the same 8 known tasks, demos, renders and R2 role views, the same
object-relative held channel, 6,001 steps, batch 32, lr 5e-5, 1,000 warmup, no EMA, one seed, one A40) with ONE difference: the channel does not enter as the
text suffix ". rel: +12 -3 +5 cm, role target".  The prompt is the plain role phrase (R2's).  The channel enters as the model's continuous STATE vector
[dx, dy, dz, role id] and is embedded by ONE new linear layer state_proj (32 -> 1024, the action expert's width, as pi0's) into ONE new suffix token placed before the
action tokens, causal as pi0's state token (image and language tokens do not attend to it).
* dx, dy, dz: exactly RAA-o's channel values in metres, float32, NOT rounded to whole cm (the text rounds to 1 cm; this is the point of the arm, the only
  information difference, and it is resolution only: both come from the same estimates).  role id: 1 object, 2 target (while held).  A frame with no
  channel (K7, K8, N7, N8, or a query where the lexicon refuses) carries [0, 0, 0, 0].  The robot's own 8-dim state (eef, gripper) is NOT given (RAA-o's text did not carry
  it either); the state vector is exactly the channel.
* Normalisation: openpi's usual quantile normalisation with the dataset's own stats, then zero-padding to 32 (so the 28 padded inputs are always 0).
* New parameters: state_proj's kernel and bias only (33,792), randomly initialised, trained with the action expert; every other tensor starts from pi05_base and the
  frozen VLM stays frozen.  No openpi file is edited: lifelong_ood/policy/raac_model.py subclasses the config / JAX model / PyTorch model; the export adds the one
  tensor (experiments/raac_export.py).  It adds no information: same estimates, same offset, same roles, more resolution.
* Training: experiments/sc_rv.py arm raac (the channel values are raao's, asserted equal by experiments/raac_check.py).  Test: experiments/raa.py --arm raac
  (the Estimator's values, unrounded; the server sets the state and leaves the prompt the role phrase).

Registered contrasts (read: experiments/raac_read.py, the same 60 states as SR 73 and 76 = dev trials 4-23 + new k 540-579, bootstrap B = 2000, rng 0, raa_read /
raao_read machinery extended not changed; its selftest reproduces SR 73's V1a, V1b and SR 76's W1a, W1b, WK exactly):
U1  primary RAA-c - RAA-o on N1-N6: U1a success, U1b placed given right grasp.  SUPPORTED iff both lower bounds > 0; PARTIAL iff one; else NOT SUPPORTED.
U2  RAA-c - RAA, N1-N6 success: help (lower bound > 0).
U3  RAA-c - rgp, N1-N6 success, non-inferiority margin -.10 (NON-INFERIOR iff lower bound > -.10), reported either way; U3b = beats.
UK  RAA-c - RAA-o on K1-K6 success, non-inferior at -.05.   Fit gate as SR 73 (R0 on K1-K6 >= .80).
Reported with no label: RAA-c - R2, RAA-c - R0 (K1-K6), cheese vs bottle pairings, per-task success / right-grasp / placed-given-right-grasp, channel error per query.
Exploratory (labelled, run after the read, NOT registered): median failed-release xy error and the correlation with the in-hand offset on N1, N4, N6 (K5), via
raa_instr.py on the same 20 states (trials 24-43), as SR 76's check.

Predictions (before data; probabilities are mine).
1. U1 primary: SUPPORTED 22%, PARTIAL 13%, NOT SUPPORTED 65%.  Point estimate of U1a about +.05 (RAA-c N1-N6 about .64, range .45-.82).  Why not higher: the
   remaining failure may be the policy's aim (SR 76), which a more exact number does not change, and a randomly initialised token that must be learned from 8 tasks in
   6,000 steps may be under-used.  Why not lower: the text path is plausibly the bottleneck and the continuous token is a direct, short route to every action token.
2. U1a lower bound > 0 alone: 22%.  Bottle pairings gain more than cheese (X3 > X2 in the point estimate): 55%.
3. U2 (RAA-c beats RAA, N1-N6): 22%.
4. U3 (RAA-c - rgp): NON-INFERIOR (lower bound > -.10) 7%; beats rgp 1%.  rgp is about .90 on N1-N6.
5. UK: RAA-c non-inferior to RAA-o on K1-K6 (lower bound > -.05): 60%.  The new token adds a path the known tasks can overfit through; R0's .85 K fit gate holds regardless.
6. Plumbing: the 12-task read finishes with no episode in which the server held a routed query without a channel (noch = 0 on held target queries) and channel
   error at test the same as SR 76's (held target channel median under 2 cm): 85%.
7. Exploratory mechanism (labelled): the median failed-release xy error on N4 is at most 4.5 cm (RAA-o 6.5): 25%; on N1 and N4 both at most 2.5 cm: 12%.  If U1a's
   lower bound is <= 0 and these errors stay near 6-7 cm, the hypothesis (text numbers are read imprecisely) is not supported at this recipe: the policy's aim is not
   limited by the number's format.
What would count against the claim: U1a lower bound <= 0 together with a failed-release xy error on N4 and N6 within 1 cm of RAA-o's (6.5 and 7.0 cm).  What would
count for it: U1a lower bound > 0 with a median failed-release xy error below 5 cm on N4.  A U1 gain with unchanged release error would mean the continuous token helps
for another reason (for example it also removes the suffix's disruption of the language prefix).  A loss on K1-K6 with a gain on N1-N6 would be a trade, not a fix.
