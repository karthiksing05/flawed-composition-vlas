## SR 78 diagnosis: why RAA-s collapsed (diagnosis only; no model code, checkpoint or registered result changed)

**Cause (evidenced): the servo was never added at serving.** `experiments/raa.py` (serve_cmd, line ~133) wraps the checkpoint in `PolicyBackbone` and `GuidedPolicy`; `GuidedPolicy.infer` (lifelong_ood/policy/serve.py ~lines 98-118) samples with `self.bk.sample(...)` and `self.bk.unnormalize(...)`. `PolicyBackbone.sample` (lifelong_ood/policy/backbone.py:252-298) is its own Euler loop on `v_full`; it never calls `model.sample_actions`. The servo lives only in `PI0PytorchS.sample_actions` (raas_model.py, last method) and `Pi0S.sample_actions`. `grep servo|raas` finds nothing in backbone.py or serve.py (experiments/raas_diag.py prints False for both). So the served policy emits the flow-matching RESIDUAL (trained on action - servo) and the servo term, which carries the held-phase transport direction, is dropped. Training used the servo (the JAX loss), so there is a train/serve mismatch of exactly the missing term.

**Why check 4 and the smoke did not catch it.** raas_check.py check 4 (job 4799403, re-run: passes) calls `PI0PytorchS.sample_actions` directly, a path serving does not use. The smoke tested plumbing only (2 of 4 episodes succeeded). Selftest SR 78 also never exercised the served path.

**Training was normal.** Loss curves raas vs raac (logs .claude_tmp/pp/raas_train_*.out, raac_train_*.out): step 0 loss .0991 vs .0987; step 400 .0343 vs .0339; step 2000 .0250 vs .0248; step 4950 .0217 vs .0217; step 5950 .0211 vs about .0218 (raac step 5850). Same grad_norm and param_norm trajectory. No training problem.

**Servo itself is correct** (no sign, scale, or normalisation error): check 2/3/4 agree with a numpy reference; unnormalised servo = K*clip(rel), (x+1)/2 affine offset cancels. Servo is a constant over the 10-step chunk in both training (`actions - servo[:,None,:]`) and the PyTorch path, so chunk handling is consistent. `m` is read from the same normalised state at train and serve.

**Offline check on the real served path (job 4799409, experiments/raas_diag.py, 15 held-target demo frames from episodes 0, 40, 80, 120, 200, same noise).** Mean |xyz - demo| over a 10-step chunk, and norm of the mean chunk bias:
| served chunk | mean abs err | mean-bias norm |
|---|---|---|
| RAA-s as served (no servo, what the rollouts ran) | .297 | .572 |
| RAA-s + servo (what training assumed) | .176 | .292 |
| RAA-c served | .167 | .279 |
Adding the missing servo brings RAA-s to RAA-c's level (the expert plus servo is sane); without it the held-phase chunk is biased by about .29 action units against the demo (mean |servo| .234 against mean |demo xyz| .388; per-frame servo up to .57), e.g. ep 0 t 53: demo x .94, served .57, served+servo .78, raac .90. This is a systematic under-command of motion toward the target during transport, which matches grasp fine (m = 0 before the grasp, servo zero), placed-given-grasp .029, and large release errors, on trained tasks as well.

**Not yet done:** a closed-loop rollout with the servo added in `PolicyBackbone`/`GuidedPolicy` was not run (it needs a model/serving code change, out of scope here). The prediction is that RAA-s then behaves like RAA-c plus a servo; SR 78's NOT SUPPORTED verdict is a verdict on the served-without-servo system, not on the hypothesis. A re-serve is the registered way to test the hypothesis; the SR 78 row should not be quoted as evidence against the servo.

**Fix sketch (not applied):** add the servo in the normalised rows inside `PolicyBackbone.sample` (after the Euler loop, dims 0-2 of each of the H steps, from `prefix`/observation.state via `raas_model.servo_norm_torch`) when the model config has `servo_k`; or have GuidedPolicy call it before `unnormalize`. Then also check guided/composed paths (`sample_composed`, `sample_poe`) if used with raas.
