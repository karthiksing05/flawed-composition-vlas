# RAA-F probe run-book (PRIMARY selection criterion, docs/notes/raaf_plan.md 6.1)

Prepared 2026-11-20.  Nothing here has been run on a real route checkpoint.  Validation used only the existing RAA-sL export (reproduction of SR 83) and the 20-step
Route B smoke checkpoint (jax/pi05_g8v_raac_lb_ae/smoke/19, scratch export); see "Validation" at the end.

## What is measured
The learned where-share on the pooled N + K sets (SR 83 design: n2_goal, n2_branches_main trials 4-23; tasks K1 K2 K4 K6 N1 N4 N5 N6; the 3rd held query; swap candidates as SR 83):
the effect of patching only the learned where-input from the counterfactual run, divided by the total effect T of the counterfactual run.  B = 2000, rng 0, clustered bootstrap (a2_read.py's).

| route | model | where-input | servo |
|---|---|---|---|
| A | pi05_g8v_raac_lf_ae | in.tok | none (servo_k = 0) |
| A0 (if trained) | pi05_g8v_raac_l_ae | in.tok | none |
| A+ | config name from the A+ build (not yet in l90c_config.py when this was written) | in.tok | none |
| B | pi05_g8v_raac_lb_ae, crop in right_wrist_0_rgb | in.tokcrop = state token and crop image patched together (one row) | none |
| A+ctrl | none (no token) | not probed | |

PASS iff where-share 95% CI lower bound >= .70 AND the pooled N + K `in.prefix` share point is below RAA-sL's AND the gate (T CI lower bound > 0).
FAILED for the probe iff the CI upper bound < .30.  RAA-sL is recomputed by the same code from results/flawed/a2/probe_raasl_{N,K} (no new RAA-sL capture).

Files (all new; a2_*.py untouched): experiments/a2f_capture.py, a2f_cap_client.py, a2_probe_f.py, raaf_probe_read.py; job scripts experiments/a2f_export.sh, a2f_capture.sh, a2f_probe.sh, a2f_read.sh.
Job logs go to /coc/flash10/ksingara3/.claude_tmp/pp/a2f_*_<name>_<jobid>.out.  All scripts carry `-p overcap -A overcap` and the full `--exclude` list.

Interpretation choices I made where the plan is silent (flag if you disagree):
1. B's `in.tok + in.crop` is read as ONE joint patch (row `in.tokcrop`: token and crop image both from the counterfactual run), which is "patching only the learned where-input".
   The sum of the two single-patch shares is reported as `sum_tok_crop` (descriptive).
2. `in.prefix` is a2_probe.py's definition (all images + prompt from the counterfactual run).  For B that includes the crop image, so B's literal prefix share contains the crop.
   The rule is applied literally; `pass_alt_nocrop` / `prefix_nocrop_share_route` repeat it with the crop image kept clean (descriptive).
3. `in.crop` patches the crop IMAGE at the input (the third image block of the prefix, prefix positions 512-767), as `in.prefix` patches the other images, not hidden states.
4. The counterfactual run's crop is centred on the swapped-in entity (a2f_cap_client.py adds meta/raa_crop_uv_{obj,tgt} to the side queries); without it the server would give the cf run a zero crop.

## Preconditions (once per route, after its training reaches step 6000)
Training chains: main (overcap) and tl (tail-lab) per route; export whichever reached step 6000 (the registered checkpoint is exp main step 6000, tag main6000).
Paths below use `R=/coc/flash10/ksingara3/flawed-composition-vlas`; run from `cd $R`.  Never run python on the login node.

## Step 1: export (CPU, about 10 min)
```
cd /coc/flash10/ksingara3/flawed-composition-vlas
sbatch -J a2fexp_A --export=ALL,ARGS='--g8v~raac_lf~--exp~main~--export~--step~6000~--tag~main6000' experiments/a2f_export.sh   # -> pytorch/pi05_g8v_raac_lf_ae_main6000
sbatch -J a2fexp_B --export=ALL,ARGS='--g8v~raac_lb~--exp~main~--export~--step~6000~--tag~main6000' experiments/a2f_export.sh   # -> pytorch/pi05_g8v_raac_lb_ae_main6000
# A0 (only if trained):  --g8v~raac_l      A+ : --g8v~<its key>  (needs the key in c_train_pi05.py's --g8v choices and the export branch, see below)
```
If a route was trained on the tl chain only, use `--exp~tl`.  If a pytorch dir for that tag already exists, the c_train export overwrites its model.safetensors: check `ls` first.
For A+, `c_train_pi05.py export` converts through raac_export.py only for the listed `--g8v` values; if the A+ key is not listed, use the scratch mode instead:
`sbatch -J a2fexp_Aplus --export=ALL,SRC=<jax step dir>,DST=/coc/flash10/ksingara3/.claude_tmp/raaf/export_Aplus experiments/a2f_export.sh` (DST must be under .claude_tmp) and pass `CKPT=` / `--ckpt` below.

## Step 2: capture (GPU A40, about 1 h each of K and N; a2_cap.sh timing was .25 h x 3)
Two jobs per route with the SAME OUTD (K tids and N tids), exactly as SR 83.  Main refuses without the a2 prereg row (present).
```
cd /coc/flash10/ksingara3/flawed-composition-vlas
O=/coc/flash10/ksingara3/flawed-composition-vlas/results/flawed/raaf/a2
for R in A B; do
  sbatch -J a2fcap_${R}_K --export=ALL,POLICY=$R,PHASE=main,TIDS=0+2+6+10,OUTD=$O/cap_$R  experiments/a2f_capture.sh
  sbatch -J a2fcap_${R}_N --export=ALL,POLICY=$R,PHASE=main,TIDS=16+22+24+26,OUTD=$O/cap_$R experiments/a2f_capture.sh
done
# A0:    POLICY=A0
# A+:    POLICY=Aplus,CONFIG=<A+ config name>,CKPT=<its pytorch dir>   (CONFIG and CKPT are required for Aplus)
```
Serving: A / A0 / A+ run `raa.py serve --arm raac` through a2f_capture.py; B runs `raa_b.py serve --arm raac` (crop in the third image) and a2f_cap_client.py with A2F_RAA=raa_b.
Check: `grep -h "n obs files\|exit=" /coc/flash10/ksingara3/.claude_tmp/pp/a2f_cap_a2fcap_*` and `ls $O/cap_B/obs | wc -l`; each task gives a *_cap.jsonl and obs/*_true.npz plus one *_cf_*.npz per swap candidate.
For B every npz must hold a nonzero `crop` (the probe asserts the key; a zero crop on the true query means no centre was sent).

## Step 3: probe (GPU A40, full rows: 237-row design, about 20 s per observation, 96 observations = about 35 min; 2 h allowed)
`--rows full` is the selection run (it also gives top1 / PRn of the 72 writes and the token-ablation drop, "reported always"); `--rows inputs` is total + in.* only.
```
cd /coc/flash10/ksingara3/flawed-composition-vlas
O=/coc/flash10/ksingara3/flawed-composition-vlas/results/flawed/raaf/a2
for R in A B; do
  sbatch -J a2fp_${R}_K --export=ALL,ARGS="--model~$R~--cap~$O/cap_$R~--out~$O/probe_${R}_K~--sets~K~--rows~full" experiments/a2f_probe.sh
  sbatch -J a2fp_${R}_N --export=ALL,ARGS="--model~$R~--cap~$O/cap_$R~--out~$O/probe_${R}_N~--sets~N~--rows~full" experiments/a2f_probe.sh
done
# A+:  --model~Aplus~--config~<name>~--ckpt~<dir>   (the label must equal the capture's POLICY)
```
Each observation file is skipped if present, so a requeued job resumes.  Row names written: ref, total, the registered write / band rows, in.prefix, in.tok, abl.tok and, for B, in.crop, in.tokcrop, in.prefix_nocrop.

## Step 4: read (CPU, about 10 min with full rows)
```
cd /coc/flash10/ksingara3/flawed-composition-vlas
O=/coc/flash10/ksingara3/flawed-composition-vlas/results/flawed/raaf/a2
Z=/coc/flash10/ksingara3/flawed-composition-vlas/results/flawed/a2
sbatch -J a2fr_A --export=ALL,ARGS="read~--route~A~--dirs~$O/probe_A_N+$O/probe_A_K~--ref-dirs~$Z/probe_raasl_N+$Z/probe_raasl_K~--out~$O/raaf_probe_A.json" experiments/a2f_read.sh
sbatch -J a2fr_B --export=ALL,ARGS="read~--route~B~--dirs~$O/probe_B_N+$O/probe_B_K~--ref-dirs~$Z/probe_raasl_N+$Z/probe_raasl_K~--out~$O/raaf_probe_B.json" experiments/a2f_read.sh
```
The script writes the JSON once (`open(..., "x")`) and refuses to overwrite; to re-read, choose a new filename and say why.  `--crop` defaults to on for route B.  The log prints the verdict line:
`PASS` / `FAILED (share CI upper < .30)` / `NOT PASSED (...)`, plus the share per set, the prefix comparison and, for B, the `pass_alt_nocrop` and `sum_tok_crop` extras.
JSON keys: `rule` (where_share, gate, prefix shares, PASS, verdict), `shares` (every in.* share, pooled and per set, with CI), `ref_shares` (RAA-sL, same code), `full_rows` (top1, PRn, abl.tok drop).

## Decision flow (plan 6.1 / 9)
A first.  A PASS: train A0.  A FAILED (CI upper < .30): A closed (kill k5), A0 not trained.  A+ and B per plan; A+ctrl gets no probe.

## Hooks left for A+ (configs were being added by another agent; none present in l90c_config.py at the time of writing)
* a2_probe_f.py: `--model Aplus --config NAME --ckpt DIR` (registry entry "Aplus" has no defaults); `--crop 1` forces the crop groups if an A+ variant carries a crop image.
* a2f_capture.sh: `POLICY=Aplus CONFIG= CKPT=`; `SERVE` is raa.py (set it to raa_b.py in the script if A+ uses the crop).  A2F_RAA selects the client module.
* raaf_probe_read.py: `--route Aplus` reads in.tok.  A+ctrl: not probed.
* If A+ changes the lexicon / channel, check that `raa.py serve --arm raac` still produces its data's state token before capturing.

## Validation done (existing checkpoints only; outputs in /coc/flash10/ksingara3/.claude_tmp/raaf/probe_smoke/, never results/)
See the results block appended at the bottom by the preparing agent.
