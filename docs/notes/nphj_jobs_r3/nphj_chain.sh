#!/bin/bash
# NPHJ training chain: NL x 4 h (afterany) of nphj_train.sh for one (CFG, SEED).  Env: CFG, SEED, LBL (e.g. pzA24_42), EXP (tl exp name), STEPS (--steps value), KEEP (--keep-period), NL (links), DEP0 (optional dependency of link 1).
# Prints the id of the LAST link on stdout.  Resumable: the exp dir; a link with the final step present exits 0 without work.  All on overcap.
J=/coc/flash10/ksingara3/flawed-composition-vlas/docs/notes/nphj_jobs_r3/nphj_train.sh
EX="sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,megabot,ripl-s1,gundam,kitt,nestor,samantha,voltron,jill,gideon,walle,xaea-12,spot,cyborg,heistotron,major,baymax"
S="~--steps~${STEPS:?}~--batch~32~--save-interval~250~--keep-period~${KEEP:?}"
prev=""
for i in $(seq 1 ${NL:?}); do
  dep=${prev:+--dependency=afterany:$prev}; [ -z "$prev" ] && [ -n "${DEP0:-}" ] && dep="--dependency=afterany:$DEP0"
  prev=$(sbatch --parsable $dep -p overcap -A overcap --exclude=$EX -J nphj_tr_${LBL:?LBL}_$i --export=ALL,ARGS=--name~${CFG:?CFG}~--exp~${EXP:?EXP}~--seed~${SEED}${S} $J)
  echo "$LBL link$i $prev" >&2
done
echo $prev
