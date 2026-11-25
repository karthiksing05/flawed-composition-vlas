#!/bin/bash
# NPHP training chain: 6 x 4 h (afterany) of nphp_train.sh for one (CFG, SEED).  Env: CFG (pi05_g8v_raah0_l_ae|raac_l_ae|raas_l_ae), SEED (43-45), LBL (label, e.g. pqH43), CHAIN = tl (tail-lab) | main (overcap).
# Prints the id of the LAST link on stdout.  Resumable: exp dir tl_s<SEED>.  DEP0 = job id the first link needs afterok.
J=/coc/flash10/ksingara3/flawed-composition-vlas/docs/notes/nphp_jobs/nphp_train.sh
EX="sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,megabot,ripl-s1,gundam,kitt,nestor,samantha,voltron,jill,gideon,walle,xaea-12,spot,cyborg,heistotron,major,baymax"
S="~--steps~6001~--batch~32~--save-interval~250~--keep-period~1000"
C=${CHAIN:-tl}
if [ "$C" = tl ]; then P="-p tail-lab -A tail-lab"; else P="-p overcap -A overcap"; fi
prev=""
for i in 1 2 3 4 5 6; do
  dep=${prev:+--dependency=afterany:$prev}; [ $i = 1 ] && [ -n "${DEP0:-}" ] && dep=--dependency=afterok:$DEP0
  prev=$(sbatch --parsable $dep $P --exclude=$EX -J nphp_tr_${LBL:?LBL}_$i --export=ALL,ARGS=--name~${CFG:?CFG}~--exp~tl_s${SEED:?SEED}~--seed~${SEED}${S} $J)
  echo "$LBL link$i $prev" >&2
done
echo $prev
