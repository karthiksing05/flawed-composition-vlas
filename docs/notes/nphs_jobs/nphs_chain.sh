#!/bin/bash
# NPHS training chain: 6 x 4 h (afterany) of nphs_train.sh for one (CFG, SEED).  Env: CFG (pi0_g8v_raah_l_ae|raac_l_ae|raas_l_ae), SEED (43-45), LBL (label, e.g. pzH43), CHAIN = tl (tail-lab) | main (overcap).
# Prints the id of the LAST link on stdout.  Resumable: exp dir tl_s<SEED>.  DEP0 = optional dependency of the first link.
J=/coc/flash10/ksingara3/flawed-composition-vlas/docs/notes/nphs_jobs/nphs_train.sh
EX="sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,megabot,ripl-s1,gundam,kitt,nestor,samantha,voltron,jill,gideon,walle,xaea-12,spot,cyborg,heistotron,major,baymax"
S="~--steps~6001~--batch~32~--save-interval~250~--keep-period~1000"
C=${CHAIN:-tl}
if [ "$C" = tl ]; then P="-p tail-lab -A tail-lab"; else P="-p overcap -A overcap"; fi
prev=""
for i in 1 2 3 4 5 6; do
  dep=${prev:+--dependency=afterany:$prev}
  prev=$(sbatch --parsable $dep $P --exclude=$EX -J nphs_tr_${LBL:?LBL}_$i --export=ALL,ARGS=--name~${CFG:?CFG}~--exp~tl_s${SEED:?SEED}~--seed~${SEED}${S} $J)
  echo "$LBL link$i $prev" >&2
done
echo $prev
