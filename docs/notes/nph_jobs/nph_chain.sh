#!/bin/bash
# plan_oracle_pi0 main training chains: 6 x 4 h (afterany), copy of raafp0_chain.sh.  CFG = config name (pi0_g8v_raas_l_ae | pi0_g8v_raac_l_ae | pi05_g8v_raac_lo_ae), LBL = job label,
# CHAIN = tl (tail-lab, default) or main (overcap).  Only sbatch runs here.
J=/coc/flash10/ksingara3/.claude_tmp/pp/nph_train.sh
EX="sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,megabot,ripl-s1,gundam,kitt,nestor,samantha,voltron,jill,gideon,walle,xaea-12,spot,cyborg,heistotron,major,baymax"
S="~--steps~6001~--batch~32~--save-interval~250~--keep-period~1000"
C=${CHAIN:-tl}
if [ "$C" = tl ]; then P="-p tail-lab -A tail-lab"; else P="-p overcap -A overcap"; fi
prev=""
for i in 1 2 3 4 5 6; do
  dep=${prev:+--dependency=afterany:$prev}
  prev=$(sbatch --parsable $dep $P --exclude=$EX -J ${LBL:?LBL}_${C}$i --export=ALL,ARGS=--name~${CFG:?CFG}~--exp~${C}${S} $J)
  echo "$LBL $C$i $prev"
done
