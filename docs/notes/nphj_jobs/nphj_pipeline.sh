#!/bin/bash
# NPHJ unattended pipeline: login node, run ONCE after the nphj prereg row is committed (>= 60 s).  Appends "<label> <jobid>" lines to nphj_pipeline_ids.txt.  Only sbatch runs here.
# Arms: pzA24 = pi0-A0 recipe, 24,001 steps, seeds 42-43 (exp t24k_s<seed>, 10 links); pzJ = jointly trained where-branch, seeds 42-45 (exp tj_s<seed>, 4 links, gated by checks_ok.marker).  Everything on overcap.
PP=/coc/flash10/ksingara3/.claude_tmp/pp; REPO=/coc/flash10/ksingara3/flawed-composition-vlas; JD=$REPO/docs/notes/nphj_jobs
EX=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,megabot,ripl-s1,gundam,kitt,nestor,samantha,voltron,jill,gideon,walle,xaea-12,spot,cyborg,heistotron,major,baymax
OV="-p overcap -A overcap --exclude=$EX"
ids=$PP/nphj_pipeline_ids.txt
sb() { lab=$1; shift; j=$(sbatch --parsable "$@"); echo "$lab $j" >> $ids; echo "$lab $j" >&2; echo $j; }
G=$JD/nphj_gated.sh
# 0. checks (identity, 30-step smoke, export parity) -> checks_ok.marker; J training links wait for it (afterany) and exit 0 without the marker
ck=$(sb check $OV -J nphj_check $JD/nphj_check.sh)
declare -A LAST EXPJ
ALLR=""; declare -A N_JOBS
roll() { # arm seed kind tids dep -> id
  local dep=""; [ -n "$5" ] && dep="--dependency=afterany:$5"
  sb roll_$1_$2_$3_$4 $OV -J nphj_a_$1_$2_$4 $dep --gres=gpu:a40:1 -c 8 --mem=64G -t 01:45:00 --export=ALL,KIND=$3,ALAB=$1,SEED=$2,TIDS=$4 $G
}
for spec in pzA24:42 pzA24:43 pzJ:42 pzJ:43 pzJ:44 pzJ:45; do
  IFS=: read a s <<< "$spec"
  if [ $a = pzA24 ]; then
    LAST[$a$s]=$(CFG=pi0_g8v_raac_l_ae SEED=$s LBL=${a}_$s EXP=t24k_s$s STEPS=24001 KEEP=6000 NL=10 bash $JD/nphj_chain.sh 2>>$ids)
  else
    LAST[$a$s]=$(CFG=pi0_g8v_raaj_l_ae SEED=$s LBL=${a}_$s EXP=tj_s$s STEPS=6001 KEEP=1000 NL=4 DEP0=$ck bash $JD/nphj_chain.sh 2>>$ids)
  fi
  echo "train_${a}_$s ${LAST[$a$s]}" >> $ids
  ex=$(sb export_${a}_$s $OV -J nphj_export_${a}$s --dependency=afterany:${LAST[$a$s]} -c 8 --mem=64G -t 02:00:00 --export=ALL,KIND=export,ALAB=$a,SEED=$s $G)
  k1=$(roll $a $s rollK 0+2+4 "$ex"); k2=$(roll $a $s rollK 6+8+10 "$ex")
  rg=$(sb readgate_${a}_$s $OV -J nphj_readgate_${a}$s --dependency=afterany:$k1:$k2 -c 4 --mem=24G -t 00:20:00 --export=ALL,KIND=readgate,ALAB=$a,SEED=$s $G)
  n1=$(roll $a $s rollN 16+18+20 "$rg"); n2=$(roll $a $s rollN 22+24+26 "$rg")
  ALLR="$ALLR:$k1:$k2:$rg:$n1:$n2"; N_JOBS[$a$s]="$n1:$n2"
done
rm=$(sb readmain $OV -J nphj_readmain --dependency=afterany$ALLR -c 4 --mem=24G -t 00:30:00 --export=ALL,KIND=readmain $G)
# E1 per (arm, seed), then pooled per arm
for a in pzA24 pzJ; do
  e=""; seeds="42 43 44 45"; [ $a = pzA24 ] && seeds="42 43"
  for s in $seeds; do e="$e:$(sb e1_${a}_$s $OV -J nphj_e1_$a$s --dependency=afterany:${N_JOBS[$a$s]} -c 4 --mem=24G -t 00:20:00 --export=ALL,KIND=e1arm,ALAB=$a,SEED=$s $G)"; done
  sb e1pool_$a $OV -J nphj_e1pool_$a --dependency=afterany$e -c 4 --mem=24G -t 00:20:00 --export=ALL,KIND=e1pool,ALAB=$a $G
done
