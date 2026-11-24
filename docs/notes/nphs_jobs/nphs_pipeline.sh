#!/bin/bash
# NPHS unattended pipeline: login node, run ONCE after the nphs prereg row is committed (>= 60 s).  Appends "<label> <jobid>" lines to nphs_pipeline_ids.txt.  Only sbatch runs here.
PP=/coc/flash10/ksingara3/.claude_tmp/pp; REPO=/coc/flash10/ksingara3/flawed-composition-vlas; JD=$REPO/docs/notes/nphs_jobs
EX=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,megabot,ripl-s1,gundam,kitt,nestor,samantha,voltron,jill,gideon,walle,xaea-12,spot,cyborg,heistotron,major,baymax
OV="-p overcap -A overcap --exclude=$EX"
ids=$PP/nphs_pipeline_ids.txt
sb() { lab=$1; shift; j=$(sbatch --parsable "$@"); echo "$lab $j" >> $ids; echo "$lab $j" >&2; echo $j; }
G=$JD/nphs_gated.sh; CPU=$PP/pz_cpu.sh
# 1. seventh set (afterwards: disjointness -> geometry); rollouts depend on the disjointness job
st=$(sb states $OV -J nphs_states --gres=gpu:a40:1 -c 4 --mem=32G -t 02:00:00 --export=ALL,WHAT=n2 $JD/nphs_states.sh)
dj=$(sb disjoint $OV -J nphs_disjoint --dependency=afterok:$st --export=ALL,VENV=openpi,CMD=experiments/nphs_disjoint.py $CPU)
ge=$(sb geom $OV -J nphs_geom --dependency=afterok:$dj --export=ALL,VENV=libero,CMD=experiments/nphs_geom.py $CPU)
# 2. trainings (new seeds): 4 chains on tail-lab, 5 on overcap
declare -A CFG=([pzH]=pi0_g8v_raah_l_ae [pzA0]=pi0_g8v_raac_l_ae [pzsL]=pi0_g8v_raas_l_ae)
declare -A LAST EXPJ
for spec in pzH:43:tl pzH:44:tl pzH:45:tl pzA0:43:tl pzA0:44:main pzA0:45:main pzsL:43:main pzsL:44:main pzsL:45:main; do
  IFS=: read a s c <<< "$spec"
  LAST[$a$s]=$(CHAIN=$c CFG=${CFG[$a]} SEED=$s LBL=$a$s bash $JD/nphs_chain.sh 2>>$ids)
  echo "train_${a}_$s ${LAST[$a$s]}" >> $ids
  EXPJ[$a$s]=$(sb export_${a}_$s $OV -J nphs_export_${a}$s --dependency=afterany:${LAST[$a$s]} -c 8 --mem=64G -t 02:00:00 --export=ALL,KIND=export,ALAB=$a,SEED=$s $G)
  if [ $a = pzH ]; then EXPJ[$a$s]=$(sb pcheck_$s $OV -J nphs_pcheck$s --dependency=afterany:${EXPJ[$a$s]} --gres=gpu:a40:1 -c 8 --mem=64G -t 00:40:00 --export=ALL,KIND=pcheck,ALAB=$a,SEED=$s $G); fi
done
# 3. rollouts
declare -A N_JOBS
ALLR=""
roll() { # arm seed kind tids dep -> id
  local dep=afterok:$dj; [ -n "$5" ] && dep=$dep,afterany:$5
  sb roll_$1_$2_$3_$4 $OV -J nphs_a_$1_$2_$4 --dependency=$dep --gres=gpu:a40:1 -c 8 --mem=64G -t 01:45:00 --export=ALL,KIND=$3,ALAB=$1,SEED=$2,TIDS=$4 $G
}
for s in 42 43 44 45; do
  for a in pzH pzA0 pzsL; do
    d=""; [ $s != 42 ] && d=${EXPJ[$a$s]}
    k1=$(roll $a $s rollK 0+2+4 "$d"); k2=$(roll $a $s rollK 6+8+10 "$d"); ALLR="$ALLR:$k1:$k2"
    if [ $a = pzH ]; then
      rg=$(sb readgate_$s $OV -J nphs_readgate$s --dependency=afterany:$k1:$k2 -c 4 --mem=24G -t 00:20:00 --export=ALL,KIND=readgate,ALAB=pzH,SEED=$s $G); ALLR="$ALLR:$rg"; nd=$rg
    else nd="$d"; fi
    n1=$(roll $a $s rollN 16+18+20 "$nd"); n2=$(roll $a $s rollN 22+24+26 "$nd"); ALLR="$ALLR:$n1:$n2"
    N_JOBS[$a$s]="$n1:$n2"
  done
done
rm=$(sb readmain $OV -J nphs_readmain --dependency=afterany$ALLR -c 4 --mem=24G -t 00:30:00 --export=ALL,KIND=readmain $G)
# 4. E1 per (arm, seed), then pooled per arm
for a in pzH pzA0 pzsL; do
  e=""
  for s in 42 43 44 45; do e="$e:$(sb e1_${a}_$s $OV -J nphs_e1_$a$s --dependency=afterany:${N_JOBS[$a$s]},afterok:$ge -c 4 --mem=24G -t 00:20:00 --export=ALL,KIND=e1arm,ALAB=$a,SEED=$s $G)"; done
  sb e1pool_$a $OV -J nphs_e1pool_$a --dependency=afterany$e -c 4 --mem=24G -t 00:20:00 --export=ALL,KIND=e1pool,ALAB=$a $G
done
