#!/bin/bash
# NPH unattended pipeline: login node, run ONCE (after nph_tl1-6 = 4847565-4847570 are queued).  Appends "<label> <jobid>" lines to nph_pipeline_ids.txt.
PP=/coc/flash10/ksingara3/.claude_tmp/pp; REPO=/coc/flash10/ksingara3/flawed-composition-vlas; CK=/coc/flash10/ksingara3/robot/checkpoints_l90c/pytorch; B=$REPO/results/flawed/nph6
EX=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,megabot,ripl-s1,gundam,kitt,nestor,samantha,voltron,jill,gideon,walle,xaea-12,spot,cyborg,heistotron,major,baymax
OV="-p overcap -A overcap --exclude=$EX"; TL="-p tail-lab -A tail-lab --exclude=$EX"
ids=$PP/nph_pipeline_ids.txt
sb() { lab=$1; shift; j=$(sbatch --parsable "$@"); echo "$lab $j" | tee -a $ids >&2; echo $j; }
G=$PP/nph_gated.sh
exp=$(sb export $OV -J nph_export --dependency=afterany:4847570 -c 8 --mem=64G -t 02:00:00 --export=ALL,KIND=export $G)
pc=$(sb pcheck $TL -J nph_pcheck --dependency=afterany:$exp --gres=gpu:a40:1 -c 8 --mem=64G -t 00:40:00 --export=ALL,KIND=pcheck $G)
roll() { # arm group kind dep  -> job id
  case $1 in
    pzsL) X="ARM=raac,NAME=pzsL,RAAPY=raa.py,CKPT=$CK/pi0_g8v_raas_l_ae_tl6000,CONFIG=pi0_g8v_raas_l_ae" ;;
    pzA0) X="ARM=raac,NAME=pzA0,RAAPY=raa.py,CKPT=$CK/pi0_g8v_raac_l_ae_tl6000,CONFIG=pi0_g8v_raac_l_ae" ;;
    pzH)  X="ARM=raac,NAME=pzH,RAAPY=raa_wh0.py,WHERE_LOG=1,CKPT=$CK/pi0_g8v_raah_l_ae_tl6000,CONFIG=pi0_g8v_raah_l_ae" ;;
  esac
  sb "roll_$1_$2" $OV -J nph_a_$1_$2 --dependency=afterany:$4 --gres=gpu:a40:1 -c 8 --mem=64G -t 01:45:00 --export=ALL,KIND=$3,OUTD=$B/a_$1,TIDS=$2,$X $G
}
k1=$(roll pzH 0+2+4 rollK $pc); k2=$(roll pzH 6+8+10 rollK $pc)
rg=$(sb readgate $OV -J nph_readgate --dependency=afterany:$k1:$k2 -c 4 --mem=24G -t 00:20:00 --export=ALL,KIND=readgate $G)
all=""
for g in 16+18+20 22+24+26; do j=$(roll pzH $g rollN $rg); all="$all:$j"; done
for a in pzA0 pzsL; do for g in 0+2+4 6+8+10 16+18+20 22+24+26; do j=$(roll $a $g rollN $rg); all="$all:$j"; done; done
rm=$(sb readmain $OV -J nph_readmain --dependency=afterany$all -c 4 --mem=24G -t 00:20:00 --export=ALL,KIND=readmain $G)
for a in pzH pzA0 pzsL; do sb e1_$a $OV -J nph_e1_$a --dependency=afterany:$rm -c 4 --mem=24G -t 00:20:00 --export=ALL,KIND=e1arm,ARMX=$a $G; done
sb e1dec1 $OV -J nph_e1d1 --dependency=afterany:$rm -c 4 --mem=24G -t 00:20:00 --export=ALL,KIND=e1dec,ARMA=pzA0,ARMB=pzH $G
sb e1dec2 $OV -J nph_e1d2 --dependency=afterany:$rm -c 4 --mem=24G -t 00:20:00 --export=ALL,KIND=e1dec,ARMA=pzsL,ARMB=pzH $G
sb e1dec3 $OV -J nph_e1d3 --dependency=afterany:$rm -c 4 --mem=24G -t 00:20:00 --export=ALL,KIND=e1dec,ARMA=pzA0,ARMB=pzsL $G
