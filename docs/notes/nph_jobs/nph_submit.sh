#!/bin/bash
# NPH: submit Phase A jobs (n2 N1-N6 + K1-K6 on the 6th set) for ARMS (space-separated) and GROUPS (default all four).  Login node, AFTER the nph row and the set exist.  Prints "A <arm> <group> <jobid>".
#   K groups: 0+2+4 (K1-K3) 6+8+10 (K4-K6)   N groups: 16+18+20 22+24+26.   Hstar K only: GROUPS="0+2+4 6+8+10".
REPO=/coc/flash10/ksingara3/flawed-composition-vlas
PP=/coc/flash10/ksingara3/.claude_tmp/pp
CK=/coc/flash10/ksingara3/robot/checkpoints_l90c/pytorch
B=$REPO/results/flawed/nph6
EXC=sky1,hal,tars,t1000,vicki,johnny5,c3po,olivaw,megabot,ripl-s1,gundam,kitt,nestor,samantha,voltron,jill,gideon,walle,xaea-12,spot,cyborg,heistotron,major,baymax
S="sbatch --parsable -p overcap -A overcap --exclude=$EXC"
for arm in ${ARMS:?}; do
  case $arm in
    pzsL)  X="ARM=raac,NAME=pzsL,RAAPY=raa.py,CKPT=$CK/pi0_g8v_raas_l_ae_tl6000,CONFIG=pi0_g8v_raas_l_ae" ;;
    pzA0)  X="ARM=raac,NAME=pzA0,RAAPY=raa.py,CKPT=$CK/pi0_g8v_raac_l_ae_tl6000,CONFIG=pi0_g8v_raac_l_ae" ;;
    pzH)   X="ARM=raac,NAME=pzH,RAAPY=raa_wh0.py,WHERE_LOG=1,CKPT=$CK/pi0_g8v_raah_l_ae_tl6000,CONFIG=pi0_g8v_raah_l_ae" ;;
    *) echo "unknown arm $arm"; continue ;;
  esac
  for g in ${GROUPS_:-0+2+4 6+8+10 16+18+20 22+24+26}; do
    echo "A $arm $g $($S -J nph_a_${arm}_$g --export=ALL,OUTD=$B/a_$arm,TIDS=$g,$X $PP/nph_roll.sh)"
  done
done
