#!/bin/bash
# wait up to $1 s for the NPH training to reach step 6000 (or the chain to die)
D=/coc/flash10/ksingara3/robot/checkpoints_l90c/jax/pi0_g8v_raah_l_ae/tl
end=$(( $(date +%s) + $1 ))
while [ $(date +%s) -lt $end ]; do
  [ -d $D/6000 ] && { echo "DONE step 6000"; exit 0; }
  n=$(squeue -u $USER -h -n nph_tl1,nph_tl2,nph_tl3,nph_tl4,nph_tl5,nph_tl6 | wc -l)
  [ "$n" = 0 ] && { echo "CHAIN GONE"; exit 1; }
  sleep 30
done
echo "waiting: steps $(ls $D 2>/dev/null | tr '\n' ' ')  jobs $(squeue -u $USER -h -o '%j:%T:%M' -n nph_tl1,nph_tl2,nph_tl3,nph_tl4,nph_tl5,nph_tl6 | tr '\n' ' ')"
