# XB shared shell helpers (sourced by xb_roll_*.sh).  Hard cap 300 GPU-h for all xb_* jobs: wall-hour guard + job time limits.
R=/coc/flash10/ksingara3/robot; B=$R/bench; REPO=/coc/flash10/ksingara3/flawed-composition-vlas
XBOUT=$REPO/results/bench/xb
CAP_H=${CAP_H:-300}
xb_gpu_hours() {   # GPU-h used so far by xb_* jobs (sacct elapsed x allocated GPUs, includes running jobs)
  sacct -u $USER -S 2026-11-26 -X -n -P -o JobName,ElapsedRaw,AllocTRES 2>/dev/null | awk -F'|' '$1 ~ /^xb_/ && $3 ~ /gres\/gpu/ {n=1; if (match($3,/gres\/gpu[^=]*=[0-9]+/)) {s=substr($3,RSTART,RLENGTH); sub(/.*=/,"",s); n=s+0}; h+=$2*n/3600} END{printf "%.2f", h+0}'
}
xb_guard() {       # skip the start when the cap is (nearly) reached
  H=$(xb_gpu_hours); echo "[guard] xb GPU-h so far: $H (cap $CAP_H)"
  awk -v h="$H" -v c="$CAP_H" 'BEGIN{exit !(h+4 >= c)}' && { echo "[guard] cap reached: skipping this job"; exit 0; }
}
xb_wait_port() {   # $1 port, $2 pid, $3 max seconds
  for i in $(seq 1 $(( $3 / 5 ))); do
    (echo > /dev/tcp/127.0.0.1/$1) 2>/dev/null && return 0
    kill -0 $2 2>/dev/null || { echo "[xb] server died"; return 1; }
    sleep 5
  done
  echo "[xb] server timeout"; return 1
}
xb_log_job() { mkdir -p $XBOUT; echo "$(date +%FT%T) $SLURM_JOB_ID $SLURM_JOB_NAME $* elapsed=${SECONDS}s" >> $XBOUT/jobs.log; }
