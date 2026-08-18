#!/bin/bash
# Launch the real-data profiling sweep on slurm.
#
#   bash submit.sh prep     # one prep job per dataset (writes the shared prepped .h5ad)
#   bash submit.sh sweep    # one training job per config, dependent on prep
#
# Configs run as SEPARATE jobs so each gets its own GPU. Two configs sharing one GPU would
# contend and the timings would measure the contention, not the configuration -- which already
# happened once here: an unrelated job holding 61GB made a training measurement look 2x slower.
set -euo pipefail

ROOT=/data1/shahs3/users/ceglian
REPO=$ROOT/codebase/tcri
WORK=$ROOT/tcri_bench
PART=componc_gpu_batch
ACCT=mcphera1
mkdir -p "$WORK/logs" "$WORK/results"

submit () {                       # submit <name> <deps> <gres> <cpus> <mem> <time> <cmd>
  local name="$1" deps="$2" gres="$3" cpus="$4" mem="$5" tlimit="$6" cmd="$7"
  local dep=""
  [ -n "$deps" ] && dep="--dependency=afterok:$deps"
  sbatch --parsable -J "$name" -p "$PART" -A "$ACCT" $dep \
    --gres="$gres" -c "$cpus" --mem="$mem" -t "$tlimit" \
    -o "$WORK/logs/$name.out" -e "$WORK/logs/$name.err" \
    --wrap "source ~/.bashrc; conda activate tcri-gpu; cd $REPO; export PYTHONPATH=.; $cmd"
}

case "${1:-}" in
prep)
  # smith_all is already 582 genes -- no HVG, just the clonotype column if it needs one
  A=$(submit prep_small "" gpu:1 4 64G 2:00:00 \
    "python benchmarks/cluster/prep_real.py --in $ROOT/smith_all.h5ad \
       --out $WORK/small.h5ad --n-top-genes 100000 --clonotype-key trb_unique \
       --target-col trb_unique --min-clone-size 10")
  # the big one: 525k x 35,683 -> 2000 HVG. Needs RAM for the load and a GPU for rapids HVG.
  B=$(submit prep_big "" gpu:1 8 320G 8:00:00 \
    "python benchmarks/cluster/prep_real.py --in $ROOT/notebooks/smith_new.h5ad \
       --out $WORK/big.h5ad --n-top-genes 2000 --clonotype-key trb \
       --target-col trb_unique --min-clone-size 10")
  echo "prep_small=$A prep_big=$B"
  echo "$A" > "$WORK/.prep_small.jobid"; echo "$B" > "$WORK/.prep_big.jobid"
  ;;
sweep)
  SMALL=$(cat "$WORK/.prep_small.jobid" 2>/dev/null || echo "")
  BIG=$(cat "$WORK/.prep_big.jobid" 2>/dev/null || echo "")
  ids=()
  # SMALL: the batch-size curve on real data, both devices. This is the honest version of the
  # synthetic curve -- 582 genes rather than 40, so the encoder is not a rounding error.
  for bs in 512 2048 8192 32768; do
    for dev in cuda cpu; do
      g="gpu:1"; [ "$dev" = cpu ] && g="gpu:0"
      ids+=("$(submit "small_${dev}_bs${bs}" "$SMALL" "$g" 8 96G 8:00:00 \
        "python benchmarks/cluster/profile_real.py --data $WORK/small.h5ad \
           --tag small_${dev}_bs${bs} --device $dev --batch-size $bs --epochs 200 \
           --n-samples 100 --clonotype-key trb_unique \
           --out $WORK/results/small_${dev}_bs${bs}.json")")
    done
  done
  # BIG: 2000 genes. Large batches only -- this is where a GPU should finally have real work.
  # No CPU point: 525k x 2000 on 10 cores would not finish inside the 24h limit, so the
  # denominator it would provide is not obtainable here. GPU configs compare against each
  # other; the CPU/GPU ratio comes from the SMALL dataset, which does fit.
  for bs in 8192 32768 65536; do
    ids+=("$(submit "big_cuda_bs${bs}" "$BIG" gpu:1 8 320G 24:00:00 \
      "python benchmarks/cluster/profile_real.py --data $WORK/big.h5ad \
         --tag big_cuda_bs${bs} --device cuda --batch-size $bs --epochs 200 \
         --n-samples 100 --n-latent 20 --n-hidden 128 --k 8 \
         --out $WORK/results/big_cuda_bs${bs}.json")")
  done
  printf '%s\n' "${ids[@]}" > "$WORK/.sweep.jobids"
  echo "submitted ${#ids[@]} jobs:"; printf '  %s\n' "${ids[@]}"
  ;;
*) echo "usage: bash submit.sh {prep|sweep}"; exit 1 ;;
esac
