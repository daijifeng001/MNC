#!/bin/bash
# Usage:
# ./experiments/scripts/cfm.sh GPU NET [--set ...]
# Example:
# ./experiments/scripts/cfm.sh 0 VGG16 \
#   --set EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400,500,600,700]"

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
NET=$2
NET_lc=${NET,,}
ITERS=30000
DATASET_TRAIN=voc_2012_seg_train
DATASET_TEST=voc_2012_seg_val
array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:2:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

LOG="experiments/logs/cfm_${NET}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

NET_INIT=data/imagenet_models/${NET}.mask.caffemodel
time ./tools/train_net.py --gpu ${GPU_ID} \
  --solver models/${NET}/cfm/solver.prototxt \
  --weights ${NET_INIT} \
  --imdb ${DATASET_TRAIN} \
  --iters ${ITERS} \
  --cfg experiments/cfgs/${NET}/cfm.yml \
  ${EXTRA_ARGS}

set +x
NET_FINAL=`grep -B 1 "done solving" ${LOG} | grep "Wrote snapshot" | awk '{print $4}'`
set -x

time ./tools/test_net.py --gpu ${GPU_ID} \
  --def models/${NET}/cfm/test.prototxt \
  --net ${NET_FINAL} \
  --imdb ${DATASET_TEST} \
  --cfg experiments/cfgs/${NET}/cfm.yml \
  --task cfm
