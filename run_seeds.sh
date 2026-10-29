#!/usr/bin/env bash
# Same config, several seeds, same data split; then mean ± SD.
# Usage: SEEDS="0 1 2" bash scripts/run_seeds.sh <dataset> <data_root> [extra train.py args...]
set -euo pipefail
DATASET=${1:?dataset}
DATA_ROOT=${2:?data_root}
shift 2
SEEDS=${SEEDS:-"0 1 2"}
CONFIG=${CONFIG:-configs/paper.yaml}
OUT=${OUT:-results/${DATASET}_seeds}

for s in $SEEDS; do
  python train.py --config "$CONFIG" --dataset "$DATASET" --data_root "$DATA_ROOT" \
    --seed "$s" --output_dir "$OUT" --run_name "seed${s}" "$@"
done
python scripts/aggregate_seeds.py --runs "$OUT"/seed*
