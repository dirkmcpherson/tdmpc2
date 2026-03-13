#!/bin/bash
# Unreliable region A/B experiment for TDMPC2 PickCube
# Runs paired seeds: oracle avoidance (background) vs baseline (foreground)
# Both conditions train in the broken env (y>0 randomizes actions)
#
# To run conditions separately:
#   ./run_unreliable_region.sh --baseline
#   ./run_unreliable_region.sh --avoidance
#   ./run_unreliable_region.sh              # (default: runs both paired)

set -euo pipefail

MODE="paired"
for arg in "$@"; do
    case $arg in
        --avoidance) MODE="avoidance" ;;
        --baseline)  MODE="baseline" ;;
        *)           echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

COMMON="task=pick-cube model_size=5 obs=state \
    wandb_project=tdmpc2_unreliable wandb_entity=jambotime \
    uncertainty_weight=false unreliable_region=true \
    unreliable_axis=1 unreliable_threshold=0.0 unreliable_side=positive \
    ee_obs_idx=18 reliability_mode=mask"

for i in {1..3}; do
    echo "**************Run $i***************"
    SEED=$(shuf -i 0-1073741824 -n 1)

    if [ "$MODE" = "avoidance" ]; then
        echo "=== Oracle avoidance only ==="
        python train.py $COMMON oracle_avoidance=true exp_name=avoidance seed=$SEED
    elif [ "$MODE" = "baseline" ]; then
        echo "=== Baseline only ==="
        python train.py $COMMON exp_name=baseline seed=$SEED
    else
        echo "=== Paired: avoidance (bg) + baseline (fg) ==="
        python train.py $COMMON oracle_avoidance=true exp_name=avoidance seed=$SEED &
        sleep 1
        python train.py $COMMON exp_name=baseline seed=$SEED
    fi
done
