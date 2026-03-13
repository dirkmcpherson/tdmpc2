#!/bin/bash
# Submit unreliable region A/B experiments to SLURM.
# Each iteration submits a paired job (avoidance + baseline, same seed).
#
# Usage:
#   ./slurm_ondeck.sh                     # 4 paired trials on l40s
#   ./slurm_ondeck.sh -n 8                # 8 paired trials
#   ./slurm_ondeck.sh -n 4 --gpu a100     # 4 trials on a100

NUM_TRIALS=4
GPU="l40s"

while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--num_trials)
            NUM_TRIALS="$2"
            shift; shift ;;
        --gpu)
            GPU="$2"
            shift; shift ;;
        -h|--help)
            echo "Usage: ./slurm_ondeck.sh [options]"
            echo "  -n, --num_trials  Number of paired trials (default: 4)"
            echo "  --gpu             GPU type: l40s, h200, a100 (default: l40s)"
            exit 0 ;;
        *)
            echo "Unknown option: $1"; exit 1 ;;
    esac
done

for i in $(seq 1 $NUM_TRIALS); do
    echo "Submitting paired trial $i / $NUM_TRIALS on $GPU"
    python slurm_submit_job.py --name ur_paired --gpu $GPU
done
