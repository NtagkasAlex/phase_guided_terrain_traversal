#!/bin/bash

ROBOT="${1:-go2}"
STEPS=200_000_000
METHODS=("pgtt" "baseline" "wild")
LEVELS=("level03" "level07" "level10" "level13")

echo "Training robot: $ROBOT (starting from level1 checkpoints)"

for ((run=0; run<1; run++)); do
    echo "=== Run $((run+1)) ==="

    for method in "${METHODS[@]}"; do
        # Start from the existing level03 checkpoint
        prev_ckpt="${ROBOT}_${method}_${LEVELS[0]}_run${run}"

        for level in "${LEVELS[@]:1}"; do
            name="${ROBOT}_${method}_${level}_run${run}"
            rm -rf "checks_stairs/checkpoint_${name}"

            python3 -m training.train \
                --robot "$ROBOT" \
                --method "$method" \
                --index "$name" \
                --checkpoint_folder "checks_stairs/checkpoint_${prev_ckpt}" \
                --terrain_file "terrains/${level}.npy" \
                --num_timesteps "$STEPS"

            prev_ckpt="$name"
        done
    done
done
