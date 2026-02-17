#!/bin/bash

ROBOT="${1:-go2}"
STEPS=200_000_000
METHODS=("pgtt" "baseline" "wild")
LEVELS=("level03" "level07" "level10" "level13")

echo "Training robot: $ROBOT"

for ((run=0; run<1; run++)); do
    echo "=== Run $((run+1)) ==="

    for method in "${METHODS[@]}"; do
        prev_ckpt=""

        for level in "${LEVELS[@]}"; do
            name="${ROBOT}_${method}_${level}_run${run}"
            rm -rf "checks_stairs/checkpoint_${name}"

            if [[ -n "$prev_ckpt" ]]; then
                python3 -m training.train \
                    --robot "$ROBOT" \
                    --method "$method" \
                    --index "$name" \
                    --checkpoint_folder "checks_stairs/checkpoint_${prev_ckpt}" \
                    --terrain_file "terrains/${level}.npy" \
                    --num_timesteps "$STEPS"
            else
                python3 -m training.train \
                    --robot "$ROBOT" \
                    --method "$method" \
                    --index "$name" \
                    --terrain_file "terrains/${level}.npy" \
                    --num_timesteps "$STEPS"
            fi

            prev_ckpt="$name"
        done
    done
done
