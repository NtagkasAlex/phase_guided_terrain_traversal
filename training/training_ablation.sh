#!/bin/bash

ROBOT="${1:-go2}"
STEPS=200_000_000
IFS=', ' read -ra RUNS <<< "${RUNS:-0}"
METHODS=("pgtt" "ablation_phase" "ablation_contact")
LEVELS=("level03" "level07" "level10" "level13")

echo "Ablation training — robot: $ROBOT, runs: ${RUNS[*]}"

for RUN in "${RUNS[@]}"; do
    echo "=== Run $((RUN+1)) ==="

    for method in "${METHODS[@]}"; do
        prev_ckpt=""

        for level in "${LEVELS[@]}"; do
            name="${ROBOT}_${method}_${level}_run${RUN}"
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
