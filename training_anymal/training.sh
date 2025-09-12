#!/bin/bash

STEPS=300_000_000

for ((run=4; run<5; run++)); do
    OFFSET=$((70 + run * 20))

    echo "=== Run $((run+1)) | Starting Index: $OFFSET ==="

    INDEXES=($((OFFSET)) $((OFFSET+1)) $((OFFSET+2)) $((OFFSET+3)))

    PGTT method
    for i in "${INDEXES[@]}"; do
        rm -r "checks_stairs/checkpoint_$i"
    done

    python3 -m training.train \
        --method "pgtt" \
        --index "${INDEXES[0]}" \
        --terrain_file "terrains/level03.npy" \
        --num_timesteps "$STEPS"

    python3 -m training.train \
        --method "pgtt" \
        --index "${INDEXES[1]}" \
        --checkpoint_folder "checks_stairs/checkpoint_${INDEXES[0]}" \
        --terrain_file "terrains/level07.npy" \
        --num_timesteps "$STEPS"

    python3 -m training.train \
        --method "pgtt" \
        --index "${INDEXES[2]}" \
        --checkpoint_folder "checks_stairs/checkpoint_${INDEXES[1]}" \
        --terrain_file "terrains/level10.npy" \
        --num_timesteps "$STEPS"

    python3 -m training.train \
        --method "pgtt" \
        --index "${INDEXES[3]}" \
        --checkpoint_folder "checks_stairs/checkpoint_${INDEXES[2]}" \
        --terrain_file "terrains/level13.npy" \
        --num_timesteps "$STEPS"

    # Baseline method
    INDEXES=($((OFFSET+4)) $((OFFSET+5)) $((OFFSET+6)) $((OFFSET+7)))

    for i in "${INDEXES[@]}"; do
        rm -r "checks_stairs/checkpoint_$i"
    done

    python3 -m training.train \
        --method "baseline" \
        --index "${INDEXES[0]}" \
        --terrain_file "terrains/level03.npy" \
        --num_timesteps "$STEPS"

    python3 -m training.train \
        --method "baseline" \
        --index "${INDEXES[1]}" \
        --checkpoint_folder "checks_stairs/checkpoint_${INDEXES[0]}" \
        --terrain_file "terrains/level07.npy" \
        --num_timesteps "$STEPS"

    python3 -m training.train \
        --method "baseline" \
        --index "${INDEXES[2]}" \
        --checkpoint_folder "checks_stairs/checkpoint_${INDEXES[1]}" \
        --terrain_file "terrains/level10.npy" \
        --num_timesteps "$STEPS"

    python3 -m training.train \
        --method "baseline" \
        --index "${INDEXES[3]}" \
        --checkpoint_folder "checks_stairs/checkpoint_${INDEXES[2]}" \
        --terrain_file "terrains/level13.npy" \
        --num_timesteps "$STEPS"

    # # Wild method
    INDEXES=($((OFFSET+8)) $((OFFSET+9)) $((OFFSET+10)) $((OFFSET+11)))

    for i in "${INDEXES[@]}"; do
        rm -r "checks_stairs/checkpoint_$i"
    done

    python3 -m training.train \
        --method "wild" \
        --index "${INDEXES[0]}" \
        --terrain_file "terrains/level03.npy" \
        --num_timesteps "$STEPS"

    python3 -m training.train \
        --method "wild" \
        --index "${INDEXES[1]}" \
        --checkpoint_folder "checks_stairs/checkpoint_${INDEXES[0]}" \
        --terrain_file "terrains/level07.npy" \
        --num_timesteps "$STEPS"  

    python3 -m training.train \
        --method "wild" \
        --index "${INDEXES[2]}" \
        --checkpoint_folder "checks_stairs/checkpoint_${INDEXES[1]}" \
        --terrain_file "terrains/level10.npy" \
        --num_timesteps "$STEPS"

    python3 -m training.train \
        --method "wild" \
        --index "${INDEXES[3]}" \
        --checkpoint_folder "checks_stairs/checkpoint_${INDEXES[2]}" \
        --terrain_file "terrains/level13.npy" \
        --num_timesteps "$STEPS"
done

# STEPS=300_000_000
# INDEXES=(50 51 52 53)
# rm -r "checks_stairs/checkpoint_${INDEXES[0]}"
# rm -r "checks_stairs/checkpoint_${INDEXES[1]}"
# rm -r "checks_stairs/checkpoint_${INDEXES[2]}"
# rm -r "checks_stairs/checkpoint_${INDEXES[3]}"

# python3 -m training.train \
#     --method "pgtt" \
#     --index "${INDEXES[0]}" \
#     --terrain_file "terrains/level04.npy" \
#     --num_timesteps "$STEPS"

# python3 -m training.train \
#     --method "pgtt" \
#     --index "${INDEXES[1]}" \
#     --checkpoint_folder "checks_stairs/checkpoint_${INDEXES[0]}" \
#     --terrain_file "terrains/level07.npy" \
#     --num_timesteps "$STEPS"


# python3 -m training.train \
#     --method "pgtt" \
#     --index "${INDEXES[2]}" \
#     --checkpoint_folder "checks_stairs/checkpoint_${INDEXES[1]}" \
#     --terrain_file "terrains/level10.npy" \
#     --num_timesteps "$STEPS"

# python3 -m training.train \
#     --method "pgtt" \
#     --index "${INDEXES[3]}" \
#     --checkpoint_folder "checks_stairs/checkpoint_${INDEXES[2]}" \
#     --terrain_file "terrains/level13.npy" \
#     --num_timesteps "$STEPS"

# STEPS=300_000_000
# INDEXES=(54 55 56 57)
# rm -r "checks_stairs/checkpoint_${INDEXES[0]}"
# rm -r "checks_stairs/checkpoint_${INDEXES[1]}"
# rm -r "checks_stairs/checkpoint_${INDEXES[2]}"
# rm -r "checks_stairs/checkpoint_${INDEXES[3]}"

# python3 -m training.train \
#     --method "baseline" \
#     --index "${INDEXES[0]}" \
#     --terrain_file "terrains/level03.npy" \
#     --num_timesteps "$STEPS"

# python3 -m training.train \
#     --method "baseline" \
#     --index "${INDEXES[1]}" \
#     --checkpoint_folder "checks_stairs/checkpoint_${INDEXES[0]}" \
#     --terrain_file "terrains/level06.npy" \
#     --num_timesteps "$STEPS"

# python3 -m training.train \
#     --method "baseline" \
#     --index "${INDEXES[2]}" \
#     --checkpoint_folder "checks_stairs/checkpoint_${INDEXES[1]}" \
#     --terrain_file "terrains/level09.npy" \
#     --num_timesteps "$STEPS"

# python3 -m training.train \
#     --method "baseline" \
#     --index "${INDEXES[3]}" \
#     --checkpoint_folder "checks_stairs/checkpoint_${INDEXES[2]}" \
#     --terrain_file "terrains/level13.npy" \
#     --num_timesteps "$STEPS"


# STEPS=300_000_000
# INDEXES=(58 59 60 61)
# rm -r "checks_stairs/checkpoint_${INDEXES[0]}"
# rm -r "checks_stairs/checkpoint_${INDEXES[1]}"
# rm -r "checks_stairs/checkpoint_${INDEXES[2]}"
# rm -r "checks_stairs/checkpoint_${INDEXES[3]}"

# python3 -m training.train \
#     --method "wild" \
#     --index "${INDEXES[0]}" \
#     --terrain_file "terrains/level04.npy" \
#     --num_timesteps "$STEPS"


# python3 -m training.train \
#     --method "wild" \
#     --index "${INDEXES[1]}" \
#     --checkpoint_folder "checks_stairs/checkpoint_${INDEXES[0]}" \
#     --terrain_file "terrains/level07.npy" \
#     --num_timesteps "$STEPS"


# python3 -m training.train \
#     --method "wild" \
#     --index "${INDEXES[2]}" \
#     --checkpoint_folder "checks_stairs/checkpoint_${INDEXES[1]}" \
#     --terrain_file "terrains/level10.npy" \
#     --num_timesteps "$STEPS"

# python3 -m training.train \
#     --method "wild" \
#     --index "${INDEXES[3]}" \
#     --checkpoint_folder "checks_stairs/checkpoint_${INDEXES[2]}" \
#     --terrain_file "terrains/level13.npy" \
#     --num_timesteps "$STEPS"
