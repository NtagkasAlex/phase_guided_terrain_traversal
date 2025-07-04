# INDEXES=(140 141 142)
# STEPS=300_000_000
# # #Remove existing checkpoints for these indexes
# rm -r "checks_stairs/checkpoint_${INDEXES[0]}"
# rm -r "checks_stairs/checkpoint_${INDEXES[1]}"
# rm -r "checks_stairs/checkpoint_${INDEXES[2]}"

# # First training run (index 0, no checkpoint)
# python3 train.py \
#     --method "pgtt" \
#     --index "${INDEXES[0]}" \
#     --terrain_file "terrains/level4.npy" \
#     --num_timesteps "$STEPS"

# #Second training run (index 1, uses checkpoint from index 0)
# python3 train.py \
#     --method "pgtt" \
#     --index "${INDEXES[1]}" \
#     --checkpoint_folder "checks_stairs/checkpoint_${INDEXES[0]}" \
#     --terrain_file "terrains/level10.npy" \
#     --num_timesteps "$STEPS"

# python3 train.py \
#     --method "pgtt" \
#     --index "${INDEXES[2]}" \
#     --checkpoint_folder "checks_stairs/checkpoint_${INDEXES[1]}" \
#     --terrain_file "terrains/level13.npy" \
#     --num_timesteps "$STEPS"


STEPS=300_000_000
INDEXES=(153 154 155)
# Remove existing checkpoints for these indexes
rm -r "checks_stairs/checkpoint_${INDEXES[0]}"
rm -r "checks_stairs/checkpoint_${INDEXES[1]}"
rm -r "checks_stairs/checkpoint_${INDEXES[2]}"

# First training run (index 0, no checkpoint)
python3 train.py \
    --method "baseline" \
    --index "${INDEXES[0]}" \
    --terrain_file "terrains/level4.npy" \
    --num_timesteps "$STEPS"

# Second training run (index 1, uses checkpoint from index 0)
python3 train.py \
    --method "baseline" \
    --index "${INDEXES[1]}" \
    --checkpoint_folder "checks_stairs/checkpoint_${INDEXES[0]}" \
    --terrain_file "terrains/level7.npy" \
    --num_timesteps "$STEPS"

python3 train.py \
    --method "baseline" \
    --index "${INDEXES[2]}" \
    --checkpoint_folder "checks_stairs/checkpoint_${INDEXES[1]}" \
    --terrain_file "terrains/level10.npy" \
    --num_timesteps "$STEPS"
