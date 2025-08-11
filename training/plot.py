import numpy as np

# Load results
wild = np.load("plots/wild_results.npy", allow_pickle=True)
pgtt = np.load("plots/pgtt_results.npy", allow_pickle=True)

methods = {
    "wild": wild,
    "pgtt": pgtt
}

# Collect stair levels and metric names
num_levels = len(wild)
metric_names = list(wild[0].keys())

# Normalize and print
for i in range(num_levels):
    print(f"\n=== Terrain Level {i + 1:02d} ===")

    # Gather metric values for all methods for this stair level
    level_metrics = {metric: {} for metric in metric_names}
    for method, results in methods.items():
        for metric in metric_names:
            value = results[i][metric]
            level_metrics[metric][method] = value

    # Compute max per metric for normalization
    metric_maxes = {metric: max(values.values()) for metric, values in level_metrics.items()}

    # Print normalized results per method
    for method in methods:
        print(f"\n→ Method: {method}")
        for metric in metric_names:
            raw_val = level_metrics[metric][method]
            max_val = metric_maxes[metric]
            norm_val = raw_val / max_val if max_val != 0 else 0
            print(f"  {metric:<20s}: {norm_val:.3f} (raw: {raw_val:.3f})")
