import argparse
import numpy as np

parser = argparse.ArgumentParser(description="Compare ablation eval results")
parser.add_argument('--robot', type=str, default='go2')
parser.add_argument('--methods', type=str, default='pgtt,ablation_phase,ablation_contact')
parser.add_argument('--ckpt', type=int, default=0)
parser.add_argument('--height', type=int, default=7, help='Stair height (cm) used in the eval sweep')
parser.add_argument('--run', type=int, default=0)
args = parser.parse_args()

methods = args.methods.split(",")

results = {}
for method in methods:
    filename = f"plots/{args.robot}_{method}_dist_ckpt{args.ckpt}_h{args.height:02d}_run{args.run}.npy"
    data = np.load(filename, allow_pickle=True)
    results[method] = {k: float(v) for k, v in data[0].items()}

metric_names = list(next(iter(results.values())).keys())

print(f"=== {args.robot} — height {args.height}cm, ckpt {args.ckpt}, run {args.run} ===\n")

header = f"{'Metric':<20s}" + "".join(f"{m:>20s}" for m in methods)
print(header)
print("-" * len(header))
for metric in metric_names:
    row = f"{metric:<20s}"
    for method in methods:
        row += f"{results[method][metric]:>20.4f}"
    print(row)

print("\n=== Normalized (relative to max across methods) ===\n")
print(header)
print("-" * len(header))
for metric in metric_names:
    max_val = max(results[m][metric] for m in methods)
    row = f"{metric:<20s}"
    for method in methods:
        val = results[method][metric]
        norm = val / max_val if max_val != 0 else 0.0
        row += f"{norm:>20.4f}"
    print(row)
