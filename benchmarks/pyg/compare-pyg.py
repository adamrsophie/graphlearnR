# PyTorch Geometric side of the comparison. Reads the graphs written by
# compare-r.R, checks that SAGEConv and GCNConv with the same weights give
# the same output as graphlearnR, and times the same layers with the same
# sizes, graphs, thread count and timing rule (median of 5 after a warm-up).
# Usage: python benchmarks/pyg/compare-pyg.py <dir> <threads>
import csv, statistics, sys, time
import numpy as np
import torch
import torch_geometric
from torch_geometric.nn import SAGEConv, GCNConv, GATConv, GINConv

out_dir, threads = sys.argv[1], int(sys.argv[2])
torch.set_num_threads(threads)
load = lambda f: np.loadtxt(f"{out_dir}/{f}", delimiter=",", ndmin=2)

def time_it(f, reps=5):
    f()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); f(); ts.append(time.perf_counter() - t0)
    return statistics.median(ts)

rows, checks = [], []
sizes = [int(s) for s in sys.argv[3].split(",")] if len(sys.argv) > 3 else [10000, 50000]
for n in sizes:
    ei = torch.tensor(load(f"edges_{n}.csv").T, dtype=torch.long)
    x = torch.tensor(load(f"x_{n}.csv"), dtype=torch.float32)
    torch.manual_seed(1)
    mlp = torch.nn.Sequential(torch.nn.Linear(32, 64), torch.nn.ReLU(), torch.nn.Linear(64, 64))
    layers = {"sage": SAGEConv(32, 64), "gcn": GCNConv(32, 64),
              "gat": GATConv(32, 16, heads=4), "gin": GINConv(mlp)}
    if n == sizes[0]:
        with torch.no_grad():
            s = layers["sage"]
            s.lin_l.weight.copy_(torch.tensor(load("sage_lin_l_weight.csv"), dtype=torch.float32))
            s.lin_l.bias.copy_(torch.tensor(load("sage_lin_l_bias.csv").ravel(), dtype=torch.float32))
            s.lin_r.weight.copy_(torch.tensor(load("sage_lin_r_weight.csv"), dtype=torch.float32))
            c = layers["gcn"]
            c.lin.weight.copy_(torch.tensor(load("gcn_lin_weight.csv"), dtype=torch.float32))
            c.bias.zero_()
            for a in ("sage", "gcn"):
                ref = torch.tensor(load(f"{a}_out_r.csv"), dtype=torch.float32)
                diff = (layers[a](x, ei) - ref).abs().max().item()
                checks.append((a, diff))
                print(f"{a}: max abs difference to graphlearnR = {diff:.3g}")
    for a, layer in layers.items():
        rows.append(("PyG", a, n, ei.shape[1],
                     time_it(lambda: layer(x, ei).sum().backward())))

with open(f"{out_dir}/timing-pyg.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["library", "layer", "nodes", "edges", "median_seconds"])
    w.writerows(rows)
with open(f"{out_dir}/equivalence.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["layer", "max_abs_difference"])
    w.writerows(checks)
with open(f"{out_dir}/versions-pyg.txt", "w") as f:
    f.write(f"torch {torch.__version__}\ntorch_geometric {torch_geometric.__version__}\nthreads {torch.get_num_threads()}\n")
for r in rows:
    print(r)
if max(d for _, d in checks) > 1e-3:
    sys.exit("graphlearnR and PyG outputs differ")
