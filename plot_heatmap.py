# -----------------------
# 7) Plot heatmaps
# -----------------------
import numpy as np, matplotlib.pyplot as plt

stats = torch.load(os.path.join(OUTDIR, "hotcold_stats.pt"))
G = len(stats["grad_L1_sum"])
L, H = stats["layers"], stats["neurons"]
heat_L1 = [np.asarray(x) for x in stats["grad_L1_sum"]]
heat_hits = [np.asarray(x) for x in stats["grad_hits"]]

def plot_heat(mat, title, path):
    plt.figure(figsize=(12, 5))
    plt.imshow(mat, aspect="auto", interpolation="nearest")
    plt.colorbar()
    plt.xlabel("Neuron (row of gate_up_proj)")
    plt.ylabel("Transformer layer")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

# per-group |grad| heatmaps
for g in range(G):
    plot_heat(heat_L1[g], f"|grad| sum per neuron (group {g})", os.path.join(OUTDIR, f"heat_L1_g{g}.png"))
    plot_heat(heat_hits[g], f"hit count per neuron (group {g})", os.path.join(OUTDIR, f"heat_hits_g{g}.png"))

# selectivity map (group 0 minus group 1, normalized)
sel = (heat_L1[0] - heat_L1[1]) / (heat_L1[0] + heat_L1[1] + 1e-8)
plot_heat(sel, "Selectivity (group0 - group1) normalized", os.path.join(OUTDIR, "heat_selectivity.png"))

print("Wrote:", OUTDIR)