from kpu_tiling import generate_loop_config, simulate_matmul, SimulationContext
import matplotlib.pyplot as plt

def plot_heatmap(tracker, title="Tile Access Heatmap"):
    plt.figure(figsize=(10, 8))
    plt.imshow(tracker, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Access Count')
    plt.title(title)
    plt.xlabel('Tile Column Index')
    plt.ylabel('Tile Row Index')
    plt.tight_layout()
    plt.show()

# Example usage
M, K, N, T = 128, 128, 128, 32
config = generate_loop_config(M, K, N, T, strategy="result-stationary")
context = SimulationContext(M//T,K//T,N//T)
simulate_matmul(config, context)
context.tracker.report()

# Visualize heatmaps
plot_heatmap(context.A_tracker, title="Matrix A Tile Access Heatmap")
plot_heatmap(context.B_tracker, title="Matrix B Tile Access Heatmap")
plot_heatmap(context.C_tracker, title="Matrix C Tile Access Heatmap")   


# Print bandwidth usage: loop_config(M=32768, K=7168, N=16384, T=32, strategy="result-stationary")
# --- Bandwidth Report ---
# DRAM_to_L3     : 469762048.00 KB (448.00 GB)
# L3_to_L2       : 469762048.00 KB (448.00 GB)
# L2_to_L1       : 469762048.00 KB (448.00 GB)
# C_evict        : 2097152.00 KB (2.00 GB)
# reuse_hits     : 114464.00 KB (0.11 GB)

# Max error vs reference: 0.002441

# --- Bandwidth Report ---
# DRAM_to_L3     : 469762048.00 KB (448.00 GB)
# L3_to_L2       : 469762048.00 KB (448.00 GB)
# L2_to_L1       : 469762048.00 KB (448.00 GB)
# C_evict        : 2097152.00 KB (2.00 GB)
# reuse_hits     : 114464.00 KB (0.11 GB)

# this is not correct - but what is?

# loop_config(M=128, K=128, N=128, T=32, strategy="result-stationary")