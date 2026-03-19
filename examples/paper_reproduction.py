"""
Paper reproduction example for bcfl-optimizer.

Sweeps S_D from 5 to 50 and finds the optimal miner configuration
for each, reproducing the style of Fig. 3 from the paper.
"""

from bcfl import DevicePool, MinerPool, sweep

# Setup: 50 devices, 15 miners (paper's baseline)
devices = DevicePool.random(n=50, seed=42)
miners = MinerPool.random(m=15, seed=42)

# Sweep S_D from 5 to 50 in steps of 5
results = sweep(devices, miners, beta=0.5)

# Print results table
print(f"{'S_D':>5}  {'S_M':>5}  {'Latency(s)':>12}  {'Energy(J)':>12}  {'Fitness':>10}  {'A_fork':>8}")
print("-" * 65)
for r in results:
    print(
        f"{len(r.selected_devices):>5}  "
        f"{len(r.selected_miners):>5}  "
        f"{r.latency:>12.4f}  "
        f"{r.energy:>12.4f}  "
        f"{r.fitness:>10.6f}  "
        f"{r.a_fork:>8.4f}"
    )

# Optional: plot if matplotlib is available
try:
    import matplotlib.pyplot as plt

    s_d_vals = [len(r.selected_devices) for r in results]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))

    ax1.plot(s_d_vals, [r.latency for r in results], "o-", color="#2563eb")
    ax1.set(xlabel="Selected Devices (S_D)", ylabel="Latency (s)", title="Round Latency")
    ax1.grid(True, alpha=0.3)

    ax2.plot(s_d_vals, [r.energy for r in results], "s-", color="#dc2626")
    ax2.set(xlabel="Selected Devices (S_D)", ylabel="Energy (J)", title="Round Energy")
    ax2.grid(True, alpha=0.3)

    ax3.plot(s_d_vals, [r.fitness for r in results], "^-", color="#16a34a")
    ax3.set(xlabel="Selected Devices (S_D)", ylabel="Fitness Score", title="Objective F(S_D, S_M)")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("paper_reproduction.png", dpi=150)
    print("\nPlot saved to paper_reproduction.png")
    plt.show()
except ImportError:
    print("\nInstall matplotlib to generate plots: pip install matplotlib")
