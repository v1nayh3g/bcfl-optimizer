"""
Quickstart example for bcfl-optimizer.

Shows how to use the library in ~15 lines to find the optimal
set of devices and miners for a blockchain-enabled FL network.
"""

from bcfl import DevicePool, MinerPool, optimize

# 1. Create device and miner pools (random for demo; use .from_csv() for real data)
devices = DevicePool.random(n=50, seed=42)
miners = MinerPool.random(m=15, seed=42)

# 2. Run the optimizer: select 30 out of 50 devices, balanced trade-off
result = optimize(devices, miners, s_d=30, beta=0.5)

# 3. Print the result
print(result)
print(f"\nSelected device indices: {result.selected_devices}")
print(f"Selected miner indices:  {result.selected_miners}")
