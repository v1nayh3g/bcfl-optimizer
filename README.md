# bcfl-optimizer

**Energy-efficient and latency-aware device selection for Blockchain-Enabled Federated Learning.**

A Python library implementing the joint optimization algorithm from:

> D. Kushwaha, M. Kalavadia, V. Hegde and O. J. Pandey,  
> *"Energy-Efficient and Latency-Aware Blockchain-Enabled Federated Learning for Edge Networks,"*  
> IEEE TCAS-II, Vol. 71, No. 3, March 2024.

---

## Installation

```bash
pip install bcfl-optimizer
```
*(To include optional integrations like Flower, Pandas, or plotting, use `pip install bcfl-optimizer[all]`)*

**Dependencies:** `numpy` (required), `pandas` (for CSV loading), `matplotlib` (for plotting examples).

## Quick Start

```python
from bcfl import DevicePool, MinerPool, optimize

# Create pools (from arrays, CSV, JSON, or random)
devices = DevicePool.random(n=50)
miners  = MinerPool.random(m=15)

# Select optimal 30 devices + miners (balanced latency-energy trade-off)
result = optimize(devices, miners, s_d=30, beta=0.5)

print(result)
# OptimizationResult(
#   devices=30, miners=2,
#   latency=3.2145s, energy=12.4532J,
#   fitness=0.012345, a_fork=1.2214, beta=0.5
# )
```

## Loading Real Device Data

```python
# From CSV
devices = DevicePool.from_csv("my_devices.csv")
# Expected columns: cpu_freq, tx_power, data_size

# From JSON
devices = DevicePool.from_json("my_devices.json")

# From NumPy arrays
devices = DevicePool(
    cpu_freq=[1.8e9, 1.2e9, ...],
    tx_power=[0.3, 0.15, ...],
    data_size=[87, 112, ...]
)
```

## API Reference

### `DevicePool(cpu_freq, tx_power, data_size, ...)`
Heterogeneous edge device pool. Methods:
- `.computation_cost(epochs, kappa)` → latency, energy per device (Eq. 1-2)
- `.upload_cost(model_size, bandwidth, noise_power)` → latency, energy (Eq. 3)
- `.total_cost(...)` → combined per-device costs

### `MinerPool(proc_freq, ver_power)`
Miner pool. Methods:
- `.verification_cost(model_size)` → latency, energy per miner (Eq. 4)

### `optimize(device_pool, miner_pool, s_d, beta=0.5, ...)`
Run Algorithm 1 to find optimal device-miner selection.  
Returns `OptimizationResult` with selected indices, latency, energy, fitness.

### `sweep(device_pool, miner_pool, beta=0.5, s_d_range=None, ...)`
Run optimization across multiple S_D values for parameter studies.

### `forking_probability(num_miners, ...)` / `forking_multiplier(...)`
Blockchain forking model (Eq. 10).

## Flower Integration

The library natively supports the [Flower](https://flower.dev/) Federated Learning framework via a custom `Strategy` wrapping `FedAvg`.

```python
import flwr as fl
from bcfl.flower import BCFLStrategy

# Automatically selects the mathmatically optimal devices every FL round!
strategy = BCFLStrategy(
    device_pool=devices,
    miner_pool=miners,
    s_d=10,                      # target 10 devices per round
    beta=0.5,                    # balanced latency/energy
    fraction_fit=1.0,
    fraction_evaluate=1.0,
)

fl.simulation.start_simulation(client_fn=client_fn, strategy=strategy)
```

## Examples

```bash
python examples/quickstart.py           # Minimal usage
python examples/paper_reproduction.py   # Reproduce Fig. 3 trade-off curves
```

## Citation

If you use this library in your research, please cite:

```bibtex
@article{kushwaha2024energy,
  title   = {Energy-Efficient and Latency-Aware Blockchain-Enabled Federated Learning for Edge Networks},
  author  = {Kushwaha, D. and Kalavadia, M. and Hegde, V. and Pandey, O. J.},
  journal = {IEEE Transactions on Circuits and Systems II: Express Briefs},
  volume  = {71},
  number  = {3},
  pages   = {1126--1130},
  year    = {2024},
  doi     = {10.1109/TCSII.2023.3322340}
}
```

## License

MIT License. See [LICENSE](LICENSE).
