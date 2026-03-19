"""
Joint latency-energy optimizer for Blockchain-Enabled Federated Learning.

Implements Algorithm 1 from:
    "Energy-Efficient and Latency-Aware Blockchain-Enabled FL for Edge Networks"
"""

import numpy as np
from dataclasses import dataclass, field
from itertools import combinations

from .devices import DevicePool
from .miners import MinerPool
from .consensus import forking_multiplier


@dataclass
class OptimizationResult:
    """Result of the joint device-miner selection optimization.

    Attributes
    ----------
    selected_devices : ndarray
        Indices of the selected edge devices.
    selected_miners : ndarray
        Indices of the selected miners.
    latency : float
        Total round latency T(S_D, S_M) in seconds.
    energy : float
        Total round energy E(S_D, S_M) in joules.
    fitness : float
        Objective value F = β·T̂² + (1-β)·Ê².
    a_fork : float
        Forking multiplier for the selected miner set.
    beta : float
        Trade-off parameter used.
    """
    selected_devices: np.ndarray
    selected_miners: np.ndarray
    latency: float
    energy: float
    fitness: float
    a_fork: float
    beta: float

    def __repr__(self):
        return (
            f"OptimizationResult(\n"
            f"  devices={len(self.selected_devices)}, "
            f"miners={len(self.selected_miners)},\n"
            f"  latency={self.latency:.4f}s, "
            f"energy={self.energy:.4f}J,\n"
            f"  fitness={self.fitness:.6f}, "
            f"a_fork={self.a_fork:.4f}, beta={self.beta}\n"
            f")"
        )


def _round_cost(device_pool, miner_pool, dev_idx, miner_idx, a_fork,
                epochs=5, kappa=1e-28, model_size=2.5e6,
                bandwidth=1e6, noise_power=1e-10,
                block_gen_time=1.0, block_gen_energy=2.0):
    """Compute total round latency T and energy E for a given selection.

    Implements Equation 9 from the paper.
    """
    # Device costs (selected subset)
    t_comp, e_comp = device_pool.computation_cost(epochs, kappa)
    t_up, e_up = device_pool.upload_cost(model_size, bandwidth, noise_power)
    t_dev = t_comp + t_up  # per-device total
    e_dev = e_comp + e_up

    # Miner costs (selected subset)
    t_ver, e_ver = miner_pool.verification_cost(model_size)

    # Eq. 9: T = A_fork * (max_device_time + max_miner_ver + E[t_bg] + t_bp)
    T = a_fork * (
        np.max(t_dev[dev_idx])
        + np.max(t_ver[miner_idx])
        + block_gen_time
    )

    # Eq. 9: E = A_fork * (sum_device_energy + sum_miner_energy + e_bg)
    E = a_fork * (
        np.sum(e_dev[dev_idx])
        + np.sum(e_ver[miner_idx])
        + block_gen_energy
    )

    return T, E


def optimize(device_pool, miner_pool, s_d, beta=0.5,
             propagation_delay=0.1, mining_rate=1.0,
             epochs=5, kappa=1e-28, model_size=2.5e6,
             bandwidth=1e6, noise_power=1e-10,
             block_gen_time=1.0, block_gen_energy=2.0,
             device_miner_map=None):
    """Find the optimal set of devices and miners to minimize F(S_D, S_M).

    Implements Algorithm 1: two-phase search.
      Phase 1: Find candidate miner sets (Ξ) with minimum A_fork.
      Phase 2: For each miner set, find the best S_D devices.

    Parameters
    ----------
    device_pool : DevicePool
        Pool of available edge devices.
    miner_pool : MinerPool
        Pool of available miners.
    s_d : int
        Number of devices to select.
    beta : float
        Trade-off parameter. 0 = energy only, 1 = latency only.
    propagation_delay : float
        Average block propagation delay δ (seconds).
    mining_rate : float
        Block generation rate λ.
    epochs : int
        Local training epochs τ.
    kappa : float
        Effective switched capacitance.
    model_size : float
        Model size in bits.
    bandwidth : float
        Uplink bandwidth in Hz.
    noise_power : float
        Background noise power N0.
    block_gen_time : float
        Expected block generation time E[t_bg].
    block_gen_energy : float
        Expected block generation energy E[e_bg].
    device_miner_map : dict or None
        Maps miner index → list of device indices.
        If None, all devices are available to all miners (fully connected).

    Returns
    -------
    OptimizationResult
    """
    N_D = device_pool.n
    N_M = miner_pool.m

    if s_d > N_D:
        raise ValueError(f"s_d={s_d} exceeds available devices ({N_D})")
    if N_M < 2:
        raise ValueError(f"Need at least 2 miners, got {N_M}")

    # Default: fully connected (every device is available to every miner)
    if device_miner_map is None:
        device_miner_map = {m: list(range(N_D)) for m in range(N_M)}

    # Compute normalization constants: costs when ALL participate
    all_dev = np.arange(N_D)
    all_min = np.arange(N_M)
    a_fork_all = forking_multiplier(N_M, propagation_delay, mining_rate)
    T_all, E_all = _round_cost(
        device_pool, miner_pool, all_dev, all_min, a_fork_all,
        epochs, kappa, model_size, bandwidth, noise_power,
        block_gen_time, block_gen_energy,
    )

    # ============================================================
    # Phase 1: Find candidate miner sets Ξ with minimum A_fork
    # (Algorithm 1, lines 1-11)
    # ============================================================
    a_fork_min = np.inf
    xi = []  # candidate miner sets

    for i in range(2, N_M + 1):
        a_fork_i = forking_multiplier(i, propagation_delay, mining_rate)

        if a_fork_i > a_fork_min:
            # A_fork only increases with more miners, so we can stop
            break

        # All combinations of i miners
        for miner_set in combinations(range(N_M), i):
            miner_set = list(miner_set)

            # Check: do these miners have at least s_d devices?
            available_devices = set()
            for m in miner_set:
                available_devices.update(device_miner_map[m])
            if len(available_devices) < s_d:
                continue

            a_fork_v = forking_multiplier(len(miner_set), propagation_delay, mining_rate)

            if a_fork_v < a_fork_min:
                a_fork_min = a_fork_v
                xi = [(miner_set, list(available_devices))]
            elif a_fork_v == a_fork_min:
                xi.append((miner_set, list(available_devices)))

        # If we found valid sets at this size, don't go larger
        if xi:
            break

    if not xi:
        raise RuntimeError(
            f"No valid miner set found with at least {s_d} associated devices"
        )

    # ============================================================
    # Phase 2: For each candidate miner set, find best S_D devices
    # (Algorithm 1, lines 12-16)
    # ============================================================
    best_fitness = np.inf
    best_result = None

    # Pre-compute per-device total costs once
    t_dev_all = (
        device_pool.computation_cost(epochs, kappa)[0]
        + device_pool.upload_cost(model_size, bandwidth, noise_power)[0]
    )

    for miner_set, available_devs in xi:
        a_fork = forking_multiplier(len(miner_set), propagation_delay, mining_rate)
        available_devs = np.array(available_devs)

        if len(available_devs) == s_d:
            # Exactly s_d available — no choice to make
            dev_selection = available_devs
        else:
            # Select s_d devices with lowest total latency from available set
            # (greedy heuristic for tractability when C(|H|,s_d) is large)
            candidate_costs = t_dev_all[available_devs]
            top_indices = np.argsort(candidate_costs)[:s_d]
            dev_selection = available_devs[top_indices]

        T, E = _round_cost(
            device_pool, miner_pool, dev_selection, miner_set, a_fork,
            epochs, kappa, model_size, bandwidth, noise_power,
            block_gen_time, block_gen_energy,
        )

        # Eq. 11: F = β·T̂² + (1-β)·Ê²
        T_hat = T / T_all if T_all > 0 else T
        E_hat = E / E_all if E_all > 0 else E
        fitness = beta * T_hat ** 2 + (1 - beta) * E_hat ** 2

        if fitness < best_fitness:
            best_fitness = fitness
            best_result = OptimizationResult(
                selected_devices=dev_selection,
                selected_miners=np.array(miner_set),
                latency=T,
                energy=E,
                fitness=fitness,
                a_fork=a_fork,
                beta=beta,
            )

    return best_result


def sweep(device_pool, miner_pool, beta=0.5, s_d_range=None, **kwargs):
    """Run optimization across multiple S_D values.

    Parameters
    ----------
    device_pool : DevicePool
    miner_pool : MinerPool
    beta : float
    s_d_range : iterable of int, optional
        Device counts to evaluate. Default: range(5, N_D+1, 5).
    **kwargs
        Passed to `optimize()`.

    Returns
    -------
    list of OptimizationResult
    """
    if s_d_range is None:
        s_d_range = range(5, device_pool.n + 1, 5)

    results = []
    for s_d in s_d_range:
        result = optimize(device_pool, miner_pool, s_d, beta, **kwargs)
        results.append(result)
    return results
