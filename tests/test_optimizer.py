"""Tests for the optimizer and consensus modules."""

import numpy as np
import pytest
from bcfl import DevicePool, MinerPool, optimize, sweep
from bcfl.consensus import forking_probability, forking_multiplier


# ==============================
# Consensus tests
# ==============================

def test_forking_probability_zero_miners():
    """With 0 propagation delay, no forking."""
    p = forking_probability(5, propagation_delay=0.0)
    assert p == 0.0


def test_forking_probability_increases():
    """More miners → higher fork probability."""
    p2 = forking_probability(2)
    p10 = forking_probability(10)
    assert p10 > p2


def test_forking_multiplier_at_least_one():
    a = forking_multiplier(2)
    assert a >= 1.0


# ==============================
# Optimizer tests
# ==============================

def _make_pools():
    devices = DevicePool.random(n=20, seed=42)
    miners = MinerPool.random(m=6, seed=42)
    return devices, miners


def test_optimize_basic():
    devices, miners = _make_pools()
    result = optimize(devices, miners, s_d=10, beta=0.5)

    assert len(result.selected_devices) == 10
    assert len(result.selected_miners) >= 2
    assert result.latency > 0
    assert result.energy > 0
    assert result.fitness > 0


def test_optimize_constraint_sm_ge_2():
    """Selected miners must be at least 2."""
    devices, miners = _make_pools()
    result = optimize(devices, miners, s_d=5, beta=0.5)
    assert len(result.selected_miners) >= 2


def test_optimize_all_devices():
    """Selecting all devices should work."""
    devices, miners = _make_pools()
    result = optimize(devices, miners, s_d=20, beta=0.5)
    assert len(result.selected_devices) == 20


def test_optimize_sd_too_large():
    devices, miners = _make_pools()
    with pytest.raises(ValueError, match="exceeds"):
        optimize(devices, miners, s_d=999)


def test_optimize_too_few_miners():
    devices = DevicePool.random(n=10, seed=0)
    miners = MinerPool(proc_freq=[3e9], ver_power=[1.5])  # only 1
    with pytest.raises(ValueError, match="at least 2"):
        optimize(devices, miners, s_d=5)


def test_optimize_latency_only():
    devices, miners = _make_pools()
    result = optimize(devices, miners, s_d=10, beta=1.0)
    assert result.beta == 1.0
    assert result.fitness > 0


def test_optimize_energy_only():
    devices, miners = _make_pools()
    result = optimize(devices, miners, s_d=10, beta=0.0)
    assert result.beta == 0.0
    assert result.fitness > 0


def test_sweep():
    devices, miners = _make_pools()
    results = sweep(devices, miners, beta=0.5, s_d_range=[5, 10, 15])
    assert len(results) == 3
    assert all(r.fitness > 0 for r in results)


def test_sweep_default_range():
    devices, miners = _make_pools()
    results = sweep(devices, miners, beta=0.5)
    # Default: range(5, 21, 5) → 4 values for n=20
    assert len(results) == 4


def test_result_repr():
    devices, miners = _make_pools()
    result = optimize(devices, miners, s_d=10, beta=0.5)
    s = repr(result)
    assert "OptimizationResult" in s
    assert "latency" in s
