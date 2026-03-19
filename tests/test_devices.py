"""Tests for the device cost model."""

import numpy as np
import pytest
from bcfl.devices import DevicePool


def test_device_pool_creation():
    pool = DevicePool(
        cpu_freq=[1.5e9, 2.0e9],
        tx_power=[0.3, 0.4],
        data_size=[100, 80],
    )
    assert pool.n == 2


def test_device_pool_validation_negative():
    with pytest.raises(ValueError, match="non-positive"):
        DevicePool(cpu_freq=[-1e9], tx_power=[0.3], data_size=[100])


def test_device_pool_validation_length():
    with pytest.raises(ValueError, match="length"):
        DevicePool(cpu_freq=[1e9, 2e9], tx_power=[0.3], data_size=[100])


def test_computation_cost_formula():
    """Verify Eq. 1-2 against hand calculation."""
    pool = DevicePool(
        cpu_freq=[1e9],
        tx_power=[0.3],
        data_size=[100],
        cycles_per_byte=2e5,
        channel_gain=[5e-8],
    )
    t, e = pool.computation_cost(epochs=5, kappa=1e-28)

    # t_comp = (5 * 2e5 * 100) / 1e9 = 1e8 / 1e9 = 0.1
    assert np.isclose(t[0], 0.1)

    # e_comp = 1e-28 * 5 * 2e5 * 100 * (1e9)^2 = 1e-28 * 1e8 * 1e18 = 1e-2
    assert np.isclose(e[0], 0.01)


def test_upload_cost_positive():
    pool = DevicePool.random(n=10, seed=0)
    t, e = pool.upload_cost()
    assert np.all(t > 0)
    assert np.all(e > 0)


def test_total_cost_sum():
    pool = DevicePool.random(n=5, seed=42)
    t_comp, e_comp = pool.computation_cost()
    t_up, e_up = pool.upload_cost()
    t_total, e_total = pool.total_cost()

    np.testing.assert_allclose(t_total, t_comp + t_up)
    np.testing.assert_allclose(e_total, e_comp + e_up)


def test_random_factory():
    pool = DevicePool.random(n=50, seed=42)
    assert pool.n == 50
    assert np.all(pool.cpu_freq > 0)


def test_repr():
    pool = DevicePool.random(n=10)
    assert "DevicePool(n=10)" in repr(pool)
