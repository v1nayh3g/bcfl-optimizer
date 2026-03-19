"""
Device cost model for Blockchain-Enabled Federated Learning.

Implements Equations 1-3 from:
    "Energy-Efficient and Latency-Aware Blockchain-Enabled FL for Edge Networks"
"""

import numpy as np


class DevicePool:
    """Represents a pool of heterogeneous edge devices in a BC-FL network.

    Parameters
    ----------
    cpu_freq : array_like
        CPU frequency (Hz) for each device (f_n).
    tx_power : array_like
        Uplink transmission power (W) for each device (p_n).
    data_size : array_like
        Number of local data samples per device (d_n).
    cycles_per_byte : array_like or float, optional
        CPU cycles to process one data sample (c_n). Default: 2e5.
    channel_gain : array_like or None, optional
        Uplink channel gain (h_n). If None, sampled from U[1e-8, 1e-7].
    seed : int or None, optional
        Random seed for reproducible channel gain sampling.

    Attributes
    ----------
    n : int
        Number of devices.
    """

    def __init__(self, cpu_freq, tx_power, data_size,
                 cycles_per_byte=2e5, channel_gain=None, seed=None):
        self.cpu_freq = np.asarray(cpu_freq, dtype=np.float64)
        self.tx_power = np.asarray(tx_power, dtype=np.float64)
        self.data_size = np.asarray(data_size, dtype=np.float64)
        self.n = len(self.cpu_freq)

        if np.isscalar(cycles_per_byte):
            self.cycles_per_byte = np.full(self.n, cycles_per_byte, dtype=np.float64)
        else:
            self.cycles_per_byte = np.asarray(cycles_per_byte, dtype=np.float64)

        if channel_gain is None:
            rng = np.random.default_rng(seed)
            self.channel_gain = rng.uniform(1e-8, 1e-7, self.n)
        else:
            self.channel_gain = np.asarray(channel_gain, dtype=np.float64)

        self._validate()

    def _validate(self):
        """Check that all arrays have consistent length and valid values."""
        arrays = {
            "cpu_freq": self.cpu_freq,
            "tx_power": self.tx_power,
            "data_size": self.data_size,
            "cycles_per_byte": self.cycles_per_byte,
            "channel_gain": self.channel_gain,
        }
        for name, arr in arrays.items():
            if len(arr) != self.n:
                raise ValueError(f"{name} has length {len(arr)}, expected {self.n}")
            if np.any(arr <= 0):
                raise ValueError(f"{name} contains non-positive values")

    # ------------------------------------------------------------------
    # Eq. 1: Computation latency  t_comp = (epochs * c_n * d_n) / f_n
    # Eq. 2: Computation energy   e_comp = kappa * epochs * c_n * d_n * f_n^2
    # ------------------------------------------------------------------
    def computation_cost(self, epochs=5, kappa=1e-28):
        """Per-device computation latency (s) and energy (J).

        Parameters
        ----------
        epochs : int
            Number of local SGD epochs (τ).
        kappa : float
            Effective switched capacitance of the chipset.

        Returns
        -------
        t_comp, e_comp : ndarray, ndarray
        """
        work = epochs * self.cycles_per_byte * self.data_size
        t_comp = work / self.cpu_freq
        e_comp = kappa * work * self.cpu_freq ** 2
        return t_comp, e_comp

    # ------------------------------------------------------------------
    # Eq. 3: Upload via Shannon capacity
    #   rate = B * log2(1 + p_n * h_n / N0)
    #   t_up = model_size / rate
    #   e_up = p_n * t_up
    # ------------------------------------------------------------------
    def upload_cost(self, model_size=2.5e6, bandwidth=1e6, noise_power=1e-10):
        """Per-device upload latency (s) and energy (J).

        Parameters
        ----------
        model_size : float
            Size of the model update in bits.
        bandwidth : float
            Available uplink bandwidth in Hz.
        noise_power : float
            Background noise power (N0).

        Returns
        -------
        t_up, e_up : ndarray, ndarray
        """
        snr = (self.tx_power * self.channel_gain) / noise_power
        rate = bandwidth * np.log2(1 + snr)
        t_up = model_size / rate
        e_up = self.tx_power * t_up
        return t_up, e_up

    def total_cost(self, epochs=5, kappa=1e-28,
                   model_size=2.5e6, bandwidth=1e6, noise_power=1e-10):
        """Per-device total latency (s) and energy (J).

        Returns
        -------
        t_total, e_total : ndarray, ndarray
        """
        t_comp, e_comp = self.computation_cost(epochs, kappa)
        t_up, e_up = self.upload_cost(model_size, bandwidth, noise_power)
        return t_comp + t_up, e_comp + e_up

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------
    @classmethod
    def from_csv(cls, path, **kwargs):
        """Load device profiles from a CSV file.

        Expected columns: cpu_freq, tx_power, data_size
        Optional columns: cycles_per_byte, channel_gain
        """
        import pandas as pd
        df = pd.read_csv(path)
        required = ["cpu_freq", "tx_power", "data_size"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"CSV missing required column: '{col}'")

        optional = {}
        if "cycles_per_byte" in df.columns:
            optional["cycles_per_byte"] = df["cycles_per_byte"].values
        if "channel_gain" in df.columns:
            optional["channel_gain"] = df["channel_gain"].values

        return cls(
            cpu_freq=df["cpu_freq"].values,
            tx_power=df["tx_power"].values,
            data_size=df["data_size"].values,
            **optional, **kwargs,
        )

    @classmethod
    def from_json(cls, path, **kwargs):
        """Load device profiles from a JSON file (list of dicts)."""
        import json
        with open(path) as f:
            records = json.load(f)
        return cls(
            cpu_freq=[r["cpu_freq"] for r in records],
            tx_power=[r["tx_power"] for r in records],
            data_size=[r["data_size"] for r in records],
            cycles_per_byte=[r.get("cycles_per_byte", 2e5) for r in records],
            channel_gain=[r["channel_gain"] for r in records] if "channel_gain" in records[0] else None,
            **kwargs,
        )

    @classmethod
    def random(cls, n=50, seed=42, **kwargs):
        """Generate a random pool (useful for simulations / testing)."""
        rng = np.random.default_rng(seed)
        return cls(
            cpu_freq=rng.uniform(1e9, 2e9, n),
            tx_power=rng.uniform(0.1, 0.5, n),
            data_size=rng.integers(50, 120, n).astype(float),
            seed=seed,
            **kwargs,
        )

    def __len__(self):
        return self.n

    def __repr__(self):
        return f"DevicePool(n={self.n})"
