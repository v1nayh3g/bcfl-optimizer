"""
Miner cost model for Blockchain-Enabled Federated Learning.

Implements Equation 4 from:
    "Energy-Efficient and Latency-Aware Blockchain-Enabled FL for Edge Networks"
"""

import numpy as np


class MinerPool:
    """Represents a pool of heterogeneous miners in a BC-FL network.

    Parameters
    ----------
    proc_freq : array_like
        Processing frequency (Hz) for each miner (f_m / B_m).
    ver_power : array_like
        Power consumption during verification (W) for each miner (P_m).

    Attributes
    ----------
    m : int
        Number of miners.
    """

    def __init__(self, proc_freq, ver_power):
        self.proc_freq = np.asarray(proc_freq, dtype=np.float64)
        self.ver_power = np.asarray(ver_power, dtype=np.float64)
        self.m = len(self.proc_freq)
        self._validate()

    def _validate(self):
        if len(self.ver_power) != self.m:
            raise ValueError(
                f"ver_power has length {len(self.ver_power)}, expected {self.m}"
            )
        if np.any(self.proc_freq <= 0):
            raise ValueError("proc_freq contains non-positive values")
        if np.any(self.ver_power <= 0):
            raise ValueError("ver_power contains non-positive values")

    # ------------------------------------------------------------------
    # Eq. 4: Verification cost
    #   t_ver = model_size / proc_freq
    #   e_ver = ver_power * t_ver
    # ------------------------------------------------------------------
    def verification_cost(self, model_size=2.5e6):
        """Per-miner verification latency (s) and energy (J).

        Parameters
        ----------
        model_size : float
            Size of the model update in bits.

        Returns
        -------
        t_ver, e_ver : ndarray, ndarray
        """
        t_ver = model_size / self.proc_freq
        e_ver = self.ver_power * t_ver
        return t_ver, e_ver

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------
    @classmethod
    def from_csv(cls, path):
        """Load miner profiles from a CSV file.

        Expected columns: proc_freq, ver_power
        """
        import pandas as pd
        df = pd.read_csv(path)
        for col in ["proc_freq", "ver_power"]:
            if col not in df.columns:
                raise ValueError(f"CSV missing required column: '{col}'")
        return cls(
            proc_freq=df["proc_freq"].values,
            ver_power=df["ver_power"].values,
        )

    @classmethod
    def from_json(cls, path):
        """Load miner profiles from a JSON file (list of dicts)."""
        import json
        with open(path) as f:
            records = json.load(f)
        return cls(
            proc_freq=[r["proc_freq"] for r in records],
            ver_power=[r["ver_power"] for r in records],
        )

    @classmethod
    def random(cls, m=15, seed=42):
        """Generate a random pool (useful for simulations / testing)."""
        rng = np.random.default_rng(seed)
        return cls(
            proc_freq=rng.uniform(2e9, 3.5e9, m),
            ver_power=rng.uniform(1.0, 2.0, m),
        )

    def __len__(self):
        return self.m

    def __repr__(self):
        return f"MinerPool(m={self.m})"
