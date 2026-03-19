"""
bcfl-optimizer
==============

Energy-efficient and latency-aware device selection for
Blockchain-Enabled Federated Learning in edge networks.

Based on:
    "Energy-Efficient and Latency-Aware
    Blockchain-Enabled Federated Learning for Edge Networks."

Quick Start
-----------
>>> from bcfl import DevicePool, MinerPool, optimize
>>> devices = DevicePool.random(n=50)
>>> miners = MinerPool.random(m=15)
>>> result = optimize(devices, miners, s_d=30, beta=0.5)
>>> print(result)
"""

__version__ = "0.1.1"

from .devices import DevicePool
from .miners import MinerPool
from .consensus import forking_probability, forking_multiplier
from .optimizer import optimize, sweep, OptimizationResult

__all__ = [
    "DevicePool",
    "MinerPool",
    "forking_probability",
    "forking_multiplier",
    "optimize",
    "sweep",
    "OptimizationResult",
]

# Optional Flower Integration
try:
    from .flower import BCFLStrategy
    __all__.append("BCFLStrategy")
except ImportError:
    pass
