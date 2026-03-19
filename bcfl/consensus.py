"""
Blockchain consensus / forking model for BC-FL.

Implements Equation 10 from:
    "Energy-Efficient and Latency-Aware Blockchain-Enabled FL for Edge Networks"
"""

import numpy as np


def forking_probability(num_miners, propagation_delay=0.1, mining_rate=1.0):
    """Probability of an accidental blockchain fork.

    Parameters
    ----------
    num_miners : int
        Number of selected miners (|S_M|).
    propagation_delay : float
        Average block propagation delay between miners (seconds).
    mining_rate : float
        Block generation rate lambda (blocks/second).

    Returns
    -------
    float
        p_fork ∈ [0, 1).

    Notes
    -----
    Simplified form of Eq. 10:  p_fork = 1 - exp(-λ · S_M · δ)
    """
    t_bp = num_miners * propagation_delay
    return 1.0 - np.exp(-mining_rate * t_bp)


def forking_multiplier(num_miners, propagation_delay=0.1, mining_rate=1.0):
    """Expected number of forking events (resource penalty multiplier).

    Parameters
    ----------
    num_miners : int
        Number of selected miners (|S_M|).
    propagation_delay : float
        Average block propagation delay between miners (seconds).
    mining_rate : float
        Block generation rate lambda (blocks/second).

    Returns
    -------
    float
        A_fork = 1 / (1 - p_fork).  Always >= 1.

    Notes
    -----
    A_fork = 1 means no forking overhead.
    A_fork = 2 means the round is expected to take double time/energy.
    """
    p_fork = forking_probability(num_miners, propagation_delay, mining_rate)
    return 1.0 / (1.0 - p_fork)
