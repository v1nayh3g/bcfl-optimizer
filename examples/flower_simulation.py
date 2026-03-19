"""
Example showing how to use BCFLStrategy with the Flower framework.

This simulates a 20-client federated learning network where clients
train a simple ML model locally, and the BCFLStrategy selects the
optimal 10 clients + verification miners per round.
"""

import os
import sys
import numpy as np

# Ensure the local bcfl package can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import flwr as fl
    from bcfl import DevicePool, MinerPool
    from bcfl.flower import BCFLStrategy
except ImportError:
    print("Please install flwr to run this example: pip install flwr")
    sys.exit(1)

# -----------------------------------------------------------------------------
# 1. Setup Network Profiles
# -----------------------------------------------------------------------------
N = 20  # Total devices
M = 5   # Total miners

devices = DevicePool.random(n=N, seed=42)
miners = MinerPool.random(m=M, seed=42)

# -----------------------------------------------------------------------------
# 2. Define Flower Client
# -----------------------------------------------------------------------------
class DummyClient(fl.client.NumPyClient):
    def __init__(self, cid: str):
        self.cid = cid
        # A tiny dummy parameter set
        self.parameters = [np.zeros(10)]
    
    def get_parameters(self, config):
        return self.parameters

    def fit(self, parameters, config):
        self.parameters = parameters
        
        # Read the penalty broadcasted by the strategy
        a_fork = config.get("a_fork", 1.0)
        round_num = config.get("server_round", 0)
        
        # In a real setup, training happens here.
        # We just log that this client was selected.
        print(f"   [Client {self.cid}] Training in Round {round_num}. Fork penalty: {a_fork:.4f}")
        
        return self.parameters, 100, {}

    def evaluate(self, parameters, config):
        # Dummy evaluation
        return 0.5, 100, {"accuracy": 0.85}

def client_fn(context: fl.common.Context):
    cid = context.node_config["partition-id"]
    return DummyClient(str(cid)).to_client()

# -----------------------------------------------------------------------------
# 3. Setup Strategy and Run Simulation
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("Starting Flower Simulation with BCFLStrategy...")
    print(f"Network: {N} Devices, {M} Miners")
    print(f"Goal: Select optimal 10 devices per round.\n")

    strategy = BCFLStrategy(
        device_pool=devices,
        miner_pool=miners,
        s_d=10,
        beta=0.5,
    )

    ray_init_args = {
        "runtime_env": {"working_dir": os.path.dirname(os.path.abspath(__file__))}
    }

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=N,
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
        ray_init_args=ray_init_args
    )
    print("\nSimulation complete.")
