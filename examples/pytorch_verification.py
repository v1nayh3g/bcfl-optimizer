"""
Full PyTorch Demonstration of BCFLStrategy

This script proves the optimizer works seamlessly with a real
PyTorch neural network, handling tensor aggregation via Flower.
"""

import os
import sys
from collections import OrderedDict

# Ensure the local bcfl package can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import flwr as fl
from bcfl import DevicePool, MinerPool
from bcfl.flower import BCFLStrategy

# -----------------------------------------------------------------------------
# 1. Define Real PyTorch Model (Tiny CNN)
# -----------------------------------------------------------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.fc1 = nn.Linear(16 * 26 * 26, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = torch.flatten(x, 1)
        return self.fc2(x) if hasattr(self, 'fc2') else self.fc1(x)

# -----------------------------------------------------------------------------
# 2. Setup Real Network Profiles
# -----------------------------------------------------------------------------
N = 30  # Total devices
M = 5   # Total miners

# Generate synthetic network profiles representing the 30 edge devices
devices = DevicePool.random(n=N, seed=42)
miners = MinerPool.random(m=M, seed=42)

# -----------------------------------------------------------------------------
# 3. Define Flower Client
# -----------------------------------------------------------------------------
class PyTorchClient(fl.client.NumPyClient):
    def __init__(self, cid: str):
        self.cid = cid
        self.net = SimpleCNN()
        
        # We generate a tiny synthetic dataset to represent local device data.
        # This proves the algorithm handles actual tensor gradients and FL training.
        self.x_train = torch.randn(32, 1, 28, 28)
        self.y_train = torch.randint(0, 10, (32,))

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        
        # Read the penalty broadcasted by the strategy
        a_fork = config.get("a_fork", 1.0)
        round_num = config.get("server_round", 0)
        
        print(f"   [Client {self.cid}] Round {round_num}. Received Fork penalty: {a_fork:.4f} -> Training locally...")
        
        # Actual PyTorch training loop
        optimizer = torch.optim.Adam(self.net.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        self.net.train()
        for _ in range(1): # 1 epoch
            optimizer.zero_grad()
            loss = criterion(self.net(self.x_train), self.y_train)
            loss.backward()
            optimizer.step()
        
        return self.get_parameters(config={}), len(self.x_train), {"loss": float(loss)}

def client_fn(context: fl.common.Context):
    cid = context.node_config["partition-id"]
    return PyTorchClient(str(cid)).to_client()

# -----------------------------------------------------------------------------
# 4. Setup Strategy and Run Simulation
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("Starting PyTorch + Flower Simulation with BCFLStrategy...")
    
    # Use BCFLStrategy to smartly pick 15 out of 30 devices per round
    strategy = BCFLStrategy(
        device_pool=devices,
        miner_pool=miners,
        s_d=15,
        beta=0.5,
    )

    ray_init_args = {
        "runtime_env": {"working_dir": os.path.dirname(os.path.abspath(__file__))}
    }

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=N,
        config=fl.server.ServerConfig(num_rounds=2),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
        ray_init_args=ray_init_args
    )
    print("\nPyTorch simulation complete. Tensor aggregation succeeded!")
