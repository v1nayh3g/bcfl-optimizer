"""
Flower framework integration for bcfl-optimizer.

Provides a custom Strategy class that uses the joint latency-energy
optimizer to select clients for each federated learning round.
"""

import logging
from typing import Callable, Dict, List, Optional, Tuple, Union

try:
    from flwr.common import (
        EvaluateIns,
        EvaluateRes,
        FitIns,
        FitRes,
        Parameters,
        Scalar,
    )
    from flwr.server.client_manager import ClientManager
    from flwr.server.client_proxy import ClientProxy
    from flwr.server.strategy import Strategy
except ImportError:
    raise ImportError(
        "Flower is not installed. Install it with: "
        "`pip install bcfl-optimizer[flower]` or `pip install flwr`."
    )

import numpy as np
from bcfl import optimize, DevicePool, MinerPool

log = logging.getLogger(__name__)


class BCFLStrategy(Strategy):
    """Flower Strategy using joint latency-energy optimization.

    Selects the optimal subset of clients (devices) for training
    by running Algorithm 1 to balance round latency and energy usage.

    Parameters
    ----------
    device_pool : DevicePool
        Hardware profiles of all available clients in the network.
    miner_pool : MinerPool
        Hardware profiles of the available miners / verifier nodes.
    s_d : int
        Number of devices to select per round.
    beta : float
        Trade-off parameter (0 = energy focused, 1 = latency focused, 0.5 = balanced).
    fit_metrics_aggregation_fn : callable, optional
        Function to aggregate metrics returned by clients during `fit`.
    evaluate_metrics_aggregation_fn : callable, optional
        Function to aggregate metrics returned by clients during `evaluate`.
    **optimizer_kwargs :
        Additional arguments passed directly to the `bcfl.optimize()` function.
    """

    def __init__(
        self,
        device_pool: DevicePool,
        miner_pool: MinerPool,
        s_d: int,
        beta: float = 0.5,
        fit_metrics_aggregation_fn: Optional[Callable] = None,
        evaluate_metrics_aggregation_fn: Optional[Callable] = None,
        **optimizer_kwargs,
    ) -> None:
        super().__init__()
        self.device_pool = device_pool
        self.miner_pool = miner_pool
        self.s_d = s_d
        self.beta = beta
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.optimizer_kwargs = optimizer_kwargs

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters. Not used in this basic strategy."""
        return None

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training using BC-FL optimization."""
        # Wait until all expected clients are registered
        client_manager.wait_for(num_clients=self.device_pool.n)
        proxies = client_manager.all()

        # Run Algorithm 1 to find the optimal set of devices and miners
        log.info(f"[Round {server_round}] Running BC-FL Optimizer (SD={self.s_d}, Beta={self.beta})")
        
        result = optimize(
            device_pool=self.device_pool,
            miner_pool=self.miner_pool,
            s_d=self.s_d,
            beta=self.beta,
            **self.optimizer_kwargs
        )

        log.info(
            f"Optimizer selected {len(result.selected_devices)} devices and "
            f"{len(result.selected_miners)} miners.\n"
            f"Expected Latency: {result.latency:.4f}s | Energy: {result.energy:.4f}J"
        )

        # Pass the calculated penalty multiplier to the clients
        config = {"a_fork": float(result.a_fork), "server_round": server_round}
        fit_ins = FitIns(parameters, config)

        # Select the proxy clients corresponding to the selected device indices
        instructions = []
        for idx in result.selected_devices:
            cid = str(idx)
            if cid in proxies:
                instructions.append((proxies[cid], fit_ins))
            else:
                log.warning(f"Selected client ID {cid} is not connected.")

        return instructions

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate training results using basic federated averaging."""
        if not results:
            return None, {}

        # Basic weighted averaging of parameters
        weights_results = [
            (self._parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        
        aggregated_ndarrays = self._aggregate(weights_results)
        parameters = self._ndarrays_to_parameters(aggregated_ndarrays)

        # Aggregate custom metrics if function provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        return parameters, metrics_aggregated

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure evaluation: ask all clients to evaluate by default."""
        config = {}
        evaluate_ins = EvaluateIns(parameters, config)
        proxies = client_manager.all()
        return [(proxy, evaluate_ins) for proxy in proxies.values()]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses."""
        if not results:
            return None, {}

        # Calculate weighted average loss
        total_examples = sum(res.num_examples for _, res in results)
        weighted_loss = sum(res.num_examples * res.loss for _, res in results)
        agg_loss = weighted_loss / total_examples

        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)

        return agg_loss, metrics_aggregated

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Centralized evaluation (not used in this strategy)."""
        return None

    # --- Helpers for Parameter Conversion ---
    def _parameters_to_ndarrays(self, parameters: Parameters) -> List[np.ndarray]:
        from flwr.common import parameters_to_ndarrays
        return parameters_to_ndarrays(parameters)

    def _ndarrays_to_parameters(self, ndarrays: List[np.ndarray]) -> Parameters:
        from flwr.common import ndarrays_to_parameters
        return ndarrays_to_parameters(ndarrays)

    def _aggregate(self, results: List[Tuple[List[np.ndarray], int]]) -> List[np.ndarray]:
        """Compute weighted average of parameters."""
        num_examples_total = sum(num_examples for _, num_examples in results)

        weighted_weights = [
            [layer * num_examples for layer in weights] for weights, num_examples in results
        ]

        # Element-wise sum of list of lists
        weights_prime: List[np.ndarray] = [
            np.sum(layer_updates, axis=0) / num_examples_total
            for layer_updates in zip(*weighted_weights)
        ]
        return weights_prime
