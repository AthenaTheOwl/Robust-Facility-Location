"""
Monte Carlo simulation engine for stress-testing facility location solutions.
"""

from dataclasses import dataclass, field
import numpy as np

from .data import ProblemData, SolutionData, OperationalResult
from .models import solve_operational


@dataclass
class SimulationResults:
    """Results from Monte Carlo simulation across multiple approaches."""
    n_simulations: int
    approach_names: list[str]
    # Per-approach arrays (each of length n_simulations)
    costs: dict[str, np.ndarray] = field(default_factory=dict)
    transport_costs: dict[str, np.ndarray] = field(default_factory=dict)
    feasible: dict[str, np.ndarray] = field(default_factory=dict)
    unmet_demands: dict[str, np.ndarray] = field(default_factory=dict)
    penalty_costs: dict[str, np.ndarray] = field(default_factory=dict)


def run_monte_carlo(
    data: ProblemData,
    solutions: dict[str, SolutionData],
    n_simulations: int = 500,
    perturbation_scale: float = 0.3,
    distribution: str = "Uniform",
    use_correlation: bool = True,
    seed: int = 123,
    progress_callback=None,
) -> SimulationResults:
    """
    Run Monte Carlo simulation across multiple solution approaches.

    For each trial:
    1. Sample a random perturbation z
    2. Compute realized demand: d_realized = d + P@z (correlated) or d*(1+z) (independent)
    3. For each solution, fix facility decisions and solve operational LP
    4. Record costs, feasibility, unmet demand

    Parameters
    ----------
    data : ProblemData
        Problem instance.
    solutions : dict[str, SolutionData]
        Solutions to evaluate, keyed by approach name.
    n_simulations : int
        Number of demand scenarios to simulate.
    perturbation_scale : float
        Scale of demand perturbations.
    distribution : str
        "Uniform" or "Normal".
    use_correlation : bool
        If True, use P matrix for correlated perturbations.
    seed : int
        Random seed.
    progress_callback : callable, optional
        Called with (current_trial, n_simulations) for progress tracking.
    """
    rng = np.random.RandomState(seed)
    names = list(solutions.keys())

    results = SimulationResults(
        n_simulations=n_simulations,
        approach_names=names,
        costs={name: np.zeros(n_simulations) for name in names},
        transport_costs={name: np.zeros(n_simulations) for name in names},
        feasible={name: np.zeros(n_simulations, dtype=bool) for name in names},
        unmet_demands={name: np.zeros(n_simulations) for name in names},
        penalty_costs={name: np.zeros(n_simulations) for name in names},
    )

    for t in range(n_simulations):
        # Sample perturbation
        if distribution == "Uniform":
            z = rng.uniform(-perturbation_scale, perturbation_scale, size=data.m)
        else:  # Normal
            z = rng.randn(data.m) * perturbation_scale

        # Compute realized demand
        if use_correlation:
            d_realized = data.d + data.P @ z
        else:
            d_realized = data.d * (1.0 + z)

        # Clip to non-negative
        d_realized = np.maximum(d_realized, 0.0)

        # Evaluate each solution
        for name, sol in solutions.items():
            op_result = solve_operational(data, sol.x, d_realized)
            results.costs[name][t] = op_result.total_cost
            results.transport_costs[name][t] = op_result.transport_cost
            results.feasible[name][t] = op_result.feasible
            results.unmet_demands[name][t] = op_result.unmet_demand
            results.penalty_costs[name][t] = op_result.penalty_cost

        if progress_callback is not None:
            progress_callback(t + 1, n_simulations)

    return results


def compute_statistics(results: SimulationResults) -> dict[str, dict]:
    """Compute summary statistics for each approach."""
    stats = {}
    for name in results.approach_names:
        costs = results.costs[name]
        stats[name] = {
            "Mean Cost": np.mean(costs),
            "Std Dev": np.std(costs),
            "Median": np.median(costs),
            "5th Percentile": np.percentile(costs, 5),
            "95th Percentile": np.percentile(costs, 95),
            "Max Cost": np.max(costs),
            "CVaR (95%)": _cvar(costs, 0.95),
            "Infeasibility Rate (%)": 100 * np.mean(~results.feasible[name]),
            "Avg Unmet Demand": np.mean(results.unmet_demands[name]),
        }
    return stats


def _cvar(costs: np.ndarray, alpha: float = 0.95) -> float:
    """Conditional Value at Risk: average cost in the worst (1-alpha) fraction."""
    threshold = np.percentile(costs, alpha * 100)
    tail = costs[costs >= threshold]
    return float(np.mean(tail)) if len(tail) > 0 else float(threshold)
