"""
Problem instance generation for the facility location problem.

Generates facility candidates, customer locations, costs, capacities,
demands, and the spatial correlation matrix P.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class ProblemData:
    """All data defining a facility location problem instance."""
    n: int                    # Number of candidate facilities
    m: int                    # Number of customers
    facilities: np.ndarray    # (n, 2) facility coordinates
    customers: np.ndarray     # (m, 2) customer coordinates
    c: np.ndarray             # (n, m) transportation cost matrix
    f: np.ndarray             # (n,) facility fixed opening costs
    s: np.ndarray             # (n,) facility capacities
    d: np.ndarray             # (m,) nominal customer demands
    P: np.ndarray             # (m, m) demand correlation matrix


@dataclass
class SolutionData:
    """Result of solving an optimization model."""
    x: np.ndarray             # (n,) facility open/close decisions
    y: np.ndarray             # (n, m) flow matrix
    objective: float          # Total objective value
    facility_cost: float      # Sum of facility opening costs
    transport_cost: float     # Sum of transportation costs
    solve_time: float         # Wall-clock solve time in seconds
    status: str               # Solver status string
    name: str = ""            # Label for this solution


@dataclass
class OperationalResult:
    """Result of solving the operational subproblem (fixed facilities)."""
    y: np.ndarray             # (n, m) flow matrix
    total_cost: float         # Facility cost + transport cost
    transport_cost: float     # Transport cost only
    feasible: bool            # Whether demand was fully met
    unmet_demand: float       # Total unmet demand (0 if feasible)
    penalty_cost: float       # Cost of unmet demand penalty


def generate_instance(
    n: int = 8,
    m: int = 30,
    seed: int = 42,
    layout: str = "Uniform",
    R_D: float = 0.25,
    demand_range: tuple = (0.75, 1.25),
    cost_range: tuple = (5.0, 6.0),
    capacity_range: tuple = (15.0, 17.0),
) -> ProblemData:
    """
    Generate a facility location problem instance.

    Parameters
    ----------
    n : int
        Number of candidate facility locations.
    m : int
        Number of customer demand nodes.
    seed : int
        Random seed for reproducibility.
    layout : str
        Spatial layout: "Uniform", "Clustered", or "Hub-and-Spoke".
    R_D : float
        Radius of demand correlation for the P matrix.
    demand_range : tuple
        (min, max) for nominal customer demands.
    cost_range : tuple
        (min, max) for facility fixed opening costs.
    capacity_range : tuple
        (min, max) for facility capacities.

    Returns
    -------
    ProblemData
        Complete problem instance.
    """
    rng = np.random.RandomState(seed)

    # Generate spatial locations based on layout
    if layout == "Clustered":
        facilities, customers = _generate_clustered(n, m, rng)
    elif layout == "Hub-and-Spoke":
        facilities, customers = _generate_hub_spoke(n, m, rng)
    else:  # Uniform
        facilities = rng.rand(n, 2) * 0.6 + 0.2  # In [0.2, 0.8]
        customers = rng.rand(m, 2)                 # In [0, 1]

    # Transportation costs = Euclidean distances
    c = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            c[i, j] = np.linalg.norm(facilities[i] - customers[j])

    # Facility costs, capacities, demands
    f = rng.rand(n) * (cost_range[1] - cost_range[0]) + cost_range[0]
    s = rng.rand(n) * (capacity_range[1] - capacity_range[0]) + capacity_range[0]
    d = rng.rand(m) * (demand_range[1] - demand_range[0]) + demand_range[0]

    # Demand correlation matrix P
    P = _compute_correlation_matrix(customers, R_D)

    return ProblemData(n=n, m=m, facilities=facilities, customers=customers,
                       c=c, f=f, s=s, d=d, P=P)


def _generate_clustered(n: int, m: int, rng: np.random.RandomState):
    """Generate clustered layout with facility candidates near cluster centers."""
    n_clusters = max(3, n // 2)
    centers = rng.rand(n_clusters, 2) * 0.6 + 0.2

    # Facilities near cluster centers
    facilities = np.zeros((n, 2))
    for i in range(n):
        center = centers[i % n_clusters]
        facilities[i] = center + rng.randn(2) * 0.08
    facilities = np.clip(facilities, 0.05, 0.95)

    # Customers clustered around centers with some spread
    customers = np.zeros((m, 2))
    for j in range(m):
        center = centers[rng.randint(n_clusters)]
        customers[j] = center + rng.randn(2) * 0.12
    customers = np.clip(customers, 0.0, 1.0)

    return facilities, customers


def _generate_hub_spoke(n: int, m: int, rng: np.random.RandomState):
    """Generate hub-and-spoke: facilities at center, customers on spokes."""
    # Hub facilities near center
    facilities = np.zeros((n, 2))
    n_hub = max(2, n // 3)
    for i in range(n_hub):
        facilities[i] = np.array([0.5, 0.5]) + rng.randn(2) * 0.05
    # Spoke facilities
    for i in range(n_hub, n):
        angle = 2 * np.pi * (i - n_hub) / (n - n_hub)
        r = 0.25 + rng.rand() * 0.1
        facilities[i] = np.array([0.5 + r * np.cos(angle), 0.5 + r * np.sin(angle)])
    facilities = np.clip(facilities, 0.05, 0.95)

    # Customers along spokes
    n_spokes = max(4, n)
    customers = np.zeros((m, 2))
    for j in range(m):
        spoke = j % n_spokes
        angle = 2 * np.pi * spoke / n_spokes + rng.randn() * 0.1
        r = rng.rand() * 0.45
        customers[j] = np.array([0.5 + r * np.cos(angle), 0.5 + r * np.sin(angle)])
    customers = np.clip(customers, 0.0, 1.0)

    return facilities, customers


def _compute_correlation_matrix(customers: np.ndarray, R_D: float) -> np.ndarray:
    """
    Compute the demand correlation matrix P.

    P[i,j] = 0.2 * exp(-||cust_i - cust_j|| / R_D) with threshold cutoff.
    Matches HW4_facilityLocation_Solution.jl lines 16-17.
    """
    m = len(customers)
    P = np.zeros((m, m))
    threshold = 0.2 * np.exp(-1.0)  # 0.2 * exp(-R_D / R_D) = 0.2 * exp(-1)

    for i in range(m):
        for j in range(m):
            dist = np.linalg.norm(customers[i] - customers[j])
            val = 0.2 * np.exp(-dist / R_D)
            if val >= threshold:
                P[i, j] = val

    return P


def update_correlation_matrix(data: ProblemData, R_D: float) -> ProblemData:
    """Return a new ProblemData with an updated P matrix for the given R_D."""
    new_P = _compute_correlation_matrix(data.customers, R_D)
    return ProblemData(
        n=data.n, m=data.m, facilities=data.facilities, customers=data.customers,
        c=data.c, f=data.f, s=data.s, d=data.d, P=new_P
    )
