"""
Shared utility functions for the facility location application.
"""

import numpy as np
from .data import SolutionData


def cost_breakdown(solution: SolutionData) -> dict:
    """Return a dictionary with cost component breakdown."""
    return {
        "Total Cost": solution.objective,
        "Facility Cost": solution.facility_cost,
        "Transport Cost": solution.transport_cost,
        "Facility Cost %": 100 * solution.facility_cost / max(solution.objective, 1e-10),
        "Transport Cost %": 100 * solution.transport_cost / max(solution.objective, 1e-10),
    }


def capacity_utilization(data, solution: SolutionData) -> np.ndarray:
    """
    Compute capacity utilization per facility.

    Returns array of shape (n,) with values in [0, 1].
    """
    util = np.zeros(data.n)
    for i in range(data.n):
        if solution.x[i] > 0.5:  # facility is open
            total_flow = np.sum(solution.y[i, :])
            util[i] = total_flow / data.s[i]
    return util


def facilities_opened(solution: SolutionData) -> int:
    """Count number of open facilities."""
    return int(np.sum(solution.x > 0.5))


def spare_capacity(data, solution: SolutionData) -> float:
    """Total unused capacity across open facilities."""
    total_cap = 0.0
    total_used = 0.0
    for i in range(data.n):
        if solution.x[i] > 0.5:
            total_cap += data.s[i]
            total_used += np.sum(solution.y[i, :])
    return total_cap - total_used


def solution_summary(data, solution: SolutionData) -> dict:
    """Comprehensive summary of a solution for comparison tables."""
    return {
        "Approach": solution.name,
        "Facilities Opened": facilities_opened(solution),
        "Total Cost": round(solution.objective, 2),
        "Facility Cost": round(solution.facility_cost, 2),
        "Transport Cost": round(solution.transport_cost, 2),
        "Avg Utilization": round(np.mean(capacity_utilization(data, solution)[solution.x > 0.5]) * 100, 1) if facilities_opened(solution) > 0 else 0,
        "Spare Capacity": round(spare_capacity(data, solution), 2),
        "Solve Time (s)": round(solution.solve_time, 3),
    }
