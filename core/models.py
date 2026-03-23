"""
Optimization models for facility location using PuLP.

Implements:
  1. Nominal (deterministic) MILP
  2. Robust counterpart with box + budget-of-uncertainty (Bertsimas-Sim)
  3. Operational subproblem (fixed facilities, for Monte Carlo)
"""

import time
import numpy as np
import pulp

from .data import ProblemData, SolutionData, OperationalResult


def solve_nominal(data: ProblemData) -> SolutionData:
    """
    Solve the nominal (deterministic) facility location problem.

    min  Σ f[i]*x[i] + Σ c[i,j]*y[i,j]
    s.t. Σ_i y[i,j] >= d[j]          ∀j   (demand satisfaction)
         Σ_j y[i,j] <= s[i]*x[i]     ∀i   (capacity)
         x[i] ∈ {0,1}, y[i,j] >= 0
    """
    n, m = data.n, data.m
    t0 = time.time()

    prob = pulp.LpProblem("Nominal_FacilityLocation", pulp.LpMinimize)

    # Decision variables
    x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(n)]
    y = [[pulp.LpVariable(f"y_{i}_{j}", lowBound=0) for j in range(m)] for i in range(n)]

    # Objective: minimize facility + transportation costs
    prob += (
        pulp.lpSum(data.f[i] * x[i] for i in range(n))
        + pulp.lpSum(data.c[i, j] * y[i][j] for i in range(n) for j in range(m))
    )

    # Demand constraints
    for j in range(m):
        prob += pulp.lpSum(y[i][j] for i in range(n)) >= data.d[j], f"demand_{j}"

    # Capacity constraints
    for i in range(n):
        prob += pulp.lpSum(y[i][j] for j in range(m)) <= data.s[i] * x[i], f"capacity_{i}"

    # Solve
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    solve_time = time.time() - t0
    status = pulp.LpStatus[prob.status]

    # Extract solution
    x_val = np.array([v.varValue or 0.0 for v in x])
    y_val = np.array([[y[i][j].varValue or 0.0 for j in range(m)] for i in range(n)])

    facility_cost = float(np.dot(data.f, x_val))
    transport_cost = float(np.sum(data.c * y_val))

    return SolutionData(
        x=x_val, y=y_val,
        objective=facility_cost + transport_cost,
        facility_cost=facility_cost,
        transport_cost=transport_cost,
        solve_time=solve_time,
        status=status,
        name="Nominal",
    )


def solve_robust(data: ProblemData, rho: float = 1.0, gamma: float = 5.0) -> SolutionData:
    """
    Solve the robust facility location with box + budget uncertainty.

    Uncertainty set: U = { z : |z_k| <= rho, ||z||_1 <= gamma }
    Demand perturbation: d_j + max_{z in U} (P·z)_j

    The robust counterpart dualizes the inner max for each customer j:

        Σ_i u[i,j] >= d[j] + rho * Σ_k nu[j,k] + gamma * lambda[j]
        lambda[j] + nu[j,k] >= +P[j,k]    ∀k
        lambda[j] + nu[j,k] >= -P[j,k]    ∀k
        nu[j,k] >= 0, lambda[j] >= 0

    This matches HW4_facilityLocation_Solution.jl lines 60-84.
    (normdummy -> nu, infdummy -> lambda in the Julia code)
    """
    n, m = data.n, data.m
    t0 = time.time()

    prob = pulp.LpProblem("Robust_FacilityLocation", pulp.LpMinimize)

    # Decision variables
    x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(n)]
    u = [[pulp.LpVariable(f"u_{i}_{j}", lowBound=0) for j in range(m)] for i in range(n)]

    # Objective
    prob += (
        pulp.lpSum(data.f[i] * x[i] for i in range(n))
        + pulp.lpSum(data.c[i, j] * u[i][j] for i in range(n) for j in range(m))
    )

    # Robust demand constraints (one set of duals per customer j)
    for j in range(m):
        # Dual variables for this customer
        # lambda_j (dual of budget constraint ||z||_1 <= gamma)
        lam = pulp.LpVariable(f"lam_{j}", lowBound=0)
        # nu[j,k] (dual of box constraint |z_k| <= rho)
        nu = [pulp.LpVariable(f"nu_{j}_{k}", lowBound=0) for k in range(m)]
        # Auxiliary y[k] variables for absolute value linearization
        # In Julia code: infdummy >= |y[k] + P[j,k]| which encodes the dual of the
        # L1 constraint. Here we follow the exact same pattern.
        yaux = [pulp.LpVariable(f"yaux_{j}_{k}") for k in range(m)]

        # |yaux[k]| <= nu[j,k]  (encoding box dual)
        for k in range(m):
            prob += nu[k] >= yaux[k], f"nu_pos_{j}_{k}"
            prob += nu[k] >= -yaux[k], f"nu_neg_{j}_{k}"

        # lambda >= |yaux[k] + P[j,k]|  (encoding budget dual)
        for k in range(m):
            prob += lam >= -yaux[k] - data.P[j, k], f"lam_pos_{j}_{k}"
            prob += lam >= yaux[k] + data.P[j, k], f"lam_neg_{j}_{k}"

        # Main robust demand constraint
        prob += (
            pulp.lpSum(u[i][j] for i in range(n))
            >= data.d[j] + rho * pulp.lpSum(nu[k] for k in range(m)) + gamma * lam
        ), f"robust_demand_{j}"

    # Capacity constraints (same as nominal — uncertainty only affects demand side)
    for i in range(n):
        prob += pulp.lpSum(u[i][j] for j in range(m)) <= data.s[i] * x[i], f"capacity_{i}"

    # Solve
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    solve_time = time.time() - t0
    status = pulp.LpStatus[prob.status]

    # Extract solution
    x_val = np.array([v.varValue or 0.0 for v in x])
    y_val = np.array([[u[i][j].varValue or 0.0 for j in range(m)] for i in range(n)])

    facility_cost = float(np.dot(data.f, x_val))
    transport_cost = float(np.sum(data.c * y_val))

    return SolutionData(
        x=x_val, y=y_val,
        objective=facility_cost + transport_cost,
        facility_cost=facility_cost,
        transport_cost=transport_cost,
        solve_time=solve_time,
        status=status,
        name=f"Robust (ρ={rho}, Γ={gamma})",
    )


def solve_operational(
    data: ProblemData,
    x_fixed: np.ndarray,
    d_realized: np.ndarray,
    penalty: float = 1000.0,
) -> OperationalResult:
    """
    Solve the operational subproblem with fixed facility decisions.

    Given fixed facility locations x and realized demand d_realized,
    minimize transportation cost. A penalty variable for unmet demand
    ensures the problem is always feasible.

    Parameters
    ----------
    data : ProblemData
        Problem instance (uses c, f, s).
    x_fixed : np.ndarray
        Binary facility decisions from the strategic model.
    d_realized : np.ndarray
        Realized customer demands for this scenario.
    penalty : float
        Penalty per unit of unmet demand.

    Returns
    -------
    OperationalResult
    """
    n, m = data.n, data.m

    prob = pulp.LpProblem("Operational", pulp.LpMinimize)

    # Flow variables
    y = [[pulp.LpVariable(f"y_{i}_{j}", lowBound=0) for j in range(m)] for i in range(n)]

    # Unmet demand slack variables
    slack = [pulp.LpVariable(f"slack_{j}", lowBound=0) for j in range(m)]

    # Objective: transport cost + penalty for unmet demand
    prob += (
        pulp.lpSum(data.c[i, j] * y[i][j] for i in range(n) for j in range(m))
        + penalty * pulp.lpSum(slack[j] for j in range(m))
    )

    # Demand constraints (with slack)
    for j in range(m):
        prob += (
            pulp.lpSum(y[i][j] for i in range(n)) + slack[j] >= d_realized[j]
        ), f"demand_{j}"

    # Capacity constraints (only open facilities)
    for i in range(n):
        if x_fixed[i] > 0.5:
            prob += pulp.lpSum(y[i][j] for j in range(m)) <= data.s[i], f"cap_{i}"
        else:
            # Facility closed — no flow
            for j in range(m):
                prob += y[i][j] == 0, f"closed_{i}_{j}"

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    # Extract results
    y_val = np.array([[y[i][j].varValue or 0.0 for j in range(m)] for i in range(n)])
    slack_val = np.array([s.varValue or 0.0 for s in slack])
    unmet = float(np.sum(slack_val))
    transport_cost = float(np.sum(data.c * y_val))
    facility_cost = float(np.dot(data.f, x_fixed))
    penalty_cost = penalty * unmet

    return OperationalResult(
        y=y_val,
        total_cost=facility_cost + transport_cost + penalty_cost,
        transport_cost=transport_cost,
        feasible=(unmet < 1e-6),
        unmet_demand=unmet,
        penalty_cost=penalty_cost,
    )
