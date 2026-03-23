"""
Advanced optimization models for facility location.

Implements:
  4. Ellipsoidal robust (static robust with L2 uncertainty set)
  5. Adaptive with affine decision rules + greedy rounding
"""

import time
import numpy as np
import pulp

from .data import ProblemData, SolutionData


def solve_robust_ellipsoidal(data: ProblemData, omega: float = 1.0) -> SolutionData:
    """
    Solve robust facility location with ellipsoidal uncertainty.

    Uncertainty set: U = { z : ||z||_2 <= omega }

    For each customer j, the worst-case demand perturbation is:
        max_{||z||_2 <= omega} (P·z)_j = omega * ||P[j,:]||_2

    This is a classic result from robust optimization: the support function
    of the L2 ball is the L2 norm of the coefficient vector.

    Since ||P[j,:]||_2 is a precomputable constant, the robust counterpart
    remains a MILP (no SOCP needed for static decisions).

    Parameters
    ----------
    data : ProblemData
        Problem instance.
    omega : float
        Ellipsoid radius (controls conservatism).
    """
    n, m = data.n, data.m
    t0 = time.time()

    # Precompute worst-case demand margins
    # For each customer j: margin_j = omega * ||P[j,:]||_2
    margins = np.array([omega * np.linalg.norm(data.P[j, :]) for j in range(m)])

    prob = pulp.LpProblem("Ellipsoidal_Robust_FL", pulp.LpMinimize)

    # Decision variables
    x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(n)]
    y = [[pulp.LpVariable(f"y_{i}_{j}", lowBound=0) for j in range(m)] for i in range(n)]

    # Objective
    prob += (
        pulp.lpSum(data.f[i] * x[i] for i in range(n))
        + pulp.lpSum(data.c[i, j] * y[i][j] for i in range(n) for j in range(m))
    )

    # Robust demand constraints: flow >= nominal demand + worst-case margin
    for j in range(m):
        prob += (
            pulp.lpSum(y[i][j] for i in range(n)) >= data.d[j] + margins[j]
        ), f"robust_demand_{j}"

    # Capacity constraints
    for i in range(n):
        prob += pulp.lpSum(y[i][j] for j in range(m)) <= data.s[i] * x[i], f"capacity_{i}"

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    solve_time = time.time() - t0
    status = pulp.LpStatus[prob.status]

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
        name=f"Ellipsoidal (Ω={omega})",
    )


def solve_adaptive(
    data: ProblemData,
    rho: float = 1.0,
    gamma: float = 5.0,
) -> SolutionData:
    """
    Solve adaptive facility location with affine decision rules.

    Flow variables become affine functions of uncertainty:
        y[i,j](z) = u[i,j] + Σ_k V[i,j,k] * z[k]

    The robust counterpart with polyhedral uncertainty U = {z : |z_k| <= rho, ||z||_1 <= gamma}
    dualizes each constraint's inner optimization.

    Binary variables x are relaxed to [0,1], then a greedy rounding heuristic
    is applied (matching HW4_facilityLocation_Solution.jl lines 152-170).

    Parameters
    ----------
    data : ProblemData
        Problem instance.
    rho : float
        Box radius for uncertainty.
    gamma : float
        Budget of uncertainty (L1 bound).
    """
    n, m = data.n, data.m
    t0 = time.time()

    # Use greedy rounding heuristic
    fixed_facilities = []
    x_final = None
    result = None
    max_iters = n + 2  # At most n rounding steps

    for iteration in range(max_iters):
        result = _solve_adaptive_relaxed(data, rho, gamma, fixed_facilities)
        if result is None:
            break

        x_val = result["x"]

        # Check if all x are binary (within tolerance)
        fractional = [(i, x_val[i]) for i in range(n)
                      if i not in fixed_facilities and 0.01 < x_val[i] < 0.99]

        if not fractional:
            x_final = np.array([1.0 if x_val[i] > 0.5 else 0.0 for i in range(n)])
            break

        # Fix the most fractional (highest value) variable to 1
        best_idx = max(fractional, key=lambda t: t[1])[0]
        fixed_facilities.append(best_idx)

    if x_final is None:
        # Fallback: use the last relaxed solution, rounded
        if result is not None:
            x_final = np.array([1.0 if result["x"][i] > 0.5 else 0.0 for i in range(n)])
        else:
            solve_time = time.time() - t0
            return SolutionData(
                x=np.zeros(n), y=np.zeros((n, m)),
                objective=float("inf"), facility_cost=0, transport_cost=0,
                solve_time=solve_time, status="Infeasible",
                name=f"Adaptive (ρ={rho}, Γ={gamma})",
            )

    # Use nominal dispatch only for visualization and capacity charts.
    y_val = _compute_nominal_flows(data, x_final)
    facility_cost = float(np.dot(data.f, x_final))

    # Re-evaluate the chosen binary design in the adaptive robust model so the
    # reported objective remains comparable to the other robust formulations.
    fixed_to_one = [i for i in range(n) if x_final[i] > 0.5]
    fixed_to_zero = [i for i in range(n) if x_final[i] <= 0.5]
    evaluation = _solve_adaptive_relaxed(
        data,
        rho,
        gamma,
        fixed_to_one=fixed_to_one,
        fixed_to_zero=fixed_to_zero,
        time_limit=60,
    )

    if evaluation is not None:
        objective = float(evaluation["objective"])
        transport_cost = max(objective - facility_cost, 0.0)
        status = "Optimal (rounded)"
    else:
        # Fallback keeps the app responsive if the post-rounding evaluation stalls.
        transport_cost = float(np.sum(data.c * y_val))
        objective = facility_cost + transport_cost
        status = "Approximate (rounded)"

    solve_time = time.time() - t0

    return SolutionData(
        x=x_final, y=y_val,
        objective=objective,
        facility_cost=facility_cost,
        transport_cost=transport_cost,
        solve_time=solve_time,
        status=status,
        name=f"Adaptive (ρ={rho}, Γ={gamma})",
    )


def _solve_adaptive_relaxed(
    data: ProblemData,
    rho: float,
    gamma: float,
    fixed_to_one: list[int],
    fixed_to_zero: list[int] | None = None,
    time_limit: int = 120,
) -> dict | None:
    """
    Solve the relaxed adaptive model with affine policies.

    Variables:
        x[i] ∈ [0,1]  (relaxed binary)
        u[i,j] >= 0    (base flow)
        V[i,j,k]       (affine coefficient: flow i→j depends on perturbation k)

    For each constraint, the worst-case over U is dualized:

    1. Positivity: u[i,j] + V[i,j,:]·z >= 0  for all z in U
       → u[i,j] >= rho * Σ_k alpha[i,j,k] + gamma * beta[i,j]
         where alpha[i,j,k] >= |V[i,j,k]|  (dual of box)
         and beta[i,j] >= |V[i,j,k]|  (dual of budget) — but actually:
         beta[i,j] + alpha[i,j,k] >= V[i,j,k] and >= -V[i,j,k]

    2. Demand: Σ_i (u[i,j] + V[i,j,:]·z) >= d[j] + (P·z)[j]  for all z in U
       (same duality pattern)

    3. Capacity: Σ_j (u[i,j] + V[i,j,:]·z) <= s[i]*x[i]  for all z in U
       (worst-case is when LHS is maximized)
    """
    n, m = data.n, data.m
    fixed_to_zero = fixed_to_zero or []

    prob = pulp.LpProblem("Adaptive_Relaxed", pulp.LpMinimize)

    # Facility location (relaxed)
    x = [pulp.LpVariable(f"x_{i}", lowBound=0, upBound=1) for i in range(n)]
    # Fix selected facilities to 1
    for i in fixed_to_one:
        prob += x[i] == 1
    for i in fixed_to_zero:
        prob += x[i] == 0

    # Base flow variables
    u = [[pulp.LpVariable(f"u_{i}_{j}", lowBound=0) for j in range(m)] for i in range(n)]

    # Affine coefficients V[i,j,k]
    V = [[[pulp.LpVariable(f"V_{i}_{j}_{k}") for k in range(m)]
          for j in range(m)] for i in range(n)]

    # We need an upper bound on the objective under worst-case z.
    # The objective is: Σ f[i]*x[i] + Σ c[i,j]*(u[i,j] + V[i,j,:]·z)
    # Worst-case: max_z Σ c[i,j]*V[i,j,k]*z[k]
    # Using same duality: obj_aux = rho * Σ obj_nu[k] + gamma * obj_lam
    F = pulp.LpVariable("F")

    # Objective dual variables
    obj_nu = [pulp.LpVariable(f"obj_nu_{k}", lowBound=0) for k in range(m)]
    obj_lam = pulp.LpVariable("obj_lam", lowBound=0)
    obj_yaux = [pulp.LpVariable(f"obj_yaux_{k}") for k in range(m)]

    for k in range(m):
        # Σ_{i,j} c[i,j]*V[i,j,k] is the coefficient of z[k] in the objective
        coeff_k = pulp.lpSum(data.c[i, j] * V[i][j][k] for i in range(n) for j in range(m))
        prob += obj_nu[k] >= obj_yaux[k]
        prob += obj_nu[k] >= -obj_yaux[k]
        prob += obj_lam >= -obj_yaux[k] - coeff_k
        prob += obj_lam >= obj_yaux[k] + coeff_k

    prob += F >= (
        pulp.lpSum(data.f[i] * x[i] for i in range(n))
        + pulp.lpSum(data.c[i, j] * u[i][j] for i in range(n) for j in range(m))
        + rho * pulp.lpSum(obj_nu[k] for k in range(m))
        + gamma * obj_lam
    )

    prob += F  # minimize F

    # --- Constraint 1: Positivity of flows ---
    # u[i,j] + V[i,j,:]·z >= 0 for all z in U
    # Worst-case: u[i,j] - max_z (-V[i,j,:]·z) >= 0
    # Dualized: u[i,j] >= rho * Σ_k p_nu[i,j,k] + gamma * p_lam[i,j]
    for i in range(n):
        for j in range(m):
            p_lam = pulp.LpVariable(f"p_lam_{i}_{j}", lowBound=0)
            p_nu = [pulp.LpVariable(f"p_nu_{i}_{j}_{k}", lowBound=0) for k in range(m)]
            p_yaux = [pulp.LpVariable(f"p_yaux_{i}_{j}_{k}") for k in range(m)]

            for k in range(m):
                prob += p_nu[k] >= p_yaux[k]
                prob += p_nu[k] >= -p_yaux[k]
                prob += p_lam >= -p_yaux[k] - V[i][j][k]
                prob += p_lam >= p_yaux[k] + V[i][j][k]

            prob += u[i][j] >= rho * pulp.lpSum(p_nu[k] for k in range(m)) + gamma * p_lam

    # --- Constraint 2: Demand satisfaction ---
    # Σ_i (u[i,j] + V[i,j,:]·z) >= d[j] + (P·z)[j]  for all z in U
    # Σ_i u[i,j] + Σ_i V[i,j,:]·z >= d[j] + P[j,:]·z
    # LHS - RHS coefficient of z[k]: Σ_i V[i,j,k] - P[j,k]
    for j in range(m):
        d_lam = pulp.LpVariable(f"d_lam_{j}", lowBound=0)
        d_nu = [pulp.LpVariable(f"d_nu_{j}_{k}", lowBound=0) for k in range(m)]
        d_yaux = [pulp.LpVariable(f"d_yaux_{j}_{k}") for k in range(m)]

        for k in range(m):
            # Using same dual pattern but for: Σ_i u[i,j] >= d[j] + max_z (P[j,:]-Σ_i V[i,j,:])'z
            # The "adversarial coefficient" is (P[j,k] - Σ_i V[i,j,k])
            adv_coeff_k = data.P[j, k] - pulp.lpSum(V[i][j][k] for i in range(n))

            prob += d_nu[k] >= d_yaux[k]
            prob += d_nu[k] >= -d_yaux[k]
            prob += d_lam >= -d_yaux[k] - adv_coeff_k
            prob += d_lam >= d_yaux[k] + adv_coeff_k

        prob += (
            pulp.lpSum(u[i][j] for i in range(n))
            >= data.d[j] + rho * pulp.lpSum(d_nu[k] for k in range(m)) + gamma * d_lam
        )

    # --- Constraint 3: Capacity ---
    # Σ_j (u[i,j] + V[i,j,:]·z) <= s[i]*x[i]  for all z in U
    # Worst-case: Σ_j u[i,j] + max_z Σ_j V[i,j,:]·z <= s[i]*x[i]
    for i in range(n):
        c_lam = pulp.LpVariable(f"c_lam_{i}", lowBound=0)
        c_nu = [pulp.LpVariable(f"c_nu_{i}_{k}", lowBound=0) for k in range(m)]
        c_yaux = [pulp.LpVariable(f"c_yaux_{i}_{k}") for k in range(m)]

        for k in range(m):
            cap_coeff_k = pulp.lpSum(V[i][j][k] for j in range(m))
            prob += c_nu[k] >= c_yaux[k]
            prob += c_nu[k] >= -c_yaux[k]
            prob += c_lam >= -c_yaux[k] - cap_coeff_k
            prob += c_lam >= c_yaux[k] + cap_coeff_k

        prob += (
            pulp.lpSum(u[i][j] for j in range(m))
            + rho * pulp.lpSum(c_nu[k] for k in range(m)) + gamma * c_lam
            <= data.s[i] * x[i]
        )

    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit))

    if prob.status != 1:
        return None

    x_val = np.array([v.varValue or 0.0 for v in x])
    return {
        "x": x_val,
        "objective": float(pulp.value(F)),
    }


def _compute_nominal_flows(data: ProblemData, x_fixed: np.ndarray) -> np.ndarray:
    """Given fixed binary x, solve the nominal flow LP to get y."""
    n, m = data.n, data.m
    prob = pulp.LpProblem("NominalFlows", pulp.LpMinimize)

    y = [[pulp.LpVariable(f"y_{i}_{j}", lowBound=0) for j in range(m)] for i in range(n)]

    prob += pulp.lpSum(data.c[i, j] * y[i][j] for i in range(n) for j in range(m))

    for j in range(m):
        prob += pulp.lpSum(y[i][j] for i in range(n)) >= data.d[j]

    for i in range(n):
        if x_fixed[i] > 0.5:
            prob += pulp.lpSum(y[i][j] for j in range(m)) <= data.s[i]
        else:
            for j in range(m):
                prob += y[i][j] == 0

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    return np.array([[y[i][j].varValue or 0.0 for j in range(m)] for i in range(n)])
