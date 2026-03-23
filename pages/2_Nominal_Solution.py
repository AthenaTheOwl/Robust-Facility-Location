"""
Page 2: Nominal (Deterministic) Solution
Solve assuming demand is known perfectly. Show fragility with "What If" analysis.
"""

import streamlit as st
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.state import get_problem_data, load_cached_value, problem_key, store_cached_value
from core.models import solve_nominal, solve_operational
from viz.network import create_network_figure, create_cost_breakdown_figure, create_capacity_chart

st.set_page_config(page_title="Nominal Solution", layout="wide")
st.title("2. Nominal (Deterministic) Solution")

# --- Get or generate problem data ---
data = get_problem_data()
problem_signature = problem_key(data)

st.markdown(f"""
Solving the facility location problem assuming demand is **known exactly**.
This is the classical optimization approach: find the minimum-cost solution
for the predicted demand vector.

**Problem size:** {data.n} candidate facilities, {data.m} customers
""")

# --- Solve ---
solution = load_cached_value("nominal_solution", problem_signature, "nominal")
if solution is None:
    with st.spinner("Solving nominal model..."):
        solution = solve_nominal(data)
        store_cached_value("nominal_solution", solution, problem_signature, "nominal")

if solution.status != "Optimal":
    st.error(f"Solver returned status: {solution.status}")
    st.stop()

# --- Results ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Cost", f"${solution.objective:.2f}")
col2.metric("Facilities Opened", f"{int(np.sum(solution.x > 0.5))} / {data.n}")
col3.metric("Facility Cost", f"${solution.facility_cost:.2f}")
col4.metric("Transport Cost", f"${solution.transport_cost:.2f}")

st.markdown(f"*Solved in {solution.solve_time:.3f}s*")

# --- Network Map ---
fig = create_network_figure(data, solution, title="Nominal Solution — Optimal for Predicted Demand")
st.plotly_chart(fig, use_container_width=True)

# --- Cost & Capacity ---
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(create_cost_breakdown_figure(solution), use_container_width=True)
with col2:
    st.plotly_chart(create_capacity_chart(data, solution), use_container_width=True)

# --- What If Analysis ---
st.markdown("---")
st.subheader("What If? — Testing Fragility")

st.markdown("""
The nominal solution is optimal **if** demand is exactly as predicted.
But what happens when demand deviates? Use the slider below to uniformly
perturb all demands and see whether the solution remains feasible.
""")

perturbation = st.slider(
    "Demand perturbation (%)",
    min_value=-50, max_value=100, value=0, step=5,
    help="Increase/decrease all customer demands by this percentage",
)

if perturbation != 0:
    scale = 1.0 + perturbation / 100.0
    d_perturbed = data.d * scale

    result = solve_operational(data, solution.x, d_perturbed)

    col1, col2, col3 = st.columns(3)
    col1.metric("Perturbed Total Cost", f"${result.total_cost:.2f}",
                delta=f"{result.total_cost - solution.objective:+.2f}")
    col2.metric("Feasible?", "Yes" if result.feasible else "NO",
                delta="demand met" if result.feasible else f"unmet: {result.unmet_demand:.3f}")
    col3.metric("Penalty Cost", f"${result.penalty_cost:.2f}")

    if not result.feasible:
        st.warning(
            f"The nominal solution **cannot serve** the perturbed demand! "
            f"{result.unmet_demand:.3f} units of demand go unmet. "
            f"This is why we need **robust optimization**."
        )
    else:
        st.success("The nominal solution can still serve this demand level.")
else:
    st.info("Move the slider to test how the nominal solution handles demand changes.")

# --- Educational ---
with st.expander("Learn: Why is the Nominal Solution Fragile?"):
    st.markdown("""
    ### The Fragility of Deterministic Optimization

    The nominal solution finds the **cheapest** network that serves **predicted** demand exactly.
    It has no incentive to build spare capacity or hedge against deviations.

    **What goes wrong:**
    - If demand increases even slightly, facilities may be over-capacity
    - Customers near the boundary between facility service areas are vulnerable
    - The solution "trusts" the forecast completely — no safety margin

    **Key insight:** The cost of building a *slightly* more expensive but *robust* network
    is often much less than the cost of failing to serve customers when demand spikes.

    This motivates **robust optimization**: instead of optimizing for one demand scenario,
    we optimize for the **worst case** within a set of plausible scenarios.
    """)
