"""
Page 5: Side-by-Side Comparison of All Approaches
"""

import streamlit as st
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.state import (
    get_problem_data,
    load_cached_value,
    make_model_key,
    problem_key,
    store_cached_value,
)
from core.models import solve_nominal, solve_robust
from core.models_advanced import solve_robust_ellipsoidal, solve_adaptive
from core.utils import solution_summary
from viz.network import create_network_figure
from viz.comparison import (
    create_cost_comparison_figure,
    create_capacity_comparison_figure,
    create_pareto_figure,
)

st.set_page_config(page_title="Comparison", layout="wide")
st.title("5. Side-by-Side Comparison")

# --- Get problem data ---
data = get_problem_data()
problem_signature = problem_key(data)

st.markdown("""
Compare all optimization approaches side-by-side. See how each one trades off
cost for protection against uncertainty.
""")

# --- Controls ---
st.sidebar.header("Comparison Parameters")
rho = st.sidebar.slider("ρ (perturbation radius)", 0.0, 1.0, 0.3, step=0.05, key="comp_rho")
gamma = st.sidebar.slider("Γ (budget)", 0.0, float(data.m), 5.0, step=1.0, key="comp_gamma")
omega = st.sidebar.slider("Ω (ellipsoid radius)", 0.0, 5.0, 1.0, step=0.1, key="comp_omega")
include_adaptive = st.sidebar.checkbox("Include Adaptive (slower)", value=False)

# --- Solve all approaches ---
with st.spinner("Solving all models..."):
    solutions = {}

    # Nominal
    nom = solve_nominal(data)
    solutions["Nominal"] = nom

    # Robust (budget)
    rob = solve_robust(data, rho=rho, gamma=gamma)
    solutions[f"Robust (Γ={gamma})"] = rob

    # Robust (box = full budget)
    rob_box = solve_robust(data, rho=rho, gamma=float(data.m))
    solutions[f"Box Robust (Γ={data.m})"] = rob_box

    # Ellipsoidal
    ell = solve_robust_ellipsoidal(data, omega=omega)
    solutions[f"Ellipsoidal (Ω={omega})"] = ell

    # Adaptive (optional)
    if include_adaptive:
        adapt = solve_adaptive(data, rho=rho, gamma=gamma)
        solutions["Adaptive"] = adapt

    comparison_model_key = make_model_key(
        model="comparison",
        rho=rho,
        gamma=gamma,
        omega=omega,
        include_adaptive=include_adaptive,
    )
    store_cached_value("comparison_solutions", solutions, problem_signature, comparison_model_key)

# --- Summary Table ---
st.subheader("Summary")
summary_rows = [solution_summary(data, sol) for sol in solutions.values()]
df = pd.DataFrame(summary_rows)
st.dataframe(df, use_container_width=True, hide_index=True)

# --- Network Maps ---
st.subheader("Network Maps")
sol_list = list(solutions.items())
n_cols = min(len(sol_list), 3)

for row_start in range(0, len(sol_list), n_cols):
    cols = st.columns(n_cols)
    for idx, col in enumerate(cols):
        sol_idx = row_start + idx
        if sol_idx < len(sol_list):
            name, sol = sol_list[sol_idx]
            with col:
                fig = create_network_figure(data, sol, title=name, height=400)
                st.plotly_chart(fig, use_container_width=True)

# --- Cost Comparison ---
st.subheader("Cost Breakdown")
fig_cost = create_cost_comparison_figure(list(solutions.values()))
st.plotly_chart(fig_cost, use_container_width=True)
if include_adaptive:
    st.caption(
        "Adaptive cost uses the worst-case planning objective. "
        "Its plotted flows are a nominal dispatch for the chosen facilities."
    )

# --- Capacity Comparison ---
st.subheader("Capacity Utilization")
fig_cap = create_capacity_comparison_figure(data, list(solutions.values()))
st.plotly_chart(fig_cap, use_container_width=True)

# --- Pareto Frontier Sweep ---
st.markdown("---")
st.subheader("Price of Robustness — Pareto Sweep")

st.markdown("""
How does the cost change as we increase the budget of uncertainty Γ?
This sweep shows the tradeoff between cost and protection level.
""")

run_sweep = st.button("Run Pareto Sweep", type="primary")
sweep_model_key = make_model_key(model="pareto", rho=rho)
sweep_results = load_cached_value("pareto_results", problem_signature, sweep_model_key)

if run_sweep or sweep_results is not None:
    if run_sweep:
        sweep_results = []
        gamma_range = np.linspace(0, float(data.m), min(20, data.m + 1))
        progress = st.progress(0)
        for idx, g in enumerate(gamma_range):
            sol = solve_robust(data, rho=rho, gamma=g)
            sweep_results.append({
                "gamma": g,
                "objective": sol.objective,
                "facilities_opened": int(np.sum(sol.x > 0.5)),
            })
            progress.progress((idx + 1) / len(gamma_range))
        store_cached_value("pareto_results", sweep_results, problem_signature, sweep_model_key)

    fig_pareto = create_pareto_figure(sweep_results)
    st.plotly_chart(fig_pareto, use_container_width=True)

# --- Educational ---
with st.expander("Learn: The Price of Robustness"):
    st.markdown("""
    ### Understanding the Comparison

    **Nominal** is the cheapest but most fragile. It's the baseline — the cost you'd
    pay if the forecast were perfect.

    **Box Robust** (Γ = M) is the most expensive and most conservative. It protects
    against *every* customer's demand being worst-case simultaneously — an unlikely scenario.

    **Budget Robust** (Γ tunable) offers a middle ground. The Bertsimas-Sim insight:
    for moderate Γ (say √M), the probability of the real demand falling outside the
    uncertainty set decays exponentially with M. You get high-probability protection
    at a modest cost premium.

    **Ellipsoidal** offers a different geometry — it restricts total squared deviation
    rather than the L1 norm. Often results in solutions between box and budget.

    **Adaptive** is the most sophisticated: it opens facilities like the robust model
    but allows flows to *react* to actual demand. This typically achieves the best
    cost-protection tradeoff.
    """)
