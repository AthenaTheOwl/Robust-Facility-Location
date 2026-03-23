"""
Page 3: Robust Solution with Polyhedral Uncertainty
Box + Budget-of-Uncertainty (Bertsimas-Sim) approach.
"""

import streamlit as st
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.data import update_correlation_matrix
from core.state import (
    get_problem_data,
    load_cached_value,
    make_model_key,
    problem_key,
    store_cached_value,
)
from core.models import solve_nominal, solve_robust
from viz.network import create_network_figure, create_cost_breakdown_figure, create_capacity_chart

st.set_page_config(page_title="Robust Solution", layout="wide")
st.title("3. Robust Solution — Polyhedral Uncertainty")

# --- Get problem data ---
data = get_problem_data()

st.markdown("""
Instead of optimizing for predicted demand, the robust model protects against
**worst-case demand** within an uncertainty set. The polyhedral approach uses
two parameters to control the shape of this set.
""")

# --- Controls ---
st.sidebar.header("Uncertainty Parameters")

rho = st.sidebar.slider(
    "ρ (perturbation radius)",
    0.0, 1.0, 0.3, step=0.05,
    help="Maximum perturbation of each individual demand component. |z_k| <= ρ",
)
gamma = st.sidebar.slider(
    "Γ (budget of uncertainty)",
    0.0, float(data.m), 5.0, step=1.0,
    help="L1 budget: at most Γ demands can deviate simultaneously. ||z||₁ <= Γ",
)
R_D = st.sidebar.slider(
    "R_D (correlation radius)",
    0.05, 0.75, st.session_state.get("R_D", 0.25), step=0.05,
    help="Controls spatial correlation of demand perturbations",
)

# Update P matrix if R_D changed
data = update_correlation_matrix(data, R_D)
st.session_state["R_D"] = R_D
problem_signature = problem_key(data)
robust_model_key = make_model_key(model="robust", rho=rho, gamma=gamma)

# --- Solve both nominal and robust ---
nominal = load_cached_value("nominal_solution", problem_signature, "nominal")
robust = load_cached_value("robust_solution", problem_signature, robust_model_key)

if nominal is None or robust is None:
    with st.spinner("Solving models..."):
        if nominal is None:
            nominal = solve_nominal(data)
            store_cached_value("nominal_solution", nominal, problem_signature, "nominal")

        if robust is None:
            robust = solve_robust(data, rho=rho, gamma=gamma)
            store_cached_value("robust_solution", robust, problem_signature, robust_model_key)

if robust.status != "Optimal":
    st.error(f"Robust solver returned status: {robust.status}. Try reducing ρ or Γ.")
    st.stop()

# --- Comparison Metrics ---
st.subheader("Nominal vs Robust")
price_of_robustness = (robust.objective - nominal.objective) / nominal.objective * 100

col1, col2, col3 = st.columns(3)
col1.metric("Nominal Cost", f"${nominal.objective:.2f}")
col2.metric("Robust Cost", f"${robust.objective:.2f}",
            delta=f"+{robust.objective - nominal.objective:.2f}")
col3.metric("Price of Robustness", f"{price_of_robustness:.1f}%",
            help="How much more the robust solution costs compared to nominal")

col1, col2, col3 = st.columns(3)
col1.metric("Nominal Facilities", f"{int(np.sum(nominal.x > 0.5))}")
col2.metric("Robust Facilities", f"{int(np.sum(robust.x > 0.5))}")
col3.metric("Solve Time", f"{robust.solve_time:.3f}s")

# --- Side-by-side Network Maps ---
st.subheader("Network Comparison")
col1, col2 = st.columns(2)
with col1:
    fig_nom = create_network_figure(data, nominal, title="Nominal Solution", height=450)
    st.plotly_chart(fig_nom, use_container_width=True)
with col2:
    fig_rob = create_network_figure(data, robust,
                                     title=f"Robust (ρ={rho}, Γ={gamma})", height=450)
    st.plotly_chart(fig_rob, use_container_width=True)

# --- Cost & Capacity Comparison ---
st.subheader("Cost & Capacity Breakdown")
col1, col2 = st.columns(2)
with col1:
    st.markdown("**Nominal**")
    st.plotly_chart(create_cost_breakdown_figure(nominal, height=250), use_container_width=True)
    st.plotly_chart(create_capacity_chart(data, nominal, height=250), use_container_width=True)
with col2:
    st.markdown("**Robust**")
    st.plotly_chart(create_cost_breakdown_figure(robust, height=250), use_container_width=True)
    st.plotly_chart(create_capacity_chart(data, robust, height=250), use_container_width=True)

# --- Facility Comparison Table ---
with st.expander("Facility Comparison Table"):
    import pandas as pd
    rows = []
    for i in range(data.n):
        rows.append({
            "Facility": i,
            "Nominal Open": "Yes" if nominal.x[i] > 0.5 else "No",
            "Robust Open": "Yes" if robust.x[i] > 0.5 else "No",
            "Capacity": round(data.s[i], 2),
            "Cost": round(data.f[i], 2),
            "Nominal Flow": round(np.sum(nominal.y[i, :]), 3),
            "Robust Flow": round(np.sum(robust.y[i, :]), 3),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

# --- Educational ---
with st.expander("Learn: The Bertsimas-Sim Budget of Uncertainty"):
    st.markdown(f"""
    ### How the Uncertainty Set Works

    The demand for customer j can deviate from its predicted value d_j by an amount
    controlled by the perturbation vector z:

    **d̃_j = d_j + (P·z)_j**

    where P is the spatial correlation matrix and z lives in the uncertainty set:

    **U = {{ z : |z_k| ≤ ρ, ||z||₁ ≤ Γ }}**

    - **ρ = {rho}**: Each component z_k can deviate by at most ρ (the "box" constraint)
    - **Γ = {gamma}**: At most Γ components can deviate simultaneously (the "budget" constraint)

    ### The Key Insight (Bertsimas & Sim, 2004)

    - When **Γ = 0**: No uncertainty → nominal solution
    - When **Γ = {data.m}** (= M): All demands can be worst-case simultaneously → extremely conservative
    - When **Γ ≈ √M ≈ {np.sqrt(data.m):.1f}**: A probabilistically meaningful level of protection

    The budget Γ acts as a **dial between optimism and pessimism**. Increasing Γ opens more
    facilities with more capacity, at higher cost, but with better worst-case guarantees.

    ### The Robust Counterpart

    The inner maximization `max_{{z ∈ U}} (P·z)_j` is dualized using LP duality,
    introducing auxiliary variables that make the problem a tractable MILP
    (no harder to solve than the nominal problem).
    """)
