"""
Page 4: Advanced Models — Ellipsoidal Robust & Adaptive
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
from core.models import solve_nominal
from core.models_advanced import solve_robust_ellipsoidal, solve_adaptive
from viz.network import create_network_figure, create_cost_breakdown_figure, create_capacity_chart
import plotly.graph_objects as go

st.set_page_config(page_title="Advanced Models", layout="wide")
st.title("4. Advanced Models")

# --- Get problem data ---
data = get_problem_data()
base_problem_signature = problem_key(data)

tab_ellipsoid, tab_adaptive = st.tabs(["Ellipsoidal Robust", "Adaptive (Affine Policies)"])

# ==============================================================================
# TAB A: Ellipsoidal Robust
# ==============================================================================
with tab_ellipsoid:
    st.subheader("Ellipsoidal Uncertainty Set")

    st.markdown("""
    Instead of the polyhedral (box + budget) uncertainty set, the ellipsoidal model uses:

    **U = { z : ||z||₂ ≤ Ω }**

    This is less conservative because it doesn't allow *all* demands to be worst-case simultaneously.
    The L2 ball restricts the total *squared* deviation, so large deviations in some demands
    force small deviations in others.
    """)

    col1, col2 = st.columns([1, 3])
    with col1:
        omega = st.slider("Ω (ellipsoid radius)", 0.0, 5.0, 1.0, step=0.1,
                           help="Controls the size of the ellipsoidal uncertainty set")
        R_D_ell = st.slider("R_D (correlation)", 0.05, 0.75, 0.25, step=0.05,
                             key="R_D_ell")

    data_ell = update_correlation_matrix(data, R_D_ell)
    ell_problem_signature = problem_key(data_ell)
    ell_model_key = make_model_key(model="ellipsoidal", omega=omega)

    ell_solution = load_cached_value("ellipsoidal_solution", ell_problem_signature, ell_model_key)
    if ell_solution is None:
        with st.spinner("Solving ellipsoidal robust model..."):
            ell_solution = solve_robust_ellipsoidal(data_ell, omega=omega)
            store_cached_value(
                "ellipsoidal_solution",
                ell_solution,
                ell_problem_signature,
                ell_model_key,
            )

    if ell_solution.status != "Optimal":
        st.error(f"Solver status: {ell_solution.status}")
    else:
        # Get nominal for comparison
        nominal = load_cached_value("nominal_solution", ell_problem_signature, "nominal")
        if nominal is None:
            nominal = solve_nominal(data_ell)
            store_cached_value("nominal_solution", nominal, ell_problem_signature, "nominal")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Cost", f"${ell_solution.objective:.2f}")
        col2.metric("Facilities Opened", f"{int(np.sum(ell_solution.x > 0.5))}")
        col3.metric("vs Nominal", f"+{(ell_solution.objective - nominal.objective)/nominal.objective*100:.1f}%")
        col4.metric("Solve Time", f"{ell_solution.solve_time:.3f}s")

        fig = create_network_figure(data_ell, ell_solution,
                                     title=f"Ellipsoidal Robust (Ω={omega})")
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(create_cost_breakdown_figure(ell_solution), use_container_width=True)
        with col2:
            st.plotly_chart(create_capacity_chart(data_ell, ell_solution), use_container_width=True)

    # Uncertainty set geometry visualization
    with st.expander("Visualize: Box vs Ellipsoid Uncertainty Sets"):
        st.markdown("""
        This 2D projection shows the difference between uncertainty set geometries.
        Pick two customer indices to visualize their joint uncertainty.
        """)
        col1, col2 = st.columns(2)
        idx1 = col1.number_input("Customer 1", 0, data.m - 1, 0)
        idx2 = col2.number_input("Customer 2", 0, data.m - 1, min(1, data.m - 1))

        rho_vis = st.slider("ρ for box comparison", 0.1, 1.0, 0.5, step=0.1, key="rho_vis")

        # Generate shapes
        theta = np.linspace(0, 2 * np.pi, 100)

        fig_geom = go.Figure()

        # Box
        box_x = [-rho_vis, rho_vis, rho_vis, -rho_vis, -rho_vis]
        box_y = [-rho_vis, -rho_vis, rho_vis, rho_vis, -rho_vis]
        fig_geom.add_trace(go.Scatter(x=box_x, y=box_y, mode="lines",
                                       name=f"Box (ρ={rho_vis})",
                                       line=dict(color="red", width=2)))

        # Ellipsoid (L2 ball)
        ell_x = omega * np.cos(theta)
        ell_y = omega * np.sin(theta)
        fig_geom.add_trace(go.Scatter(x=ell_x, y=ell_y, mode="lines",
                                       name=f"Ellipsoid (Ω={omega})",
                                       line=dict(color="blue", width=2)))

        # Budget constraint (L1 ball = diamond)
        gamma_vis = 1.0
        diamond_x = [0, gamma_vis, 0, -gamma_vis, 0]
        diamond_y = [gamma_vis, 0, -gamma_vis, 0, gamma_vis]
        fig_geom.add_trace(go.Scatter(x=diamond_x, y=diamond_y, mode="lines",
                                       name=f"L1 Budget (Γ={gamma_vis})",
                                       line=dict(color="green", width=2, dash="dash")))

        fig_geom.update_layout(
            title=f"Uncertainty Sets (projected onto customers {idx1} & {idx2})",
            xaxis_title=f"z[{idx1}]", yaxis_title=f"z[{idx2}]",
            xaxis=dict(scaleanchor="y", scaleratio=1),
            height=400,
        )
        st.plotly_chart(fig_geom, use_container_width=True)

    with st.expander("Learn: Why Ellipsoids Are Less Conservative"):
        st.markdown("""
        ### Box vs Ellipsoid

        **Box uncertainty** assumes every demand can independently take its worst-case value.
        This is like saying "a storm hits every customer simultaneously" — unrealistically conservative.

        **Ellipsoidal uncertainty** restricts the *total* deviation: if some demands are high,
        others must be lower. This models the realistic situation where uncertainty is
        correlated but not perfectly aligned.

        **Mathematical insight:** For the demand constraint, the robust counterpart with
        ellipsoidal uncertainty adds a margin of `Ω · ||P[j,:]||₂` — the L2 norm of the
        correlation vector. This is a precomputable constant, so the model remains a standard MILP!

        **Probabilistic interpretation:** If demand perturbations follow a multivariate
        normal distribution, then Ω ≈ √(χ²_{m,α}) gives (1-α)% confidence. For practical
        purposes, Ω ≈ 1 to 3 covers most realistic scenarios.
        """)

# ==============================================================================
# TAB B: Adaptive (Affine Decision Rules)
# ==============================================================================
with tab_adaptive:
    st.subheader("Adaptive Optimization with Affine Policies")

    st.markdown("""
    The key limitation of static robust optimization: facility flow decisions are fixed
    **before** uncertainty is revealed. In reality, a firm can *adjust* its operations
    after observing actual demand.

    **Adaptive optimization** models this flexibility using **affine decision rules**:

    **y[i,j](z) = u[i,j] + Σ_k V[i,j,k] · z[k]**

    The base flow u[i,j] is the planned allocation, and V[i,j,k] captures how the flow
    adjusts in response to perturbation z[k].
    """)

    st.warning(
        "The adaptive model is computationally intensive (O(n·m²) variables). "
        "For large instances, it may take 1-2 minutes to solve."
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        rho_adapt = st.slider("ρ (perturbation radius)", 0.0, 1.0, 0.3, step=0.05, key="rho_adapt")
        gamma_adapt = st.slider("Γ (budget)", 0.0, float(data.m), 5.0, step=1.0, key="gamma_adapt")

    run_adaptive = st.button("Solve Adaptive Model", type="primary")
    adaptive_model_key = make_model_key(model="adaptive", rho=rho_adapt, gamma=gamma_adapt)

    if run_adaptive:
        with st.spinner("Solving adaptive model with greedy rounding (this may take a while)..."):
            adapt_solution = solve_adaptive(data, rho=rho_adapt, gamma=gamma_adapt)
            store_cached_value(
                "adaptive_solution",
                adapt_solution,
                base_problem_signature,
                adaptive_model_key,
            )

    adapt_solution = load_cached_value("adaptive_solution", base_problem_signature, adaptive_model_key)
    if adapt_solution is not None:

        if "Infeasible" in adapt_solution.status:
            st.error(f"Adaptive model: {adapt_solution.status}. Try reducing ρ or Γ.")
        else:
            nominal = load_cached_value("nominal_solution", base_problem_signature, "nominal")
            if nominal is None:
                nominal = solve_nominal(data)
                store_cached_value("nominal_solution", nominal, base_problem_signature, "nominal")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Cost", f"${adapt_solution.objective:.2f}")
            col2.metric("Facilities Opened", f"{int(np.sum(adapt_solution.x > 0.5))}")
            col3.metric("vs Nominal", f"+{(adapt_solution.objective - nominal.objective)/nominal.objective*100:.1f}%")
            col4.metric("Solve Time", f"{adapt_solution.solve_time:.3f}s")

            fig = create_network_figure(data, adapt_solution,
                                         title=f"Adaptive (ρ={rho_adapt}, Γ={gamma_adapt})")
            st.plotly_chart(fig, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(create_cost_breakdown_figure(adapt_solution), use_container_width=True)
            with col2:
                st.plotly_chart(create_capacity_chart(data, adapt_solution), use_container_width=True)
            st.caption(
                "The objective above is the adaptive model's worst-case planning cost. "
                "The network and capacity visuals use a nominal dispatch for the selected facilities."
            )
    else:
        st.info("Run the adaptive model to evaluate the current ρ and Γ settings.")

    with st.expander("Learn: Why Adaptive Solutions Are Better"):
        st.markdown("""
        ### Static vs Adaptive Robust

        **Static robust:** "I must decide all flows *now*, before seeing actual demand."
        This forces over-provisioning because the firm can't redirect flows later.

        **Adaptive robust:** "I decide facility locations now, but can adjust flows *after*
        observing demand." The affine policy `y(z) = u + V·z` captures this adaptivity.

        **Why it helps:**
        - The adaptive solution typically opens **fewer facilities** than static robust
          at the same protection level
        - It achieves **lower cost** because it doesn't need to hedge as aggressively
        - The tradeoff: the model is computationally harder (more variables)

        **The greedy rounding heuristic:**
        Since the adaptive model with binary facility variables is intractable, we:
        1. Relax x[i] from {0,1} to [0,1]
        2. Solve the relaxed problem
        3. Fix the most "open" fractional facility to 1
        4. Re-solve and repeat until all facilities are binary

        This is the same heuristic used in the reference Julia code.
        """)
