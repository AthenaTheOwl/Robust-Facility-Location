"""
Page 6: Monte Carlo Simulation
Stress-test solutions with thousands of random demand scenarios.
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
from core.models_advanced import solve_robust_ellipsoidal
from core.simulation import run_monte_carlo, compute_statistics
from viz.simulation_plots import (
    create_cost_histogram,
    create_cost_cdf,
    create_cost_boxplot,
    create_infeasibility_chart,
    create_tail_risk_chart,
)

st.set_page_config(page_title="Monte Carlo Simulation", layout="wide")
st.title("6. Monte Carlo Simulation")

# --- Get problem data ---
data = get_problem_data()
problem_signature = problem_key(data)

st.markdown("""
The ultimate test: fix the facility decisions from each approach, then simulate
**thousands of random demand scenarios** and see which approach performs best.

This reveals the true value of robust optimization: a modest cost premium
buys dramatically better worst-case performance.
""")

# --- Controls ---
st.sidebar.header("Simulation Parameters")
n_sims = st.sidebar.slider("Number of simulations", 50, 2000, 300, step=50)
perturbation_scale = st.sidebar.slider("Perturbation scale", 0.05, 1.0, 0.3, step=0.05,
                                        help="Controls the magnitude of demand randomness")
distribution = st.sidebar.selectbox("Perturbation distribution", ["Uniform", "Normal"])
use_correlation = st.sidebar.checkbox("Use demand correlation (P matrix)", value=True)
sim_seed = st.sidebar.number_input("Simulation seed", value=123, step=1)

st.sidebar.markdown("---")
st.sidebar.header("Solution Parameters")
rho_mc = st.sidebar.slider("ρ", 0.0, 1.0, 0.3, step=0.05, key="mc_rho")
gamma_mc = st.sidebar.slider("Γ", 0.0, float(data.m), 5.0, step=1.0, key="mc_gamma")
omega_mc = st.sidebar.slider("Ω", 0.0, 5.0, 1.0, step=0.1, key="mc_omega")

# --- Solve approaches ---
run_sim = st.button("Run Monte Carlo Simulation", type="primary")
mc_model_key = make_model_key(
    model="monte_carlo",
    n_sims=n_sims,
    perturbation_scale=perturbation_scale,
    distribution=distribution,
    use_correlation=use_correlation,
    sim_seed=int(sim_seed),
    rho=rho_mc,
    gamma=gamma_mc,
    omega=omega_mc,
)
results = load_cached_value("mc_results", problem_signature, mc_model_key)
adaptive_solution = load_cached_value(
    "adaptive_solution",
    problem_signature,
    make_model_key(model="adaptive", rho=rho_mc, gamma=gamma_mc),
)

if run_sim:
    with st.spinner("Solving optimization models..."):
        solutions = {}
        solutions["Nominal"] = solve_nominal(data)
        solutions[f"Robust (Γ={gamma_mc})"] = solve_robust(data, rho=rho_mc, gamma=gamma_mc)
        solutions[f"Ellipsoidal (Ω={omega_mc})"] = solve_robust_ellipsoidal(data, omega=omega_mc)

        if adaptive_solution is not None:
            solutions["Adaptive"] = adaptive_solution

    st.markdown(f"**Simulating {n_sims} demand scenarios...**")
    progress_bar = st.progress(0)

    def update_progress(current, total):
        progress_bar.progress(current / total)

    results = run_monte_carlo(
        data=data,
        solutions=solutions,
        n_simulations=n_sims,
        perturbation_scale=perturbation_scale,
        distribution=distribution,
        use_correlation=use_correlation,
        seed=sim_seed,
        progress_callback=update_progress,
    )

    store_cached_value("mc_results", results, problem_signature, mc_model_key)
    progress_bar.empty()

if results is not None:

    # --- Summary Statistics ---
    st.subheader("Summary Statistics")
    st.caption(
        "Each simulation keeps facility openings fixed and re-optimizes flows after demand is realized. "
        "This isolates the strategic value of each location design."
    )
    stats = compute_statistics(results)

    # Display as metric cards
    cols = st.columns(len(results.approach_names))
    for idx, name in enumerate(results.approach_names):
        s = stats[name]
        with cols[idx]:
            st.markdown(f"**{name}**")
            st.metric("Mean Cost", f"${s['Mean Cost']:.2f}")
            st.metric("Std Dev", f"${s['Std Dev']:.2f}")
            st.metric("95th Pctile", f"${s['95th Percentile']:.2f}")
            st.metric("CVaR (95%)", f"${s['CVaR (95%)']:.2f}")
            rate = s['Infeasibility Rate (%)']
            st.metric("Infeasibility", f"{rate:.1f}%",
                       delta="SAFE" if rate == 0 else f"{rate:.1f}% fail",
                       delta_color="off" if rate == 0 else "inverse")

    # --- Full Statistics Table ---
    with st.expander("Full Statistics Table"):
        df_stats = pd.DataFrame(stats).T
        df_stats = df_stats.round(2)
        st.dataframe(df_stats, use_container_width=True)

    # --- Visualizations ---
    st.subheader("Cost Distributions")

    tab_hist, tab_cdf, tab_box = st.tabs(["Histogram", "CDF", "Box Plot"])
    with tab_hist:
        st.plotly_chart(create_cost_histogram(results), use_container_width=True)
    with tab_cdf:
        st.plotly_chart(create_cost_cdf(results), use_container_width=True)
        st.caption(
            "The CDF is the most informative plot: a curve shifted **left** means lower costs, "
            "and a curve with a **short right tail** means less risk. "
            "Look for the robust solution's tighter distribution and shorter tail."
        )
    with tab_box:
        st.plotly_chart(create_cost_boxplot(results), use_container_width=True)

    st.subheader("Risk Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(create_infeasibility_chart(results), use_container_width=True)
    with col2:
        st.plotly_chart(create_tail_risk_chart(results), use_container_width=True)

    # --- Educational ---
    with st.expander("Learn: Interpreting the Results"):
        st.markdown("""
        ### What to Look For

        **Mean Cost:** The robust solution typically has a slightly higher average cost than
        nominal. This is the "insurance premium" for robustness.

        **Standard Deviation:** The robust solution should have much lower variance — more
        predictable costs regardless of what demand does.

        **95th Percentile / CVaR:** These tail risk metrics show the cost in bad scenarios.
        The robust solution dramatically outperforms nominal here.

        **Infeasibility Rate:** What fraction of scenarios result in unmet demand? The nominal
        solution often fails in many scenarios, while the robust solution maintains feasibility.

        **The CDF plot** is the single most powerful visualization: it shows that the robust
        solution's cost distribution is tightly concentrated (steep curve) while the nominal
        solution has a long right tail (gradual slope at high costs).

        ### The Key Insight

        **Robust optimization is cheap insurance.** A modest cost premium (typically 5-20%)
        buys dramatically better worst-case performance. The nominal solution "saves" a small
        amount on average but risks catastrophic failure when demand deviates.
        """)

else:
    if adaptive_solution is None:
        st.info(
            "Click **Run Monte Carlo Simulation** to start the stress test. "
            "Adaptive is included only after you solve the matching ρ and Γ case on the Advanced Models page."
        )
    else:
        st.info("Click **Run Monte Carlo Simulation** to start the stress test.")
