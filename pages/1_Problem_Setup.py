"""
Page 1: Problem Setup
Configure the facility location problem instance and visualize the geography.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.data import generate_instance
from core.state import set_problem_data
from viz.network import create_network_figure

st.set_page_config(page_title="Problem Setup", layout="wide")
st.title("1. Problem Setup")

st.markdown("""
Configure the facility location problem instance. Adjust the parameters below to create
different scenarios, then explore the geography of candidate facilities and customer demand nodes.
""")

# --- Sidebar Controls ---
st.sidebar.header("Problem Parameters")

n = st.sidebar.slider("Number of candidate facilities (N)", 3, 15, 8)
m = st.sidebar.slider("Number of customers (M)", 10, 80, 30)
seed = st.sidebar.number_input("Random seed", value=42, step=1)
layout = st.sidebar.selectbox("Spatial layout", ["Uniform", "Clustered", "Hub-and-Spoke"])

st.sidebar.markdown("---")
st.sidebar.subheader("Cost & Demand Parameters")
demand_min, demand_max = st.sidebar.slider("Demand range", 0.1, 3.0, (0.75, 1.25), step=0.05)
cost_min, cost_max = st.sidebar.slider("Facility cost range", 1.0, 15.0, (5.0, 6.0), step=0.5)
cap_min, cap_max = st.sidebar.slider("Capacity range", 5.0, 30.0, (15.0, 17.0), step=0.5)
R_D = st.sidebar.slider("Correlation radius (R_D)", 0.05, 0.75, 0.25, step=0.05)

# --- Generate Instance ---
data = generate_instance(
    n=n, m=m, seed=seed, layout=layout, R_D=R_D,
    demand_range=(demand_min, demand_max),
    cost_range=(cost_min, cost_max),
    capacity_range=(cap_min, cap_max),
)

# Store in session state for other pages
set_problem_data(data)
st.session_state["R_D"] = R_D

# --- Network Map ---
st.subheader("Network Geography")
fig = create_network_figure(data, title="Candidate Facilities & Customer Demand Nodes")
st.plotly_chart(fig, use_container_width=True)

# --- Summary Statistics ---
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Demand", f"{np.sum(data.d):.2f}")
    st.metric("Avg Demand per Customer", f"{np.mean(data.d):.3f}")
with col2:
    st.metric("Total Available Capacity", f"{np.sum(data.s):.2f}")
    st.metric("Capacity / Demand Ratio", f"{np.sum(data.s) / np.sum(data.d):.2f}x")
with col3:
    st.metric("Avg Facility Cost", f"${np.mean(data.f):.2f}")
    st.metric("Avg Transport Distance", f"{np.mean(data.c):.3f}")

# --- Data Tables ---
with st.expander("View Raw Data"):
    tab1, tab2 = st.tabs(["Facilities", "Customers"])
    with tab1:
        import pandas as pd
        df_fac = pd.DataFrame({
            "Facility": range(n),
            "X": data.facilities[:, 0].round(3),
            "Y": data.facilities[:, 1].round(3),
            "Fixed Cost": data.f.round(2),
            "Capacity": data.s.round(2),
        })
        st.dataframe(df_fac, use_container_width=True)
    with tab2:
        df_cust = pd.DataFrame({
            "Customer": range(m),
            "X": data.customers[:, 0].round(3),
            "Y": data.customers[:, 1].round(3),
            "Demand": data.d.round(3),
        })
        st.dataframe(df_cust, use_container_width=True)

# --- Distance/Correlation Heatmaps ---
with st.expander("View Distance & Correlation Matrices"):
    col1, col2 = st.columns(2)
    with col1:
        fig_dist = px.imshow(
            data.c, labels=dict(x="Customer", y="Facility", color="Distance"),
            title="Transportation Cost (Distance) Matrix",
            color_continuous_scale="Viridis",
        )
        fig_dist.update_layout(height=400)
        st.plotly_chart(fig_dist, use_container_width=True)
    with col2:
        fig_corr = px.imshow(
            data.P, labels=dict(x="Customer", y="Customer", color="Correlation"),
            title=f"Demand Correlation Matrix P (R_D={R_D})",
            color_continuous_scale="Hot",
        )
        fig_corr.update_layout(height=400)
        st.plotly_chart(fig_corr, use_container_width=True)

# --- Educational ---
with st.expander("Learn: What is the Facility Location Problem?"):
    st.markdown("""
    ### The Facility Location Problem

    A firm must decide **where to open facilities** (warehouses, factories, hospitals, data centers)
    to serve geographically distributed customers at minimum total cost.

    **Real-world examples:**
    - Amazon deciding where to build fulfillment centers
    - A government placing emergency response stations
    - A telecom company positioning cell towers
    - An EV manufacturer choosing charging station locations

    **The tradeoff:** Opening more facilities reduces transportation costs (customers are closer)
    but increases fixed costs. The optimal solution balances these two forces.

    **The challenge:** Customer demand is **uncertain**. A solution optimized for today's
    demand forecast may perform poorly when actual demand differs. This motivates
    **robust optimization** — designing solutions that perform well under uncertainty.
    """)
