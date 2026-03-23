"""
Robust Facility Location Explorer
==================================
An interactive Streamlit application demonstrating how robust and adaptive
optimization protects facility location decisions against demand uncertainty.
"""

import streamlit as st

st.set_page_config(
    page_title="Robust Facility Location Explorer",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Robust Facility Location Explorer")

st.markdown("""
### Why Robust Optimization?

When a firm decides where to build warehouses, factories, or service centers, it faces a
fundamental challenge: **demand is uncertain**. A solution optimized for predicted demand
may fail catastrophically when reality deviates from the forecast.

This application demonstrates four approaches to facility location under uncertainty,
each offering a different tradeoff between cost and protection:

| Approach | Philosophy | Conservatism |
|----------|-----------|--------------|
| **Nominal** | Optimize for predicted demand exactly | None — fragile to any deviation |
| **Polyhedral Robust** | Protect against worst-case in a box/budget set | High (box) to moderate (budget) |
| **Ellipsoidal Robust** | Protect against worst-case in an ellipsoid | Moderate — correlated deviations |
| **Adaptive** | Let operational decisions *react* to realized demand | Low — best of both worlds |

### How to Use This App

Navigate through the pages in the sidebar:

1. **Problem Setup** — Configure the facility network and see the geography
2. **Nominal Solution** — Solve assuming perfect demand knowledge
3. **Robust Solution** — Add polyhedral uncertainty protection
4. **Advanced Models** — Explore ellipsoidal and adaptive approaches
5. **Comparison** — See all solutions side-by-side
6. **Monte Carlo** — Stress-test with thousands of random demand scenarios
""")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "Built with [Streamlit](https://streamlit.io), "
    "[PuLP](https://coin-or.github.io/pulp/), and "
    "[CVXPY](https://www.cvxpy.org/)"
)
