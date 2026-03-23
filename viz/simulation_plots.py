"""
Visualization for Monte Carlo simulation results.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.simulation import SimulationResults, compute_statistics


# Color palette for approaches
APPROACH_COLORS = {
    "Nominal": "#1f77b4",
    "Robust": "#d62728",
    "Ellipsoidal": "#2ca02c",
    "Adaptive": "#ff7f0e",
}


def _get_color(name: str) -> str:
    """Get color for an approach, falling back to defaults."""
    for key, color in APPROACH_COLORS.items():
        if key.lower() in name.lower():
            return color
    colors = list(APPROACH_COLORS.values())
    return colors[hash(name) % len(colors)]


def create_cost_histogram(results: SimulationResults, height: int = 400) -> go.Figure:
    """Overlaid histograms of total cost across approaches."""
    fig = go.Figure()
    for name in results.approach_names:
        fig.add_trace(go.Histogram(
            x=results.costs[name],
            name=name,
            opacity=0.6,
            marker_color=_get_color(name),
            nbinsx=40,
        ))
    fig.update_layout(
        barmode="overlay",
        title=dict(text="Cost Distribution Across Scenarios", x=0.5),
        xaxis_title="Total Cost ($)",
        yaxis_title="Frequency",
        height=height,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def create_cost_cdf(results: SimulationResults, height: int = 400) -> go.Figure:
    """CDF (cumulative distribution) of costs — shows tail risk."""
    fig = go.Figure()
    for name in results.approach_names:
        sorted_costs = np.sort(results.costs[name])
        cdf = np.arange(1, len(sorted_costs) + 1) / len(sorted_costs)
        fig.add_trace(go.Scatter(
            x=sorted_costs, y=cdf,
            mode="lines",
            name=name,
            line=dict(color=_get_color(name), width=2.5),
        ))
    fig.update_layout(
        title=dict(text="Cumulative Distribution of Costs (CDF)", x=0.5),
        xaxis_title="Total Cost ($)",
        yaxis_title="Probability (cost ≤ x)",
        height=height,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def create_cost_boxplot(results: SimulationResults, height: int = 400) -> go.Figure:
    """Box plots comparing cost distributions."""
    fig = go.Figure()
    for name in results.approach_names:
        fig.add_trace(go.Box(
            y=results.costs[name],
            name=name,
            marker_color=_get_color(name),
            boxmean="sd",
        ))
    fig.update_layout(
        title=dict(text="Cost Distribution (Box Plot)", x=0.5),
        yaxis_title="Total Cost ($)",
        height=height,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def create_infeasibility_chart(results: SimulationResults, height: int = 350) -> go.Figure:
    """Bar chart showing infeasibility rates."""
    names = results.approach_names
    rates = [100 * np.mean(~results.feasible[name]) for name in names]
    colors = [_get_color(name) for name in names]

    fig = go.Figure(go.Bar(
        x=names, y=rates,
        marker_color=colors,
        text=[f"{r:.1f}%" for r in rates],
        textposition="auto",
    ))
    fig.update_layout(
        title=dict(text="Infeasibility Rate (% of scenarios with unmet demand)", x=0.5),
        yaxis_title="Infeasibility Rate (%)",
        height=height,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def create_tail_risk_chart(results: SimulationResults, height: int = 350) -> go.Figure:
    """Compare tail risk metrics (95th percentile and CVaR) across approaches."""
    stats = compute_statistics(results)
    names = results.approach_names
    colors = [_get_color(name) for name in names]

    p95 = [stats[name]["95th Percentile"] for name in names]
    cvar = [stats[name]["CVaR (95%)"] for name in names]
    max_cost = [stats[name]["Max Cost"] for name in names]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="95th Percentile", x=names, y=p95, marker_color="royalblue"))
    fig.add_trace(go.Bar(name="CVaR (95%)", x=names, y=cvar, marker_color="firebrick"))
    fig.add_trace(go.Bar(name="Max Cost", x=names, y=max_cost, marker_color="darkgray"))
    fig.update_layout(
        barmode="group",
        title=dict(text="Tail Risk Comparison", x=0.5),
        yaxis_title="Cost ($)",
        height=height,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig
