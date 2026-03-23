"""
Comparison visualizations: side-by-side charts, cost breakdowns, Pareto frontier.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.data import ProblemData, SolutionData
from core.utils import solution_summary


def create_comparison_table_data(data: ProblemData, solutions: list[SolutionData]) -> list[dict]:
    """Generate comparison summary data for all solutions."""
    return [solution_summary(data, sol) for sol in solutions]


def create_cost_comparison_figure(solutions: list[SolutionData], height: int = 400) -> go.Figure:
    """Grouped bar chart comparing cost components across approaches."""
    names = [s.name for s in solutions]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Facility Cost",
        x=names,
        y=[s.facility_cost for s in solutions],
        marker_color="royalblue",
        text=[f"${s.facility_cost:.2f}" for s in solutions],
        textposition="auto",
    ))
    fig.add_trace(go.Bar(
        name="Transport Cost",
        x=names,
        y=[s.transport_cost for s in solutions],
        marker_color="darkorange",
        text=[f"${s.transport_cost:.2f}" for s in solutions],
        textposition="auto",
    ))
    fig.update_layout(
        barmode="stack",
        title=dict(text="Cost Comparison", x=0.5),
        yaxis_title="Cost ($)",
        height=height,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def create_capacity_comparison_figure(
    data: ProblemData,
    solutions: list[SolutionData],
    height: int = 400,
) -> go.Figure:
    """Bar chart comparing capacity utilization across approaches."""
    names = [s.name for s in solutions]

    avg_utils = []
    spare_caps = []
    for sol in solutions:
        open_idx = np.where(sol.x > 0.5)[0]
        if len(open_idx) > 0:
            used = np.array([np.sum(sol.y[i, :]) for i in open_idx])
            caps = data.s[open_idx]
            avg_utils.append(np.mean(used / caps) * 100)
            spare_caps.append(np.sum(caps - used))
        else:
            avg_utils.append(0)
            spare_caps.append(0)

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Avg Utilization (%)", "Total Spare Capacity"))
    fig.add_trace(go.Bar(x=names, y=avg_utils, marker_color="teal",
                          text=[f"{u:.1f}%" for u in avg_utils], textposition="auto",
                          showlegend=False), row=1, col=1)
    fig.add_trace(go.Bar(x=names, y=spare_caps, marker_color="salmon",
                          text=[f"{s:.2f}" for s in spare_caps], textposition="auto",
                          showlegend=False), row=1, col=2)
    fig.update_layout(height=height, margin=dict(l=40, r=40, t=60, b=40))
    return fig


def create_pareto_figure(sweep_results: list[dict], height: int = 400) -> go.Figure:
    """
    Plot the cost vs robustness Pareto frontier.

    sweep_results: list of dicts with keys "gamma", "objective", "facilities_opened"
    """
    gammas = [r["gamma"] for r in sweep_results]
    costs = [r["objective"] for r in sweep_results]
    facilities = [r["facilities_opened"] for r in sweep_results]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=gammas, y=costs, mode="lines+markers",
                   name="Total Cost", line=dict(color="royalblue", width=2),
                   marker=dict(size=8)),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=gammas, y=facilities, mode="lines+markers",
                   name="Facilities Opened", line=dict(color="darkorange", width=2, dash="dash"),
                   marker=dict(size=8)),
        secondary_y=True,
    )

    fig.update_xaxes(title_text="Γ (Budget of Uncertainty)")
    fig.update_yaxes(title_text="Total Cost ($)", secondary_y=False)
    fig.update_yaxes(title_text="# Facilities Opened", secondary_y=True)
    fig.update_layout(
        title=dict(text="Price of Robustness — Cost vs Protection Level", x=0.5),
        height=height,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig
