"""
Network visualization for facility location solutions.

Creates Plotly figures showing facilities, customers, and flow connections.
"""

import numpy as np
import plotly.graph_objects as go

from core.data import ProblemData, SolutionData


def create_network_figure(
    data: ProblemData,
    solution: SolutionData = None,
    title: str = "Facility Location Network",
    show_flows: bool = True,
    height: int = 500,
) -> go.Figure:
    """
    Create a Plotly figure of the facility-customer network.

    Parameters
    ----------
    data : ProblemData
        Problem instance.
    solution : SolutionData, optional
        If provided, shows open/closed facilities and flow lines.
    title : str
        Figure title.
    show_flows : bool
        Whether to draw flow lines between facilities and customers.
    height : int
        Figure height in pixels.
    """
    fig = go.Figure()

    if solution is not None:
        # Draw flow lines first (behind markers)
        if show_flows:
            _add_flow_lines(fig, data, solution)

        # Open facilities (filled squares)
        open_mask = solution.x > 0.5
        if np.any(open_mask):
            open_idx = np.where(open_mask)[0]
            utilizations = []
            for i in open_idx:
                used = np.sum(solution.y[i, :])
                utilizations.append(f"Facility {i}<br>Cost: {data.f[i]:.2f}<br>"
                                    f"Capacity: {data.s[i]:.2f}<br>"
                                    f"Used: {used:.2f} ({100*used/data.s[i]:.0f}%)")
            fig.add_trace(go.Scatter(
                x=data.facilities[open_idx, 0],
                y=data.facilities[open_idx, 1],
                mode="markers",
                marker=dict(
                    size=data.s[open_idx] * 1.2,
                    color="royalblue",
                    symbol="square",
                    line=dict(width=2, color="darkblue"),
                    opacity=0.8,
                ),
                text=utilizations,
                hoverinfo="text",
                name="Open Facilities",
            ))

        # Closed facilities (hollow crosses)
        closed_mask = ~open_mask
        if np.any(closed_mask):
            closed_idx = np.where(closed_mask)[0]
            fig.add_trace(go.Scatter(
                x=data.facilities[closed_idx, 0],
                y=data.facilities[closed_idx, 1],
                mode="markers",
                marker=dict(
                    size=12,
                    color="lightgray",
                    symbol="x",
                    line=dict(width=2, color="gray"),
                ),
                text=[f"Facility {i} (closed)" for i in closed_idx],
                hoverinfo="text",
                name="Closed Facilities",
            ))

        # Customers (colored by primary supplier)
        primary_supplier = np.argmax(solution.y, axis=0)
        colors = _facility_colors(data.n)
        customer_colors = [colors[primary_supplier[j]] if np.sum(solution.y[:, j]) > 1e-6 else "gray"
                           for j in range(data.m)]
        customer_text = [
            f"Customer {j}<br>Demand: {data.d[j]:.3f}<br>"
            f"Served by: Facility {primary_supplier[j]}<br>"
            f"Served: {np.sum(solution.y[:, j]):.3f}"
            for j in range(data.m)
        ]
        fig.add_trace(go.Scatter(
            x=data.customers[:, 0],
            y=data.customers[:, 1],
            mode="markers",
            marker=dict(
                size=data.d * 15,
                color=customer_colors,
                symbol="circle",
                line=dict(width=1, color="darkgray"),
                opacity=0.7,
            ),
            text=customer_text,
            hoverinfo="text",
            name="Customers",
        ))
    else:
        # No solution — just show locations
        fig.add_trace(go.Scatter(
            x=data.facilities[:, 0],
            y=data.facilities[:, 1],
            mode="markers",
            marker=dict(size=14, color="royalblue", symbol="square",
                        line=dict(width=2, color="darkblue")),
            text=[f"Facility {i}<br>Cost: {data.f[i]:.2f}<br>Capacity: {data.s[i]:.2f}"
                  for i in range(data.n)],
            hoverinfo="text",
            name="Candidate Facilities",
        ))
        fig.add_trace(go.Scatter(
            x=data.customers[:, 0],
            y=data.customers[:, 1],
            mode="markers",
            marker=dict(size=data.d * 15, color="darkorange", symbol="circle",
                        line=dict(width=1, color="darkgray"), opacity=0.7),
            text=[f"Customer {j}<br>Demand: {data.d[j]:.3f}" for j in range(data.m)],
            hoverinfo="text",
            name="Customers",
        ))

    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis=dict(range=[-0.05, 1.05], showgrid=False, zeroline=False, title="X"),
        yaxis=dict(range=[-0.05, 1.05], showgrid=False, zeroline=False, title="Y",
                   scaleanchor="x", scaleratio=1),
        height=height,
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        plot_bgcolor="white",
    )

    return fig


def _add_flow_lines(fig: go.Figure, data: ProblemData, solution: SolutionData):
    """Add flow lines between facilities and customers."""
    colors = _facility_colors(data.n)
    max_flow = max(np.max(solution.y), 1e-6)

    for i in range(data.n):
        for j in range(data.m):
            flow = solution.y[i, j]
            if flow > 1e-6:
                width = max(0.5, 4.0 * flow / max_flow)
                fig.add_trace(go.Scatter(
                    x=[data.facilities[i, 0], data.customers[j, 0]],
                    y=[data.facilities[i, 1], data.customers[j, 1]],
                    mode="lines",
                    line=dict(width=width, color=colors[i]),
                    opacity=0.3,
                    hoverinfo="skip",
                    showlegend=False,
                ))


def _facility_colors(n: int) -> list:
    """Generate a list of distinct colors for facilities."""
    base_colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    ]
    return [base_colors[i % len(base_colors)] for i in range(n)]


def create_cost_breakdown_figure(solution: SolutionData, height: int = 300) -> go.Figure:
    """Create a horizontal bar chart showing cost breakdown."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=["Cost"],
        x=[solution.facility_cost],
        name="Facility Cost",
        orientation="h",
        marker_color="royalblue",
        text=[f"${solution.facility_cost:.2f}"],
        textposition="inside",
    ))
    fig.add_trace(go.Bar(
        y=["Cost"],
        x=[solution.transport_cost],
        name="Transport Cost",
        orientation="h",
        marker_color="darkorange",
        text=[f"${solution.transport_cost:.2f}"],
        textposition="inside",
    ))
    fig.update_layout(
        barmode="stack",
        title=dict(text=f"Total Cost: ${solution.objective:.2f}", x=0.5),
        height=height,
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
    )
    return fig


def create_capacity_chart(data: ProblemData, solution: SolutionData, height: int = 300) -> go.Figure:
    """Create a bar chart showing capacity utilization per facility."""
    open_idx = np.where(solution.x > 0.5)[0]
    if len(open_idx) == 0:
        return go.Figure()

    used = [np.sum(solution.y[i, :]) for i in open_idx]
    available = [data.s[i] for i in open_idx]
    labels = [f"F{i}" for i in open_idx]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=used, name="Used Capacity",
        marker_color="royalblue",
    ))
    fig.add_trace(go.Bar(
        x=labels, y=[a - u for a, u in zip(available, used)], name="Spare Capacity",
        marker_color="lightblue",
    ))
    fig.update_layout(
        barmode="stack",
        title=dict(text="Capacity Utilization", x=0.5),
        yaxis_title="Units",
        height=height,
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
    )
    return fig
