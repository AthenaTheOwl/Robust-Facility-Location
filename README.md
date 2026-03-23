# Robust Facility Location Explorer

An interactive application for exploring how **robust and adaptive optimization** protects facility location decisions against demand uncertainty.

The goal is to make the math tangible. Instead of reading formulations on paper, you configure a problem, solve it four different ways, and stress-test each solution with Monte Carlo simulation — watching how the cost-protection tradeoff plays out across thousands of scenarios.

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/built%20with-Streamlit-ff4b4b)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

## The Problem

A firm must decide **where to open facilities** — warehouses, hospitals, data centers, charging stations — to serve customers spread across a geography. Each candidate facility has a fixed opening cost and a capacity limit. Serving a customer from a facility costs proportional to the distance between them. The objective is to minimize the total of opening costs plus transportation costs while meeting every customer's demand.

The complication: **demand is uncertain at the time facilities are built.** You commit capital to locations and capacities today, but actual demand materializes later and may differ from the forecast. A plan that's optimal for the predicted demand can fail — over-capacity at some facilities, shortfalls at others — when reality deviates even modestly.

This is the core tension the app explores: **how to choose facility locations that perform well not just for one demand forecast, but across a range of plausible demand scenarios**, and what that protection costs.

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

No commercial solvers needed. All optimization uses [PuLP](https://coin-or.github.io/pulp/) with the free CBC solver.

## What Can Be Studied

| Question | Where to explore it |
|----------|-------------------|
| How fragile is a cost-optimal plan? | Page 2 — perturb demand with the "What If" slider and watch the nominal solution fail |
| How much does robustness cost? | Page 3 — compare nominal vs robust costs and facility counts side-by-side |
| How does conservatism scale with the uncertainty budget? | Page 5 — Pareto sweep over Γ shows the exact cost-vs-protection curve |
| Which uncertainty geometry fits the problem? | Page 4 — compare box, ellipsoid, and L1 ball in a 2D projection |
| Does adaptivity reduce the price of robustness? | Page 4 — compare adaptive cost to static robust at the same ρ, Γ |
| Which approach survives real-world randomness? | Page 6 — Monte Carlo stress test with tunable scale, distribution, and correlation |
| How does spatial demand correlation change the solution? | Pages 1 & 3 — vary R_D and watch the correlation matrix and solution shift |
| What's the tail risk of each approach? | Page 6 — CVaR, 95th percentile, max cost side-by-side |
| How does network geography affect vulnerability? | Page 1 — switch between Uniform, Clustered, and Hub-and-Spoke layouts |

## Features

### Four Optimization Approaches

**Nominal (Deterministic).**
Classical mixed-integer linear program assuming the demand forecast is perfect. This produces the cheapest possible network — and the most fragile. The app includes a "What If" slider (Page 2) that perturbs demand from -50% to +100%, showing exactly when unmet demand appears and penalty costs spike.

**Polyhedral Robust (Bertsimas-Sim).**
Protects against worst-case demand within a box + budget uncertainty set. Two parameters control conservatism:
- **ρ** (0 to 1) — how much any single customer's demand can deviate from forecast
- **Γ** (0 to M) — how many customers can deviate simultaneously

When Γ = 0 you get the nominal solution. When Γ = M every customer can be worst-case at once (extremely conservative). The interesting region is in between. Demand perturbations are spatially correlated through a matrix P, controlled by a radius parameter R_D — nearby customers' demands move together. The robust counterpart is reformulated via LP duality into a tractable MILP with auxiliary variables, following [Bertsimas & Sim (2004)](https://doi.org/10.1007/s10107-003-0396-4).

**Ellipsoidal Robust.**
Replaces the polyhedral set with an L2 ball (`‖z‖₂ ≤ Ω`). This is less conservative because it bounds total squared deviation — large deviations in some customers force small deviations in others. The worst-case demand margin for each customer reduces to `Ω · ‖P[j,:]‖₂`, a precomputable constant, so the model stays MILP. Page 4 includes a 2D geometry visualization comparing the box, ellipsoid, and L1 diamond to build geometric intuition. The formulation follows the framework in [Ben-Tal, El Ghaoui & Nemirovski (2009)](https://doi.org/10.1515/9781400831050).

**Adaptive (Affine Decision Rules).**
Facility locations are decided upfront, but operational flows are allowed to react to realized demand through affine policies: `y(z) = u + V·z`. This models the real-world flexibility a firm has to redirect shipments after observing actual orders. The result typically opens fewer facilities than static robust at the same protection level. Binary facility variables are relaxed to [0, 1] and recovered via a greedy rounding heuristic. The reported objective is the worst-case planning cost; network visualizations show nominal dispatch flows for interpretability.

### Monte Carlo Stress Test

Page 6 fixes the facility decisions from each approach, then generates hundreds or thousands of random demand scenarios and re-solves the operational dispatch for each one. The output includes:

- **Cost histograms** — overlaid distributions showing robust solutions have tighter, more predictable cost profiles
- **CDF curves** — the single most revealing visualization: the nominal solution has a long expensive right tail while robust solutions are concentrated
- **Infeasibility rates** — what fraction of scenarios result in unmet demand
- **Tail risk metrics** — 95th percentile, CVaR (average cost in the worst 5%), maximum cost
- **Box plots** — compact summary of spread and outliers

Simulation parameters are fully configurable: number of scenarios (50–2000), perturbation magnitude, distribution (Uniform or Normal), and whether demand perturbations are spatially correlated through the P matrix.

### Pareto Frontier

Page 5 sweeps the budget parameter Γ from 0 to M and plots total cost and number of open facilities against the uncertainty budget — the "price of robustness" curve. This shows exactly how much additional cost each increment of protection requires.

### Scenario-Aware Caching

Problem instances and model parameters are fingerprinted (BLAKE2b for data arrays, JSON hash for model settings). Solutions are cached at two levels — problem signature and model signature — so changing a slider doesn't re-solve models that haven't been affected. Changing the problem instance automatically invalidates all downstream results.

## Configurable Parameters

**Problem instance** — N candidate facilities (3–15), M customers (10–80), random seed, spatial layout (Uniform, Clustered, Hub-and-Spoke), demand range, facility cost range, capacity range.

**Uncertainty** — ρ (perturbation radius per demand), Γ (L1 budget across demands), Ω (ellipsoid radius), R_D (spatial correlation radius for the P matrix).

**Simulation** — number of Monte Carlo scenarios, perturbation scale, distribution family, correlation on/off, simulation seed.

## Mathematical Formulations

### Nominal MILP
```
min  Σ fᵢxᵢ + Σ cᵢⱼyᵢⱼ
s.t. Σᵢ yᵢⱼ ≥ dⱼ           ∀j  (demand satisfaction)
     Σⱼ yᵢⱼ ≤ sᵢxᵢ         ∀i  (capacity linking)
     xᵢ ∈ {0,1}, yᵢⱼ ≥ 0
```

### Polyhedral Robust Counterpart

Uncertainty set: `U = { z : |zₖ| ≤ ρ, ‖z‖₁ ≤ Γ }`

Demand under perturbation: `d̃ⱼ = dⱼ + (P·z)ⱼ` where P is a spatial correlation matrix.

The worst-case demand for each customer j is `dⱼ + max_{z∈U} (P·z)ⱼ`. Dualizing the inner maximization over the polyhedral set introduces auxiliary variables (ν for the box dual, λ for the budget dual) and yields a tractable MILP:

```
Σᵢ uᵢⱼ ≥ dⱼ + ρ·Σₖ νⱼₖ + Γ·λⱼ    ∀j
λⱼ + νⱼₖ ≥ +Pⱼₖ                      ∀j,k
λⱼ + νⱼₖ ≥ −Pⱼₖ                      ∀j,k
νⱼₖ ≥ 0,  λⱼ ≥ 0
```

### Ellipsoidal Robust

Uncertainty set: `U = { z : ‖z‖₂ ≤ Ω }`

The support function of the L2 ball gives: `max_{‖z‖₂ ≤ Ω} Pⱼ·z = Ω·‖P[j,:]‖₂`. This is a precomputable constant per customer, so the robust demand constraint becomes `Σᵢ yᵢⱼ ≥ dⱼ + Ω·‖P[j,:]‖₂` — still a linear constraint, and the overall model stays MILP.

### Adaptive (Affine Decision Rules)

Flow variables become affine functions of the uncertainty realization:

```
yᵢⱼ(z) = uᵢⱼ + Σₖ Vᵢⱼₖ · zₖ
```

Each constraint (positivity, demand, capacity) that must hold for all z ∈ U is dualized using the same box + budget duality pattern as the polyhedral robust model, but now with the affine coefficients V as additional decision variables. Binary facility variables are relaxed to [0, 1] and recovered through greedy rounding: solve the relaxation, fix the most fractional facility to 1, re-solve, repeat until all facilities are binary.

## Tech Stack

| Component | Role |
|-----------|------|
| [Streamlit](https://streamlit.io) | Interactive multi-page web UI |
| [PuLP](https://coin-or.github.io/pulp/) + CBC | MILP solver (free, no commercial license) |
| [Plotly](https://plotly.com/python/) | Interactive network maps, charts, and distribution plots |
| [NumPy](https://numpy.org) | Array operations, distance matrices, correlation matrix |
| [Pandas](https://pandas.pydata.org) | Data tables and summary statistics |
| [SciPy](https://scipy.org) | Supporting numerical routines |
| Python 3.10+ | Language runtime |

All solvers are free and open source. No Gurobi, CPLEX, or other commercial license is required.

## Project Structure

```
├── app.py                    # Streamlit entry point and landing page
├── requirements.txt
├── core/
│   ├── data.py               # ProblemData/SolutionData dataclasses, instance generation, P matrix
│   ├── models.py             # Nominal MILP, polyhedral robust counterpart, operational LP
│   ├── models_advanced.py    # Ellipsoidal robust, adaptive affine policies + greedy rounding
│   ├── simulation.py         # Monte Carlo engine, CVaR computation, summary statistics
│   ├── state.py              # Scenario-aware caching, BLAKE2b fingerprinting, session state
│   └── utils.py              # Cost breakdowns, capacity utilization, solution summaries
├── viz/
│   ├── network.py            # Facility-customer network maps, cost bars, capacity charts
│   ├── comparison.py         # Grouped cost comparisons, Pareto frontier plots
│   └── simulation_plots.py   # Histograms, CDFs, box plots, infeasibility and tail risk charts
└── pages/
    ├── 1_Problem_Setup.py    # Configure instance, visualize geography and correlation
    ├── 2_Nominal_Solution.py # Solve deterministic MILP, "What If" fragility analysis
    ├── 3_Robust_Solution.py  # Polyhedral robust with ρ/Γ controls, nominal comparison
    ├── 4_Advanced_Models.py  # Ellipsoidal and adaptive tabs, uncertainty set geometry
    ├── 5_Comparison.py       # Side-by-side all approaches, Pareto sweep
    └── 6_Monte_Carlo.py      # Stress test, cost distributions, risk metrics
```

## License

MIT
