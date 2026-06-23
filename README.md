<!-- ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ -->

# N° 06 · robust facility location explorer

> *where to build, when you don't know demand.*

the math, made tangible. configure a problem, solve it four different ways, and stress-test each solution with monte carlo — watching the cost-protection tradeoff play out across thousands of scenarios. no commercial solvers required.

`python` · `streamlit` · `pulp` + `CBC` · `plotly` · `MIT` · 2024 · **status: solved**

```bash
pip install -r requirements.txt
streamlit run app.py
```

<!-- ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ -->

## the problem

a firm decides **where to open facilities** — warehouses, hospitals, data centers, charging stations — to serve customers spread across a geography. each candidate has a fixed opening cost and a capacity limit. serving costs scale with distance. the objective: minimize opening + transport, while meeting every customer's demand.

the catch: **demand is uncertain when you decide.** the plan that's optimal for the forecast can fall over when reality drifts even modestly. so the question becomes: how do you choose locations that perform well across a *range* of plausible demands — and what does that protection cost?

## the four approaches

| approach | what it is |
|---|---|
| **nominal**         | classical MILP, assumes the forecast is correct. cheapest possible. most fragile. |
| **polyhedral robust** *(bertsimas-sim)* | worst-case demand within a box + budget set. two knobs: `ρ` (per-customer deviation) and `Γ` (how many can deviate at once). |
| **ellipsoidal robust** | swap the polyhedron for an L2 ball of radius `Ω`. less conservative — large deviations in some force small in others. stays MILP. |
| **adaptive** *(affine decision rules)* | facilities decided up front; flows are allowed to react to realized demand via `y(z) = u + V·z`. fewer facilities at the same protection level, typically. |

binary facility variables in the adaptive model are relaxed to `[0,1]` and recovered by greedy rounding (most-fractional-first, re-solve, repeat).

## what's there to study

| question | where |
|---|---|
| how fragile is the cost-optimal plan?              | page 2 — perturb demand with the "what if" slider |
| how much does robustness cost?                     | page 3 — nominal vs robust, side-by-side |
| how does conservatism scale with `Γ`?              | page 5 — pareto sweep |
| which uncertainty geometry fits the problem?       | page 4 — box, ellipsoid, and L1 ball in 2D |
| does adaptivity reduce the price of robustness?    | page 4 — adaptive vs static at the same `ρ`, `Γ` |
| which approach survives real-world randomness?     | page 6 — monte carlo with tunable scale, dist, correlation |
| how does spatial correlation change the solution?  | pages 1 & 3 — vary `R_D`, watch the P matrix shift |
| what's the tail risk?                              | page 6 — CVaR, 95th percentile, max cost |
| how does network geography matter?                 | page 1 — uniform, clustered, hub-and-spoke layouts |

## the monte carlo

page 6 fixes facility decisions, then generates 50–2,000 random demand scenarios and re-solves the operational dispatch for each. outputs:

- **cost histograms** — robust solutions sit tighter
- **CDF curves** — the most revealing chart in the app: nominal has a long expensive right tail; robust solutions don't
- **infeasibility rates** — fraction of scenarios with unmet demand
- **tail risk** — 95th percentile, CVaR (avg of worst 5%), max
- **box plots** — spread + outliers, compactly

configurable: scenarios, perturbation magnitude, distribution (uniform / normal), correlation on/off via the P matrix, seed.

## scenario-aware caching

problem instances and model parameters are fingerprinted (BLAKE2b for arrays, JSON hash for settings). solutions cache at two levels — problem signature and model signature. moving a slider doesn't re-solve models that haven't been affected. changing the instance invalidates everything downstream.

<!-- ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ -->

## the formulations

### nominal MILP
```
min  Σ fᵢxᵢ + Σ cᵢⱼyᵢⱼ
s.t. Σᵢ yᵢⱼ ≥ dⱼ              ∀j   demand
     Σⱼ yᵢⱼ ≤ sᵢxᵢ            ∀i   capacity
     xᵢ ∈ {0,1},  yᵢⱼ ≥ 0
```

### polyhedral robust counterpart

uncertainty set: `U = { z : |zₖ| ≤ ρ, ‖z‖₁ ≤ Γ }`. demand under perturbation: `d̃ⱼ = dⱼ + (P·z)ⱼ`. dualizing the inner max gives a tractable MILP with auxiliary variables (`ν` for the box dual, `λ` for the budget dual):

```
Σᵢ uᵢⱼ ≥ dⱼ + ρ·Σₖ νⱼₖ + Γ·λⱼ    ∀j
λⱼ + νⱼₖ ≥ +Pⱼₖ                    ∀j,k
λⱼ + νⱼₖ ≥ −Pⱼₖ                    ∀j,k
νⱼₖ, λⱼ ≥ 0
```

### ellipsoidal robust

uncertainty set: `‖z‖₂ ≤ Ω`. the support function of the L2 ball gives a precomputable constant per customer: `dⱼ + Ω·‖P[j,:]‖₂`. linear constraint, MILP overall.

### adaptive (affine decision rules)

flows become affine in the realization: `yᵢⱼ(z) = uᵢⱼ + Σₖ Vᵢⱼₖ · zₖ`. each constraint that must hold for all `z ∈ U` is dualized using the same box + budget pattern, with `V` as additional decisions.

<!-- ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ -->

## configurable parameters

**problem instance** — N candidate facilities (3–15), M customers (10–80), seed, layout (uniform / clustered / hub-and-spoke), demand range, facility cost range, capacity range.

**uncertainty** — `ρ`, `Γ`, `Ω`, `R_D` (spatial correlation radius for P).

**simulation** — # scenarios, scale, distribution, correlation on/off, seed.

## the floorplan

```
app.py                       streamlit entry + landing
core/
  data.py                    ProblemData/SolutionData, instance generation, P matrix
  models.py                  nominal MILP, polyhedral robust, operational LP
  models_advanced.py         ellipsoidal, adaptive + greedy rounding
  simulation.py              monte carlo, CVaR, summary stats
  state.py                   scenario-aware caching, BLAKE2b fingerprinting
  utils.py                   cost breakdowns, capacity utilization
viz/
  network.py                 facility-customer maps, cost bars
  comparison.py              grouped comparisons, pareto plots
  simulation_plots.py        histograms, CDFs, box plots, tail risk
pages/
  1_Problem_Setup.py
  2_Nominal_Solution.py
  3_Robust_Solution.py
  4_Advanced_Models.py
  5_Comparison.py
  6_Monte_Carlo.py
```

## limitations

- the adaptive model is the most expensive piece; can be slow on larger instances
- problem instances are synthetic — for teaching and exploration, not production planning
- the adaptive page reports the worst-case planning objective while plotting nominal dispatch flows, for interpretability
- monte carlo fixes facility openings and re-optimizes flows; not a full multistage operational process

## roadmap

- deployed demo + screenshots / GIFs
- curated scenario presets that make the nominal-vs-robust contrast visible with minimal tuning
- adaptive solve performance for medium instances
- scenario import / export
- model-invariant tests beyond import + compile

## live demo

Deploy with Streamlit Cloud using:

```text
streamlit_app.py
```

Local run:

```bash
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

## connects to

- `semiconductor-e2e-manufacturing-optimization` for the same uncertainty idea in wafer sourcing.
- `world-food-program-robust-simulator` for humanitarian network planning under disrupted supply.
- `procurement-negotiation-lab` for the negotiation layer that would sit after a location or sourcing choice.

## colophon

based on:
- bertsimas & sim, *the price of robustness* (2004) — https://doi.org/10.1287/opre.1030.0065
- ben-tal, el ghaoui & nemirovski, *robust optimization* (2009) — https://doi.org/10.1515/9781400831050
- bertsimas & tsitsiklis, *introduction to linear optimization*

`MIT` license. *built downstairs.* — [the basement, room 7](https://github.com/AthenaTheOwl)
