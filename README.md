# Robust Facility Location

A warehouse plan is cheap until demand moves. This app lets the forecast lie a little, then a lot, and asks which facilities still earn their rent.

## What it does

This Streamlit app explores facility-location decisions under uncertain demand. You configure a synthetic network, solve it four ways, and stress-test each solution with Monte Carlo scenarios.

The app uses open-source solvers only.

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Four approaches

| Approach | What it assumes |
|---|---|
| Nominal | The forecast is correct. Cheapest plan, shortest fuse. |
| Polyhedral robust | Demand can move within a Bertsimas-Sim box-plus-budget set. |
| Ellipsoidal robust | Demand moves inside an L2 ball. |
| Adaptive | Facilities are decided up front; flows react to realized demand through affine rules. |

Binary facility variables in the adaptive model are relaxed to `[0,1]` and recovered by greedy rounding.

## What to inspect

- How fragile the cost-optimal plan is when demand shifts.
- How protection cost changes as `Gamma` grows.
- Whether box, ellipsoid, or L1 geometry matches the story you are telling.
- Whether adaptive flows reduce the cost of protection on a given instance.
- How tail risk looks under Monte Carlo: infeasibility, 95th percentile, CVaR, and max cost.

## Formulations

Nominal MILP:

```text
min  sum_i f_i x_i + sum_ij c_ij y_ij
s.t. sum_i y_ij >= d_j              for each customer j
     sum_j y_ij <= s_i x_i          for each facility i
     x_i in {0,1}, y_ij >= 0
```

Polyhedral uncertainty:

```text
U = { z : |z_k| <= rho, ||z||_1 <= Gamma }
```

Ellipsoidal uncertainty:

```text
||z||_2 <= Omega
```

Adaptive flows:

```text
y_ij(z) = u_ij + sum_k V_ijk * z_k
```

## Live demo

Deploy with Streamlit Cloud using:

```text
streamlit_app.py
```

Local run:

```bash
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

## Floorplan

```text
app.py
core/
  data.py
  models.py
  models_advanced.py
  simulation.py
  state.py
  utils.py
viz/
  network.py
  comparison.py
  simulation_plots.py
pages/
  1_Problem_Setup.py
  2_Nominal_Solution.py
  3_Robust_Solution.py
  4_Advanced_Models.py
  5_Comparison.py
  6_Monte_Carlo.py
```

## Connects to

- `semiconductor-e2e-manufacturing-optimization` for the same uncertainty idea in wafer sourcing.
- `world-food-program-robust-simulator` for humanitarian network planning under disrupted supply.
- `procurement-negotiation-lab` for the negotiation layer after a location or sourcing choice.

## Limits

- The adaptive model can be slow on larger instances.
- Instances are synthetic and meant for exploration.
- Monte Carlo fixes facility openings and re-optimizes flows inside a single exploration model.

## References

- Bertsimas and Sim, *The Price of Robustness* (2004): https://doi.org/10.1287/opre.1030.0065
- Ben-Tal, El Ghaoui, and Nemirovski, *Robust Optimization* (2009): https://doi.org/10.1515/9781400831050
- Bertsimas and Tsitsiklis, *Introduction to Linear Optimization*

## License

MIT.
