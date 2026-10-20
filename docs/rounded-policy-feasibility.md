# Evaluate the rounded policy after solving the relaxation

Method update: October 18, 2026. Proposed experiments below are not reported as executed results.

The adaptive formulation relaxes facility decisions and recovers binary openings by rounding. A fractional objective can provide useful optimization information, while the rounded deployment is a different policy. Its capacity and service feasibility must be checked again before comparing costs with nominal or robust integer solutions. `solve_adaptive` already re-solves the adaptive robust model with the rounded openings fixed; when that re-solve returns no solution (infeasible or out of time), it falls back to a nominal-flow cost, labels the result `Approximate (rounded)` and sets a `warning` that the Advanced Models and Comparison pages show in place of the worst-case cost caption.

The next proposed synthetic fixture leaves a needed facility open at 0.01 or less in the final relaxation, which the last rounding step sets to zero, then creates a customer whose feasible service depends on that opening. Greedy rounding fixes the largest fractional opening to one and re-solves, so a merely smaller opening is re-optimized rather than dropped. After rounding, re-optimize flows with fixed binary openings and verify every demand and capacity constraint. If the rounded policy fails, report that failure instead of a nominal-flow cost or the relaxation's attractive objective.

For Monte Carlo comparison, `run_monte_carlo` already freezes openings before each evaluation demand draw and uses the same draws for every policy; keep that. Report feasible-draw count, unmet demand, total cost and the tail metric's conditioning. A CVaR calculated only over successful runs can hide catastrophic infeasibility. The current CVaR covers every draw and prices unmet demand at a fixed per-unit penalty, so its value depends on that penalty. Keep solver status and rounding rule version in the result receipt. Uncertainty-set geometry and probabilistic coverage remain separate claims.

## References

- https://doi.org/10.1287/opre.1030.0065
- https://web.stanford.edu/~boyd/cvxbook/
