# Evaluate the rounded policy after solving the relaxation

Method update: October 18, 2026. Proposed experiments below are not reported as executed results.

The adaptive formulation relaxes facility decisions and recovers binary openings by rounding. A fractional objective can provide useful optimization information, while the rounded deployment is a different policy. Its capacity and service feasibility must be checked again before comparing costs with nominal or robust integer solutions.

The next proposed synthetic fixture makes two facilities partly open in the relaxation, then creates a customer whose feasible service depends on the smaller opening. After rounding, re-optimize flows with fixed binary openings and verify every demand and capacity constraint. If the rounded policy fails, report that failure instead of keeping the relaxation's attractive objective.

For Monte Carlo comparison, freeze openings before observing each evaluation demand draw and use the same draws for every policy. Report feasible-draw count, unmet demand, total cost and the tail metric's conditioning. A CVaR calculated only over successful runs can hide catastrophic infeasibility. Keep solver status and rounding rule version in the result receipt. Uncertainty-set geometry and probabilistic coverage remain separate claims.

## References

- https://doi.org/10.1287/opre.1030.0065
- https://web.stanford.edu/~boyd/cvxbook/
