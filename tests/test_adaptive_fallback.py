from __future__ import annotations

import numpy as np
import pytest

import core.models_advanced as models_advanced
from core.data import generate_instance


def _fake_relaxed(evaluation_result, loop_x=(1.0, 1.0, 0.0)):
    """Stand-in for _solve_adaptive_relaxed: binary openings, then a chosen re-solve outcome."""

    def fake(data, rho, gamma, fixed_to_one, fixed_to_zero=None, time_limit=120):
        if fixed_to_zero is None:
            # Rounding loop: already binary, so solve_adaptive stops after one pass.
            return {"x": np.array(loop_x), "objective": 0.0, "proven_optimal": True}
        # Post-rounding re-solve with the openings fixed.
        return evaluation_result

    return fake


def test_failed_post_rounding_resolve_sets_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    data = generate_instance(n=3, m=4, seed=0)
    monkeypatch.setattr(models_advanced, "_solve_adaptive_relaxed", _fake_relaxed(None))

    solution = models_advanced.solve_adaptive(data, rho=1.0, gamma=3.0)

    assert solution.status == "Approximate (rounded)"
    assert "not a worst-case adaptive cost" in solution.warning


def test_successful_post_rounding_resolve_has_no_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    data = generate_instance(n=3, m=4, seed=0)
    evaluation = {"x": np.array([1.0, 1.0, 0.0]), "objective": 100.0, "proven_optimal": True}
    monkeypatch.setattr(models_advanced, "_solve_adaptive_relaxed", _fake_relaxed(evaluation))

    solution = models_advanced.solve_adaptive(data, rho=1.0, gamma=3.0)

    assert solution.status == "Optimal (rounded)"
    assert solution.objective == pytest.approx(100.0)
    assert solution.warning == ""


def test_timed_out_post_rounding_resolve_sets_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    # CBC's time limit yields status 1 with an unproven iterate; that must not pass as optimal.
    data = generate_instance(n=3, m=4, seed=0)
    evaluation = {"x": np.array([1.0, 1.0, 0.0]), "objective": 100.0, "proven_optimal": False}
    monkeypatch.setattr(models_advanced, "_solve_adaptive_relaxed", _fake_relaxed(evaluation))

    solution = models_advanced.solve_adaptive(data, rho=1.0, gamma=3.0)

    assert solution.status == "Approximate (rounded)"
    assert "not a worst-case adaptive cost" in solution.warning


def test_infeasible_nominal_dispatch_is_flagged(monkeypatch: pytest.MonkeyPatch) -> None:
    data = generate_instance(n=3, m=4, seed=0)
    monkeypatch.setattr(
        models_advanced, "_solve_adaptive_relaxed", _fake_relaxed(None, loop_x=(0.0, 0.0, 0.0))
    )

    solution = models_advanced.solve_adaptive(data, rho=1.0, gamma=3.0)

    assert solution.status == "Approximate (rounded)"
    assert "nominal dispatch for these openings is also infeasible" in solution.warning
