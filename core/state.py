"""
Streamlit session-state helpers for scenario-aware caching.
"""

from __future__ import annotations

import json
from hashlib import blake2b
from typing import Any

import numpy as np
import streamlit as st

from .data import ProblemData
from .data import generate_instance


DERIVED_STATE_KEYS = (
    "nominal_solution",
    "robust_solution",
    "ellipsoidal_solution",
    "adaptive_solution",
    "comparison_solutions",
    "mc_results",
    "pareto_results",
)


def problem_key(data: ProblemData) -> str:
    """Create a stable fingerprint for a problem instance."""
    digest = blake2b(digest_size=16)
    digest.update(f"{data.n}|{data.m}".encode("utf-8"))

    for array in (
        data.facilities,
        data.customers,
        data.c,
        data.f,
        data.s,
        data.d,
        data.P,
    ):
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("utf-8"))
        digest.update(str(contiguous.shape).encode("utf-8"))
        digest.update(contiguous.tobytes())

    return digest.hexdigest()


def make_model_key(**kwargs: Any) -> str:
    """Create a stable cache key for model parameters."""
    return json.dumps(kwargs, sort_keys=True, separators=(",", ":"))


def clear_derived_state() -> None:
    """Remove cached results derived from the current problem instance."""
    for key in DERIVED_STATE_KEYS:
        st.session_state.pop(key, None)


def set_problem_data(data: ProblemData) -> ProblemData:
    """Persist problem data and invalidate derived results if the scenario changed."""
    new_key = problem_key(data)
    old_key = st.session_state.get("problem_key")

    st.session_state["problem_data"] = data
    st.session_state["problem_key"] = new_key

    if old_key is not None and old_key != new_key:
        clear_derived_state()

    return data


def get_problem_data() -> ProblemData:
    """Load the active problem instance, creating the default one if needed."""
    data = st.session_state.get("problem_data")
    if data is None:
        data = generate_instance()
        return set_problem_data(data)

    if st.session_state.get("problem_key") != problem_key(data):
        set_problem_data(data)

    return st.session_state["problem_data"]


def store_cached_value(
    cache_key: str,
    value: Any,
    problem_signature: str,
    model_signature: str = "",
) -> Any:
    """Store a cached object together with the scenario/model keys that produced it."""
    st.session_state[cache_key] = {
        "problem_key": problem_signature,
        "model_key": model_signature,
        "value": value,
    }
    return value


def load_cached_value(
    cache_key: str,
    problem_signature: str,
    model_signature: str = "",
):
    """Load a cached object only when its scenario/model keys still match."""
    cached = st.session_state.get(cache_key)
    if not cached:
        return None

    if cached.get("problem_key") != problem_signature:
        return None

    if model_signature and cached.get("model_key") != model_signature:
        return None

    return cached.get("value")
