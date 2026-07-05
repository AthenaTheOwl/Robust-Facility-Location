from __future__ import annotations

import json
import os
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path
from typing import Iterable

DEFAULT_OUTPUT = Path(__file__).with_name("scorecard.jsonl")
BIG_INSTANCE = "n=8,m=30,seed=42"
ADR_INSTANCE = "n=5,m=12,seed=42"


class ScorecardError(Exception):
    """Typed failure surfaced by the headless scorecard command."""


def _dependencies_available() -> bool:
    return all(find_spec(module_name) is not None for module_name in ("numpy", "pulp"))


def _candidate_interpreters() -> list[Path]:
    candidates = []
    env_python = os.environ.get("SCORECARD_PYTHON")
    if env_python:
        candidates.append(Path(env_python))
    candidates.append(Path(r"C:\Python314\python.exe"))
    return [path for path in candidates if path.exists() and str(path) != sys.executable]


def _rerun_with_solver_python(argv: list[str]) -> int:
    for interpreter in _candidate_interpreters():
        check = subprocess.run(
            [str(interpreter), "-c", "import numpy, pulp"],
            capture_output=True,
            text=True,
        )
        if check.returncode != 0:
            continue

        env = dict(os.environ)
        env["SCORECARD_NO_REEXEC"] = "1"
        completed = subprocess.run(
            [str(interpreter), "-m", "reports.scorecard", *argv],
            env=env,
        )
        return completed.returncode

    raise ScorecardError("required Python packages missing: numpy, pulp")


def _parse_output_path(argv: list[str]) -> Path:
    if not argv:
        return DEFAULT_OUTPUT
    if len(argv) == 2 and argv[0] == "--output":
        return Path(argv[1])
    raise ScorecardError("usage: python -m reports.scorecard [--output PATH]")


def _json_ready(value):
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    try:
        return value.item()
    except AttributeError:
        return value


def build_rows() -> list[dict]:
    from core.data import generate_instance
    from core.models import solve_nominal, solve_robust
    from core.models_advanced import solve_adaptive, solve_robust_ellipsoidal
    from core.utils import solution_summary

    approaches = (
        ("Nominal", lambda data: solve_nominal(data)),
        ("Robust (Gamma=3)", lambda data: solve_robust(data, rho=1.0, gamma=3.0)),
        ("Robust (Gamma=5)", lambda data: solve_robust(data, rho=1.0, gamma=5.0)),
        ("Ellipsoidal", lambda data: solve_robust_ellipsoidal(data, omega=1.0)),
    )

    data = generate_instance(n=8, m=30, seed=42)
    summaries: list[dict] = []

    for label, solver in approaches:
        solution = solver(data)
        solution.name = label
        if solution.status != "Optimal":
            raise ScorecardError(f"{label} returned non-Optimal status: {solution.status}")

        row = solution_summary(data, solution)
        row.pop("Solve Time (s)", None)
        row["instance"] = BIG_INSTANCE
        summaries.append({key: _json_ready(value) for key, value in row.items()})

    nominal_total = float(summaries[0]["Total Cost"])
    for row in summaries:
        row["cost_delta"] = round(float(row["Total Cost"]) - nominal_total, 2)

    # solve_adaptive measured ~269s on n=8,m=30 versus ~3s here. Cross-instance
    # deltas are meaningless, so this ADR companion gets its own nominal row.
    data_small = generate_instance(n=5, m=12, seed=42)

    nominal_small = solve_nominal(data_small)
    nominal_small.name = "Nominal (ADR instance)"
    if nominal_small.status != "Optimal":
        raise ScorecardError(
            f"Nominal (ADR instance) returned non-Optimal status: {nominal_small.status}"
        )

    nominal_small_row = solution_summary(data_small, nominal_small)
    nominal_small_row.pop("Solve Time (s)", None)
    nominal_small_row["instance"] = ADR_INSTANCE
    nominal_small_json = {key: _json_ready(value) for key, value in nominal_small_row.items()}

    adaptive = solve_adaptive(data_small, rho=1.0, gamma=3.0)
    adaptive.name = "Adaptive (rho=1, Gamma=3)"
    if adaptive.status != "Optimal (rounded)":
        raise ScorecardError(
            "Adaptive (rho=1, Gamma=3) expected status 'Optimal (rounded)' "
            f"but got {adaptive.status!r}; 'Approximate (rounded)' depends on "
            "wall-clock LP time limits and is not deterministic enough to commit"
        )

    adaptive_row = solution_summary(data_small, adaptive)
    adaptive_row.pop("Solve Time (s)", None)
    adaptive_row["instance"] = ADR_INSTANCE
    adaptive_json = {key: _json_ready(value) for key, value in adaptive_row.items()}

    nominal_small_total = float(nominal_small_json["Total Cost"])
    for row in (nominal_small_json, adaptive_json):
        row["cost_delta"] = round(float(row["Total Cost"]) - nominal_small_total, 2)

    summaries.extend((nominal_small_json, adaptive_json))

    return summaries


def write_rows(rows: Iterable[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row, sort_keys=True) for row in rows]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    try:
        raw_argv = list(sys.argv[1:] if argv is None else argv)
        if not _dependencies_available() and os.environ.get("SCORECARD_NO_REEXEC") != "1":
            return _rerun_with_solver_python(raw_argv)

        output_path = _parse_output_path(raw_argv)
        rows = build_rows()
        write_rows(rows, output_path)
        robust_rows = [
            row
            for row in rows
            if row["instance"] == BIG_INSTANCE and row["Approach"] != "Nominal"
        ]
        cheapest = min(robust_rows, key=lambda row: float(row["Total Cost"]))
        adaptive_row = next(row for row in rows if "Adaptive" in row["Approach"])
        print(
            f"wrote {len(rows)} rows to {output_path}; "
            f"cheapest robust approach: {cheapest['Approach']}; "
            f"adaptive ADR-instance delta: {adaptive_row['cost_delta']:+.2f}"
        )
        return 0
    except ScorecardError as exc:
        print(f"ERROR[scorecard]: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"ERROR[scorecard]: unexpected failure: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
