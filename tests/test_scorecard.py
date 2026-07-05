from __future__ import annotations

import json
import numbers
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def run_scorecard(output_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "reports.scorecard", "--output", str(output_path)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.fixture(scope="module")
def scorecard_rows() -> list[dict]:
    with tempfile.TemporaryDirectory(prefix="scorecard-", dir=REPO_ROOT / "tests") as temp_dir:
        output_path = Path(temp_dir) / "scorecard.jsonl"
        run_scorecard(output_path)
        return [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]


def test_scorecard_shape(scorecard_rows: list[dict]) -> None:
    assert len(scorecard_rows) == 6
    assert sum(1 for row in scorecard_rows if "Adaptive" in row["Approach"]) == 1

    nominal_by_instance = {
        row["instance"]: row["Total Cost"]
        for row in scorecard_rows
        if "Nominal" in row["Approach"]
    }

    for row in scorecard_rows:
        assert "Approach" in row
        assert "Total Cost" in row
        assert "cost_delta" in row
        assert "instance" in row
        assert "Solve Time (s)" not in row
        assert isinstance(row["cost_delta"], numbers.Real)
        if "Nominal" not in row["Approach"]:
            assert row["Total Cost"] >= nominal_by_instance[row["instance"]] - 1e-6


def test_scorecard_output_is_byte_identical() -> None:
    with tempfile.TemporaryDirectory(prefix="scorecard-", dir=REPO_ROOT / "tests") as temp_dir:
        temp_path = Path(temp_dir)
        first = temp_path / "first.jsonl"
        second = temp_path / "second.jsonl"

        run_scorecard(first)
        run_scorecard(second)

        assert first.read_bytes() == second.read_bytes()
