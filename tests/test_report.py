import json
from pathlib import Path

import pytest

from rca_sim.report import build_report


def _summary(path: Path, *, partition: str, seed: int | None = None) -> Path:
    path.mkdir(parents=True)
    method = "ppo" if seed is not None else "stop"
    cases = 1000 if partition == "test" else 256
    (path / "summary.json").write_text(json.dumps([{
        "method": method,
        "episodes": cases,
        "cases": cases,
        "action_seeds": 0,
        "accuracy": 0.5,
        "mean_return": 0.3,
        "mean_credits_spent": 4.0,
        "mean_probes_taken": 4.0,
        "mean_planning_seconds": 0.01,
        "mean_evidence_records": 12.0,
    }]), encoding="utf-8")
    return path


def test_report_keeps_test_rows_separate(tmp_path: Path) -> None:
    inputs = [
        _summary(tmp_path / f"test_s{seed}", partition="test", seed=seed)
        for seed in (101, 202, 303)
    ]
    output = tmp_path / "report"
    result = build_report(inputs, output)
    assert result["partitions"] == ["test"]
    assert result["ppo_test_across_seeds"]["mean_accuracy"] == 0.5
    assert (output / "report.md").is_file()


def test_report_rejects_wrong_test_case_count(tmp_path: Path) -> None:
    source = _summary(tmp_path / "test_s101", partition="test", seed=101)
    payload = json.loads((source / "summary.json").read_text())
    payload[0]["cases"] = 999
    (source / "summary.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="expected 1000"):
        build_report([source], tmp_path / "report")
