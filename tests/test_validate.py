"""Tests for the step 8.1 and 8.2 validation runner."""

import json

import pytest

import rca_sim.validate as validation


def test_full_probability_and_inference_validation_passes() -> None:
    report = validation.run_validation()

    assert report.passed
    assert report.observation_samples_per_hypothesis == 10_000
    assert report.prior_samples == 60_000
    assert report.transition_samples == 10_000
    assert len(report.checks) == 14
    assert all(check.passed for check in report.checks)
    assert {check.name.split()[0] for check in report.checks} == {
        "8.1",
        "8.2",
        "8.3",
        "8.4",
        "8.5",
    }


def test_validation_rejects_statistically_inadequate_sample_counts() -> None:
    with pytest.raises(ValueError, match="observation_samples"):
        validation.run_validation(observation_samples=9_999)
    with pytest.raises(ValueError, match="prior_samples"):
        validation.run_validation(prior_samples=9_999)
    with pytest.raises(ValueError, match="transition_samples"):
        validation.run_validation(transition_samples=9_999)


def test_cli_writes_the_same_machine_readable_report(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = validation.ValidationReport(
        seed=7,
        observation_samples_per_hypothesis=10_000,
        prior_samples=10_000,
        transition_samples=10_000,
        checks=(validation.ValidationCheck("8.1 fixture", True, {"value": 1}),),
    )
    monkeypatch.setattr(validation, "run_validation", lambda **kwargs: expected)
    output = tmp_path / "validation.json"

    exit_code = validation.main(
        [
            "--seed",
            "7",
            "--observation-samples",
            "10000",
            "--prior-samples",
            "10000",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    assert json.loads(output.read_text(encoding="utf-8")) == expected.as_dict()
