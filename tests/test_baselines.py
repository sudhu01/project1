"""Tests for immediate-stop and random acquisition baselines."""

import json

import numpy as np
import pytest

from rca_sim.baselines import (
    ImmediateStopPolicy,
    RandomIncludingStopPolicy,
    ScriptedInvestigatorPolicy,
)
from rca_sim.environment import InvestigationEnv
from rca_sim.evaluate import aggregate_results, evaluate_baselines, main
from rca_sim.tools import STOP_ACTION_INDEX
from rca_sim.world import generate_incident_world


def _cases(count: int = 4):
    return tuple(
        generate_incident_world(n_services=8, incident_seed=900 + index)
        for index in range(count)
    )


def test_immediate_stop_never_acquires_evidence() -> None:
    rows = evaluate_baselines(
        _cases(),
        include_random=False,
        include_random_smoke=False,
        include_script=False,
    )

    assert len(rows) == 4
    assert {row.method for row in rows} == {"stop"}
    assert {row.action_seed for row in rows} == {None}
    assert all(row.actions == (STOP_ACTION_INDEX,) for row in rows)
    assert all(row.probes_taken == 0 for row in rows)
    assert all(row.credits_spent == 0 for row in rows)
    assert all(row.episode_return == float(row.diagnosis_correct) for row in rows)


@pytest.mark.parametrize("probe_budget", [1, 2, 4, 6])
def test_random_acquisition_excludes_stop_until_probe_budget(probe_budget: int) -> None:
    rows = evaluate_baselines(
        _cases(1),
        probe_budgets=(probe_budget,),
        action_seeds=(17,),
        include_stop=False,
        include_random_smoke=False,
        include_script=False,
    )
    row = rows[0]

    assert row.method == f"random_{probe_budget}_probes"
    assert all(action != STOP_ACTION_INDEX for action in row.actions[:-1])
    if row.termination_reason == "stop":
        assert row.actions[-1] == STOP_ACTION_INDEX
        assert row.probes_taken == probe_budget
    else:
        assert row.actions[-1] != STOP_ACTION_INDEX
        assert row.probes_taken <= probe_budget


def test_random_policy_replays_with_same_action_seed_on_same_cases() -> None:
    settings = dict(
        probe_budgets=(4,),
        action_seeds=(23,),
        include_stop=False,
        include_random_smoke=False,
        include_script=False,
    )

    first = evaluate_baselines(_cases(), **settings)
    second = evaluate_baselines(_cases(), **settings)

    assert first == second


def test_random_action_seed_does_not_change_incident_cases() -> None:
    rows = evaluate_baselines(
        _cases(2),
        probe_budgets=(2,),
        action_seeds=(1, 2, 3),
        include_stop=False,
        include_random_smoke=False,
        include_script=False,
    )

    by_case = {}
    for row in rows:
        by_case.setdefault(row.case_id, set()).add(row.action_seed)
    assert len(by_case) == 2
    assert all(seeds == {1, 2, 3} for seeds in by_case.values())


def test_random_including_stop_samples_only_feasible_actions() -> None:
    env = InvestigationEnv()
    observation, _ = env.reset(seed=71)
    policy = RandomIncludingStopPolicy(seed=72)

    action = policy.select_action(observation)

    assert observation["action_mask"][action]


def test_policy_rejects_malformed_action_mask() -> None:
    policy = ImmediateStopPolicy()
    with pytest.raises(ValueError, match="action_mask"):
        policy.select_action({"action_mask": np.ones(40, dtype=np.bool_)})


def test_aggregate_results_keeps_random_variants_separate() -> None:
    rows = evaluate_baselines(
        _cases(3),
        probe_budgets=(1, 2),
        action_seeds=(4, 5),
        include_random_smoke=False,
        include_script=False,
    )

    summary = {row.method: row for row in aggregate_results(rows)}

    assert set(summary) == {"stop", "random_1_probes", "random_2_probes"}
    assert summary["stop"].episodes == 3
    assert summary["stop"].action_seeds == 0
    assert summary["random_1_probes"].episodes == 6
    assert summary["random_1_probes"].cases == 3
    assert summary["random_1_probes"].action_seeds == 2


def test_evaluate_cli_writes_episode_rows_and_summary(tmp_path) -> None:
    case_dir = tmp_path / "cases"
    case_dir.mkdir()
    for index, seed in enumerate((1001, 1002)):
        (case_dir / f"case_{index:03d}.json").write_text(
            json.dumps(
                {
                    "case_id": f"validation-{index:03d}",
                    "generator_version": "sim_v0",
                    "incident_seed": seed,
                    "n_services": 8,
                    "extra_edge_probability": 0.15,
                }
            ),
            encoding="utf-8",
        )
    output = tmp_path / "results"

    exit_code = main(
        [
            "--cases",
            str(case_dir),
            "--methods",
            "stop",
            "random",
            "--probe-budgets",
            "1",
            "--action-seeds",
            "7",
            "8",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    episode_rows = [
        json.loads(line)
        for line in (output / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert len(episode_rows) == 6
    assert {row["case_id"] for row in episode_rows} == {
        "validation-000",
        "validation-001",
    }
    assert {row["method"] for row in summary} == {"stop", "random_1_probes"}


def test_script_starts_at_entry_then_uses_dependency_breadth_first_order() -> None:
    env = InvestigationEnv()
    observation, info = env.reset(seed=930)
    policy = ScriptedInvestigatorPolicy(confidence_threshold=1.0)
    entry = int(info["entry_service"])

    first_action = policy.select_action(observation)
    observation, _, terminated, _, _ = env.step(first_action)

    assert first_action == 4 * entry
    assert not terminated

    adjacency = observation["adjacency"]
    order = [entry]
    seen = {entry}
    cursor = 0
    while cursor < len(order):
        caller = order[cursor]
        cursor += 1
        for dependency in np.flatnonzero(adjacency[caller]):
            service_id = int(dependency)
            if service_id not in seen:
                seen.add(service_id)
                order.append(service_id)

    second_action = policy.select_action(observation)
    assert second_action == 4 * order[1]


def test_script_checks_confidence_after_entry_probe() -> None:
    env = InvestigationEnv()
    observation, info = env.reset(seed=931)
    policy = ScriptedInvestigatorPolicy(confidence_threshold=0.0)

    first_action = policy.select_action(observation)
    observation, _, terminated, _, _ = env.step(first_action)

    assert first_action == 4 * int(info["entry_service"])
    assert not terminated
    assert policy.select_action(observation) == STOP_ACTION_INDEX


def test_script_threshold_variants_are_evaluated_without_action_seeds() -> None:
    rows = evaluate_baselines(
        _cases(2),
        script_thresholds=(0.60, 0.75, 0.90, 0.99),
        include_stop=False,
        include_random=False,
        include_random_smoke=False,
    )

    assert len(rows) == 8
    assert {row.method for row in rows} == {
        "script_0.60",
        "script_0.75",
        "script_0.90",
        "script_0.99",
    }
    assert {row.action_seed for row in rows} == {None}
    assert all(row.actions[0] != STOP_ACTION_INDEX for row in rows)


@pytest.mark.parametrize("threshold", [-0.01, 1.01, float("nan")])
def test_script_rejects_invalid_confidence_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match="confidence_threshold"):
        ScriptedInvestigatorPolicy(confidence_threshold=threshold)
