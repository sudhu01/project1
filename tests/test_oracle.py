"""Tests for exact one-step and finite-horizon planning references."""

import numpy as np
import pytest

from rca_sim.baselines import OneStepValueOfInformationPolicy
from rca_sim.environment import InvestigationEnv
from rca_sim.evaluate import evaluate_baselines
from rca_sim.fixtures import FixtureEnv, complementarity_fixture
from rca_sim.oracle import one_step_plan
from rca_sim.tools import STOP_ACTION_INDEX, probe_action_index
from rca_sim.world import generate_incident_world


def _initial_plan(seed: int = 940):
    env = InvestigationEnv()
    observation, _ = env.reset(
        options={"case": generate_incident_world(n_services=8, incident_seed=seed)}
    )
    plan = one_step_plan(
        graph=env.public_graph,
        posterior=env.belief_posterior,
        seen_evidence_ids=(),
        action_mask=observation["action_mask"],
        lambda_cost=env.lambda_cost,
    )
    return env, observation, plan


def test_one_step_plan_uses_uniform_prior_stop_value() -> None:
    _, _, plan = _initial_plan()

    assert plan.stop_value == pytest.approx(0.125)
    assert plan.action_values[STOP_ACTION_INDEX] == pytest.approx(0.125)
    assert plan.best_action in plan.action_values


def test_one_step_plan_enumerates_each_template_output_space() -> None:
    _, _, plan = _initial_plan()

    assert plan.outcome_counts[probe_action_index(0, "metrics.quick")] == 8
    assert plan.outcome_counts[probe_action_index(0, "metrics.detailed")] == 64
    assert plan.outcome_counts[probe_action_index(0, "logs.quick")] == 4
    assert plan.outcome_counts[probe_action_index(0, "logs.detailed")] == 64


def test_one_step_plan_only_enumerates_unseen_records() -> None:
    env, observation, _ = _initial_plan(seed=941)
    service_id = 0
    quick = probe_action_index(service_id, "metrics.quick")
    detailed = probe_action_index(service_id, "metrics.detailed")
    observation, _, terminated, _, _ = env.step(quick)
    assert not terminated

    plan = one_step_plan(
        graph=env.public_graph,
        posterior=env.belief_posterior,
        seen_evidence_ids=tuple(record.evidence_id for record in env.evidence_records),
        action_mask=observation["action_mask"],
        lambda_cost=env.lambda_cost,
    )

    assert quick not in plan.action_values
    assert plan.outcome_counts[detailed] == 8


def test_one_step_plan_favors_stop_when_probe_values_do_not_exceed_it() -> None:
    env, observation, _ = _initial_plan(seed=942)

    plan = one_step_plan(
        graph=env.public_graph,
        posterior=env.belief_posterior,
        seen_evidence_ids=(),
        action_mask=observation["action_mask"],
        lambda_cost=1.0,
    )

    assert plan.best_action == STOP_ACTION_INDEX


def test_voi_policy_uses_public_environment_state() -> None:
    env, observation, expected = _initial_plan(seed=943)
    policy = OneStepValueOfInformationPolicy(env)

    action = policy.select_action(observation)

    assert action == expected.best_action
    assert policy.last_plan == expected


def test_one_step_values_are_valid_expected_returns() -> None:
    _, _, plan = _initial_plan(seed=944)

    for action, value in plan.action_values.items():
        assert np.isfinite(value)
        assert value <= 1.0
        if action != STOP_ACTION_INDEX:
            assert value >= plan.stop_value - 0.20


def test_evaluator_records_voi_runtime_separately_from_probe_credits() -> None:
    case = generate_incident_world(n_services=8, incident_seed=946)

    rows = evaluate_baselines(
        (case,),
        include_stop=False,
        include_random=False,
        include_random_smoke=False,
        include_script=False,
    )

    assert len(rows) == 1
    assert rows[0].method == "voi1"
    assert rows[0].action_seed is None
    assert rows[0].planning_seconds > 0.0
    assert rows[0].credits_spent <= 8


def test_fixture_planner_exposes_exact_finite_horizon_value_and_cache() -> None:
    env = FixtureEnv(
        complementarity_fixture(),
        initial_budget=2,
        max_probes=2,
        lambda_cost=0.05,
    )
    env.reset(seed=945)

    assert env.optimal_value(lookahead=1) == pytest.approx(0.5)
    assert env.optimal_value(lookahead=2) == pytest.approx(0.9)
    assert env.optimal_action(lookahead=2) == probe_action_index(
        0,
        "metrics.quick",
    )
    assert env.planner_cache_entries > 1
