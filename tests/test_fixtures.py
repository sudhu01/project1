"""Tests for the controlled decision fixtures in plan step 8.3."""

import numpy as np
import pytest

from rca_sim.evidence import EvidenceLedger
from rca_sim.fixtures import (
    FixtureBelief,
    FixtureEnv,
    complementarity_fixture,
    duplicate_evidence_fixture,
    expensive_discriminator_fixture,
    fixture_catalog,
    known_answer_fixture,
    last_credit_evidence_fixture,
    no_affordable_probe_fixture,
    one_perfect_probe_fixture,
    scope_choice_fixture,
    useless_evidence_fixture,
)
from rca_sim.tools import ACTION_SPACE_SIZE, STOP_ACTION_INDEX, probe_action_index


def test_catalog_contains_every_planned_fixture() -> None:
    assert tuple(fixture_catalog()) == (
        "known_answer",
        "one_perfect_probe",
        "useless_evidence",
        "duplicate_evidence",
        "expensive_discriminator",
        "scope_choice",
        "last_credit_evidence",
        "no_affordable_probe",
        "two_probe_complementarity",
    )


@pytest.mark.parametrize("hidden_index", [0, 1])
def test_known_answer_uses_visible_initial_evidence_and_stops(hidden_index: int) -> None:
    env = FixtureEnv(known_answer_fixture())
    observation, _ = env.reset(seed=1, options={"hidden_index": hidden_index})

    assert len(env.evidence_records) == 1
    assert env.posterior[hidden_index] == pytest.approx(1.0)
    assert env.optimal_action() == STOP_ACTION_INDEX
    assert observation["node_features"][hidden_index, 4] == pytest.approx(1.0)


def test_fixture_observation_keeps_padded_shapes_and_zeroes_inactive_entries() -> None:
    env = FixtureEnv(one_perfect_probe_fixture())
    observation, _ = env.reset(seed=2)

    assert env.action_space.n == ACTION_SPACE_SIZE
    assert env.observation_space.contains(observation)
    assert observation["node_mask"].tolist() == [True, True] + [False] * 8
    assert not observation["node_features"][2:].any()
    assert not observation["action_mask"][8:STOP_ACTION_INDEX].any()
    assert not observation["node_features"][:2, 6:8].any()


def test_one_perfect_probe_is_taken_once_then_stop_is_optimal() -> None:
    action = probe_action_index(0, "metrics.quick")
    env = FixtureEnv(one_perfect_probe_fixture())
    env.reset(seed=3, options={"hidden_index": 1})

    assert env.optimal_action() == action
    _, reward, terminated, _, _ = env.step(action)

    assert reward == pytest.approx(-0.05)
    assert not terminated
    assert env.optimal_action() == STOP_ACTION_INDEX


def test_useless_positive_cost_evidence_makes_stop_optimal() -> None:
    env = FixtureEnv(useless_evidence_fixture())
    env.reset(seed=4)

    values = env.decision_values()

    assert values[STOP_ACTION_INDEX] == pytest.approx(0.5)
    assert values[probe_action_index(0, "metrics.quick")] == pytest.approx(0.45)
    assert env.optimal_action() == STOP_ACTION_INDEX


def test_duplicate_template_paths_do_not_update_belief_twice() -> None:
    definition = duplicate_evidence_fixture()
    env = FixtureEnv(definition)
    env.reset(seed=5, options={"hidden_index": 0})
    quick = probe_action_index(0, "metrics.quick")
    detailed = probe_action_index(0, "metrics.detailed")
    ledger = EvidenceLedger()
    belief = FixtureBelief(definition)

    quick_records = env.collect_action(quick)
    belief.update(ledger.add(quick_records))
    before = belief.posterior
    repeated_records = env.collect_action(detailed)

    assert repeated_records == quick_records
    assert ledger.add(repeated_records) == ()
    assert belief.update(repeated_records) == ()
    np.testing.assert_array_equal(belief.posterior, before)


def test_expensive_discriminator_changes_choice_at_break_even() -> None:
    action = probe_action_index(0, "metrics.quick")
    below = FixtureEnv(
        expensive_discriminator_fixture(cost_credits=4),
        lambda_cost=0.10,
    )
    at = FixtureEnv(
        expensive_discriminator_fixture(cost_credits=5),
        lambda_cost=0.10,
    )
    above = FixtureEnv(
        expensive_discriminator_fixture(cost_credits=6),
        lambda_cost=0.10,
    )
    for env in (below, at, above):
        env.reset(seed=6)

    assert below.decision_values()[action] == pytest.approx(0.6)
    assert below.optimal_action() == action
    assert at.decision_values()[action] == pytest.approx(0.5)
    assert at.optimal_action() == STOP_ACTION_INDEX
    assert above.optimal_action() == STOP_ACTION_INDEX


def test_scope_choice_changes_with_budget_and_prior() -> None:
    quick = probe_action_index(0, "metrics.quick")
    detailed = probe_action_index(0, "metrics.detailed")
    roomy = FixtureEnv(scope_choice_fixture(), initial_budget=8)
    tight = FixtureEnv(scope_choice_fixture(), initial_budget=1)
    confident = FixtureEnv(
        scope_choice_fixture(prior=(0.95, 0.05)),
        initial_budget=8,
    )
    for env in (roomy, tight, confident):
        env.reset(seed=7)

    assert roomy.optimal_action() == detailed
    assert tight.optimal_action() == quick
    assert confident.optimal_action() == STOP_ACTION_INDEX


def test_last_credit_probe_scores_the_updated_diagnosis() -> None:
    action = probe_action_index(0, "metrics.quick")
    env = FixtureEnv(last_credit_evidence_fixture(), initial_budget=1)
    env.reset(seed=8, options={"hidden_index": 1})

    _, reward, terminated, truncated, info = env.step(action)

    assert terminated
    assert not truncated
    assert info["diagnosis_correct"] is True
    assert reward == pytest.approx(0.95)
    assert env.remaining_credits == 0


def test_no_affordable_probe_stops_without_negative_budget() -> None:
    env = FixtureEnv(no_affordable_probe_fixture(), initial_budget=1)
    observation, _ = env.reset(seed=9)

    assert observation["action_mask"].sum() == 1
    assert observation["action_mask"][STOP_ACTION_INDEX]
    _, _, terminated, _, _ = env.step(STOP_ACTION_INDEX)

    assert terminated
    assert env.remaining_credits == 1


def test_two_probe_complementarity_requires_joint_lookahead() -> None:
    first = probe_action_index(0, "metrics.quick")
    second = probe_action_index(1, "metrics.quick")
    env = FixtureEnv(complementarity_fixture())
    env.reset(seed=10, options={"hidden_index": 2})

    assert env.optimal_action(lookahead=1) == STOP_ACTION_INDEX
    assert env.optimal_action(lookahead=2) == first

    env.step(first)
    np.testing.assert_allclose(
        env.posterior[[0, 1]].sum(),
        env.posterior[[2, 3]].sum(),
    )
    assert env.optimal_action(lookahead=1) == second
    env.step(second)
    assert env.posterior[[2, 3]].sum() == pytest.approx(1.0)
