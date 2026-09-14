"""Tests for the investigation environment."""

import json

import gymnasium as gym
import numpy as np
import pytest

from rca_sim.belief import ExactBeliefEstimator
from rca_sim.environment import InvestigationEnv
from rca_sim.tools import ProbeTool, collect_probe_evidence, probe_action_index
from rca_sim.world import generate_incident_world


def _assert_observations_equal(
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
) -> None:
    assert left.keys() == right.keys()
    for key in left:
        np.testing.assert_array_equal(left[key], right[key])


def _world_with_initial_diagnosis(*, correct: bool):
    for incident_seed in range(100):
        world = generate_incident_world(n_services=8, incident_seed=incident_seed)
        if (world.hypothesis.cause_service == 0) is correct:
            return world
    raise AssertionError("could not find a suitable deterministic incident")


def test_reset_initializes_the_declared_episode_state_and_public_info() -> None:
    env = InvestigationEnv()

    observation, info = env.reset(seed=20260914)

    assert env.observation_space.contains(observation)
    assert env.action_space.n == 41
    assert env.remaining_credits == 8
    assert env.probes_taken == 0
    assert env.episode_return == 0.0
    assert env.evidence_records == ()
    assert env.executed_probe_actions == ()
    assert env.last_pre_action_observation is None
    assert observation["action_mask"][:32].all()
    assert not observation["action_mask"][32:40].any()
    assert observation["action_mask"][40]
    assert info == {
        "schema_version": "sim_v0",
        "generator_version": "sim_v0",
        "n_services": 8,
        "entry_service": info["entry_service"],
        "initial_budget": 8,
        "remaining_credits": 8,
        "credits_spent": 0,
        "max_probes": 6,
        "probes_taken": 0,
        "predicted_service": 0,
        "predicted_fault": "cpu",
        "termination_reason": None,
    }
    assert not {"incident_seed", "cause_service", "workload"} & info.keys()


def test_reset_seed_replays_world_independent_of_prior_episode_actions() -> None:
    first_env = InvestigationEnv()
    first, _ = first_env.reset(seed=17)
    first_env.step(probe_action_index(0, "metrics.quick"))
    replayed, _ = first_env.reset(seed=17)

    second_env = InvestigationEnv()
    expected, _ = second_env.reset(seed=17)

    _assert_observations_equal(first, replayed)
    _assert_observations_equal(replayed, expected)


def test_reset_clears_evidence_history_budget_and_saved_transition() -> None:
    env = InvestigationEnv()
    env.reset(seed=8)
    env.step(probe_action_index(0, "metrics.quick"))

    observation, _ = env.reset(seed=9)

    assert env.evidence_records == ()
    assert env.executed_probe_actions == ()
    assert env.remaining_credits == 8
    assert env.probes_taken == 0
    assert env.last_pre_action_observation is None
    assert not observation["node_features"][:, 8:35].any()


def test_reset_loads_exact_case_mapping_or_json_manifest(tmp_path) -> None:
    descriptor = {
        "case_id": "validation-007",
        "generator_version": "sim_v0",
        "incident_seed": 777,
        "n_services": 6,
        "extra_edge_probability": 0.3,
    }
    case_path = tmp_path / "case_007.json"
    case_path.write_text(json.dumps(descriptor), encoding="utf-8")

    mapping_env = InvestigationEnv()
    mapping_observation, mapping_info = mapping_env.reset(
        seed=1,
        options={"case": descriptor},
    )
    file_env = InvestigationEnv()
    file_observation, file_info = file_env.reset(
        seed=999,
        options={"case_path": case_path},
    )

    _assert_observations_equal(mapping_observation, file_observation)
    assert mapping_info["case_id"] == "validation-007"
    assert file_info["case_id"] == "validation-007"
    assert mapping_info["n_services"] == 6


def test_probe_charges_cost_merges_evidence_updates_belief_and_history() -> None:
    world = generate_incident_world(n_services=3, incident_seed=41)
    env = InvestigationEnv(n_services=3)
    before, _ = env.reset(options={"case": world})
    action = probe_action_index(1, "metrics.detailed")
    expected_records = collect_probe_evidence(world, "metrics.detailed", 1)
    expected_belief = ExactBeliefEstimator(world.graph)
    expected_belief.update(expected_records)

    after, reward, terminated, truncated, info = env.step(action)

    assert reward == pytest.approx(-0.10)
    assert not terminated
    assert not truncated
    assert env.remaining_credits == 6
    assert env.probes_taken == 1
    assert env.executed_probe_actions == (action,)
    assert env.evidence_records == expected_records
    np.testing.assert_allclose(
        after["node_features"][:3, 4],
        expected_belief.belief_service,
    )
    assert after["node_features"][1, 37:41].tolist() == [0, 1, 0, 0]
    assert after["node_features"][1, 41] == pytest.approx(0.25)
    assert info["new_evidence_ids"] == tuple(
        record.evidence_id for record in expected_records
    )
    assert info["step_cost"] == 2
    assert info["diagnosis_correct"] is None
    saved = env.last_pre_action_observation
    assert saved is not None
    _assert_observations_equal(saved, before)


def test_quick_then_detailed_updates_only_from_unseen_evidence() -> None:
    world = generate_incident_world(n_services=3, incident_seed=42)
    quick_then_detailed = InvestigationEnv(n_services=3)
    quick_then_detailed.reset(options={"case": world})
    quick_then_detailed.step(probe_action_index(1, "metrics.quick"))
    combined, _, _, _, combined_info = quick_then_detailed.step(
        probe_action_index(1, "metrics.detailed")
    )

    detailed_only = InvestigationEnv(n_services=3)
    detailed_only.reset(options={"case": world})
    direct, _, _, _, _ = detailed_only.step(
        probe_action_index(1, "metrics.detailed")
    )

    assert len(quick_then_detailed.evidence_records) == 6
    assert len(combined_info["new_evidence_ids"]) == 3
    assert all(item.endswith("/1") for item in combined_info["new_evidence_ids"])
    np.testing.assert_allclose(
        combined["node_features"][:3, 4:8],
        direct["node_features"][:3, 4:8],
    )
    assert combined["node_features"][1, 37:41].tolist() == [1, 1, 0, 0]
    assert combined["node_features"][1, 41] == pytest.approx(3 / 8)
    assert quick_then_detailed.remaining_credits == 5
    assert detailed_only.remaining_credits == 6


def test_transition_observations_are_fresh_copies() -> None:
    env = InvestigationEnv()
    before, _ = env.reset(seed=22)
    before["node_features"][0, 0] = 99
    after, _, _, _, _ = env.step(probe_action_index(0, "metrics.quick"))
    saved = env.last_pre_action_observation
    assert saved is not None

    assert saved["node_features"][0, 0] in (0, 1)
    saved["node_features"][0, 0] = 77
    assert env.last_pre_action_observation["node_features"][0, 0] in (0, 1)
    assert after["node_features"][0, 0] in (0, 1)


def test_budget_exhaustion_terminates_and_scores_updated_diagnosis() -> None:
    env = InvestigationEnv(initial_budget=1)
    env.reset(seed=31)

    _, reward, terminated, truncated, info = env.step(
        probe_action_index(0, "metrics.quick")
    )

    assert terminated
    assert not truncated
    assert env.termination_reason == "budget"
    assert info["termination_reason"] == "budget"
    assert reward == pytest.approx(-0.05 + float(info["diagnosis_correct"]))
    assert env.remaining_credits == 0


def test_horizon_exhaustion_terminates_after_the_completed_probe() -> None:
    env = InvestigationEnv(max_probes=1)
    env.reset(seed=32)

    observation, reward, terminated, truncated, info = env.step(
        probe_action_index(0, "metrics.quick")
    )

    assert terminated
    assert not truncated
    assert info["termination_reason"] == "horizon"
    assert info["probes_taken"] == 1
    assert reward == pytest.approx(-0.05 + float(info["diagnosis_correct"]))
    assert not observation["action_mask"].any()


def test_no_feasible_probe_auto_finalizes_without_an_extra_stop_step() -> None:
    env = InvestigationEnv(
        n_services=1,
        initial_budget=3,
        available_tools={0: frozenset({ProbeTool.LOGS})},
    )
    env.reset(seed=33)

    _, reward, terminated, truncated, info = env.step(
        probe_action_index(0, "logs.quick")
    )

    assert terminated
    assert not truncated
    assert env.remaining_credits == 1
    assert env.probes_taken == 1
    assert info["termination_reason"] == "no_probe_available"
    assert reward == pytest.approx(-0.10 + float(info["diagnosis_correct"]))


def test_terminal_score_uses_the_posterior_after_the_final_probe() -> None:
    selected = None
    action = probe_action_index(0, "metrics.quick")
    for incident_seed in range(1, 500):
        world = generate_incident_world(n_services=2, incident_seed=incident_seed)
        estimator = ExactBeliefEstimator(world.graph)
        before = estimator.predicted_service
        estimator.update(collect_probe_evidence(world, "metrics.quick", 0))
        if before != world.hypothesis.cause_service == estimator.predicted_service:
            selected = world
            break
    assert selected is not None

    env = InvestigationEnv(n_services=2, max_probes=1)
    env.reset(options={"case": selected})
    _, reward, terminated, _, info = env.step(action)

    assert terminated
    assert info["diagnosis_correct"] is True
    assert reward == pytest.approx(0.95)


def test_invalid_probe_is_rejected_before_any_episode_state_changes() -> None:
    env = InvestigationEnv(initial_budget=1)
    before, _ = env.reset(seed=44)
    expensive = probe_action_index(0, "logs.detailed")

    with pytest.raises(ValueError, match="invalid"):
        env.step(expensive)

    assert env.remaining_credits == 1
    assert env.probes_taken == 0
    assert env.evidence_records == ()
    assert env.executed_probe_actions == ()
    assert env.last_pre_action_observation is None
    replayed, _ = env.reset(seed=44)
    _assert_observations_equal(before, replayed)


def test_step_requires_reset_and_rejects_steps_after_auto_termination() -> None:
    env = InvestigationEnv(max_probes=1)
    with pytest.raises(RuntimeError, match="reset"):
        env.step(0)

    env.reset(seed=55)
    env.step(0)
    with pytest.raises(RuntimeError, match="terminated"):
        env.step(0)


def test_collection_cutoff_preserves_live_observation_and_episode_state() -> None:
    env = InvestigationEnv()
    env.reset(seed=56)
    returned, reward, terminated, truncated, _ = env.step(
        probe_action_index(0, "metrics.quick")
    )

    cutoff_observation = env.current_observation

    assert cutoff_observation is not None
    _assert_observations_equal(cutoff_observation, returned)
    assert reward == pytest.approx(-0.05)
    assert not terminated
    assert not truncated
    assert not env.terminated
    assert env.termination_reason is None
    assert env.probes_taken == 1
    assert env.remaining_credits == 7

    cutoff_observation["node_features"][0, 0] = 99
    assert env.current_observation["node_features"][0, 0] in (0, 1)

    _, stop_reward, stop_terminated, stop_truncated, stop_info = env.step(40)
    assert stop_terminated
    assert not stop_truncated
    assert stop_info["termination_reason"] == "stop"
    assert env.episode_return == pytest.approx(reward + stop_reward)


def test_external_time_limit_reports_truncation_without_true_termination() -> None:
    base_env = InvestigationEnv()
    wrapped = gym.wrappers.TimeLimit(base_env, max_episode_steps=1)
    wrapped.reset(seed=57)

    observation, reward, terminated, truncated, info = wrapped.step(
        probe_action_index(0, "metrics.quick")
    )

    assert reward == pytest.approx(-0.05)
    assert not terminated
    assert truncated
    assert info["termination_reason"] is None
    assert not base_env.terminated
    assert base_env.termination_reason is None
    assert base_env.current_observation is not None
    _assert_observations_equal(base_env.current_observation, observation)


def test_second_step_after_stop_cannot_score_the_episode_twice() -> None:
    env = InvestigationEnv()
    env.reset(seed=58)
    env.step(40)
    scored_return = env.episode_return

    with pytest.raises(RuntimeError, match="terminated"):
        env.step(40)

    assert env.episode_return == scored_return
    assert env.termination_reason == "stop"


@pytest.mark.parametrize(("correct", "expected_reward"), [(True, 1.0), (False, 0.0)])
def test_immediate_stop_scores_current_service_diagnosis_without_cost(
    correct: bool,
    expected_reward: float,
) -> None:
    env = InvestigationEnv()
    before, _ = env.reset(
        options={"case": _world_with_initial_diagnosis(correct=correct)}
    )

    after, reward, terminated, truncated, info = env.step(40)

    assert terminated
    assert not truncated
    assert reward == expected_reward
    assert env.episode_return == expected_reward
    assert env.termination_reason == "stop"
    assert env.probes_taken == 0
    assert env.remaining_credits == 8
    assert env.evidence_records == ()
    assert env.executed_probe_actions == ()
    assert info["action_type"] == "stop"
    assert info["step_cost"] == 0
    assert info["new_evidence_ids"] == ()
    assert info["diagnosis_correct"] is correct
    assert info["termination_reason"] == "stop"
    assert not after["action_mask"].any()
    saved = env.last_pre_action_observation
    assert saved is not None
    _assert_observations_equal(saved, before)


def test_episode_return_is_terminal_utility_minus_total_credit_cost() -> None:
    env = InvestigationEnv()
    env.reset(seed=67)

    _, quick_reward, quick_done, _, quick_info = env.step(
        probe_action_index(1, "metrics.quick")
    )
    _, detailed_reward, detailed_done, _, detailed_info = env.step(
        probe_action_index(1, "metrics.detailed")
    )
    _, stop_reward, stop_done, _, stop_info = env.step(40)

    assert quick_reward == pytest.approx(-0.05)
    assert detailed_reward == pytest.approx(-0.10)
    assert quick_info["diagnosis_correct"] is None
    assert detailed_info["diagnosis_correct"] is None
    assert not quick_done
    assert not detailed_done
    assert stop_done
    expected = float(stop_info["diagnosis_correct"]) - 0.05 * 3
    assert quick_reward + detailed_reward + stop_reward == pytest.approx(expected)
    assert env.episode_return == pytest.approx(expected)
    assert stop_info["episode_return"] == pytest.approx(expected)
    assert -0.40 <= env.episode_return <= 1.0


def test_full_budget_episode_return_reaches_documented_default_bounds() -> None:
    env = InvestigationEnv()
    env.reset(seed=68)

    rewards = [env.step(probe_action_index(0, "logs.detailed"))[1]]
    rewards.append(env.step(probe_action_index(1, "metrics.detailed"))[1])
    _, final_reward, terminated, _, info = env.step(
        probe_action_index(1, "logs.quick")
    )
    rewards.append(final_reward)

    assert terminated
    assert info["termination_reason"] == "budget"
    assert info["credits_spent"] == 8
    expected = float(info["diagnosis_correct"]) - 0.40
    assert sum(rewards) == pytest.approx(expected)
    assert env.episode_return == pytest.approx(expected)
    assert env.episode_return == pytest.approx(
        -0.40
    ) or env.episode_return == pytest.approx(0.60)


def test_correct_current_prediction_does_not_end_a_probe_transition() -> None:
    world = _world_with_initial_diagnosis(correct=True)
    env = InvestigationEnv()
    env.reset(options={"case": world})

    _, reward, terminated, _, info = env.step(
        probe_action_index(0, "metrics.quick")
    )

    assert not terminated
    assert reward == pytest.approx(-0.05)
    assert info["diagnosis_correct"] is None
    assert info["termination_reason"] is None
