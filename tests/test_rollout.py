"""Tests for PPO configuration, rollout storage, and returns."""

from pathlib import Path

import numpy as np
import pytest
import torch

from rca_sim.environment import InvestigationEnv
from rca_sim.model import SmallPolicyNetwork
from rca_sim.ppo import (
    PPOConfig,
    configure_torch_runtime,
    load_ppo_config,
    ppo_losses,
    update_policy,
)
from rca_sim.rollout import RolloutCollector, compute_gae
from rca_sim.tools import ACTION_SPACE_SIZE
from rca_sim.train import CheckpointManager, evaluation_seed_streams, load_checkpoint


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_pilot_ppo_settings_have_documented_update_geometry() -> None:
    config = load_ppo_config(PROJECT_ROOT / "configs" / "ppo_v0.yaml")

    assert config.transitions_per_update == 1_024
    assert config.total_updates == 100
    assert config.minibatches_per_epoch == 4
    assert config.maximum_optimizer_steps_per_update == 16
    assert config.entropy_coefficient == 0.01


def test_ppo_config_rejects_partial_rollouts() -> None:
    with pytest.raises(ValueError, match="complete rollouts"):
        PPOConfig(total_transitions=1_025)


def test_torch_runtime_uses_documented_cpu_thread_count() -> None:
    device = configure_torch_runtime(PPOConfig(torch_num_threads=1))

    assert device == torch.device("cpu")
    assert torch.get_num_threads() == 1


def test_collector_records_complete_on_policy_transitions() -> None:
    torch.manual_seed(101)
    policy = SmallPolicyNetwork()
    policy.train()
    original_parameters = {
        name: parameter.detach().clone()
        for name, parameter in policy.named_parameters()
    }
    collector = RolloutCollector(
        [InvestigationEnv() for _ in range(2)],
        seed=101,
    )

    rollout = collector.collect(policy, rollout_steps_per_env=4)

    assert rollout.size == 8
    assert rollout.action_masks.shape == (8, ACTION_SPACE_SIZE)
    assert rollout.observations["action_mask"].shape == (8, ACTION_SPACE_SIZE)
    assert torch.equal(rollout.action_masks, rollout.observations["action_mask"])
    assert rollout.actions.shape == (8,)
    assert rollout.old_log_probabilities.shape == (8,)
    assert rollout.old_values.shape == (8,)
    assert rollout.rewards.shape == (8,)
    assert rollout.terminations.shape == (8,)
    assert rollout.next_values.shape == (8,)
    assert torch.equal(rollout.episode_boundaries, rollout.terminations)
    assert all(rollout.case_ids)
    assert all(action.kind in {"probe", "stop"} for action in rollout.semantic_actions)
    assert torch.all(rollout.next_values[rollout.terminations] == 0.0)
    assert policy.training
    for name, parameter in policy.named_parameters():
        assert torch.equal(parameter, original_parameters[name])


def test_collector_keeps_unfinished_incident_across_rollout_boundary() -> None:
    torch.manual_seed(202)
    policy = SmallPolicyNetwork()
    with torch.no_grad():
        policy.stop_head[-1].bias.fill_(-100.0)
    environment = InvestigationEnv(initial_budget=8, max_probes=6)
    collector = RolloutCollector([environment], seed=303)

    first = collector.collect(policy, rollout_steps_per_env=1)
    second = collector.collect(policy, rollout_steps_per_env=1)

    assert not bool(first.terminations[0])
    assert first.case_ids[0] == second.case_ids[0]
    assert environment.probes_taken == 2


def test_gae_cuts_two_episodes_and_bootstraps_boundary_by_hand() -> None:
    result = compute_gae(
        rewards=torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]),
        old_values=torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0]),
        next_values=torch.tensor([20.0, 0.0, 40.0, 0.0, 60.0]),
        terminations=torch.tensor([False, True, False, True, False]),
        num_envs=1,
        rollout_steps_per_env=5,
        gamma=1.0,
        gae_lambda=0.5,
    )

    torch.testing.assert_close(
        result.raw_advantages,
        torch.tensor([2.0, -18.0, -5.0, -36.0, 15.0]),
    )
    torch.testing.assert_close(
        result.value_targets,
        torch.tensor([12.0, 2.0, 25.0, 4.0, 65.0]),
    )
    torch.testing.assert_close(result.normalized_advantages.mean(), torch.tensor(0.0))
    torch.testing.assert_close(
        result.normalized_advantages.std(unbiased=False),
        torch.tensor(1.0),
    )


def test_gae_lambda_one_matches_bootstrapped_return_minus_value() -> None:
    result = compute_gae(
        rewards=torch.tensor([1.0, 2.0]),
        old_values=torch.tensor([0.5, 1.0]),
        next_values=torch.tensor([1.0, 3.0]),
        terminations=torch.tensor([False, False]),
        num_envs=1,
        rollout_steps_per_env=2,
        gamma=1.0,
        gae_lambda=1.0,
    )

    torch.testing.assert_close(result.value_targets, torch.tensor([6.0, 5.0]))
    torch.testing.assert_close(
        result.raw_advantages,
        result.value_targets - torch.tensor([0.5, 1.0]),
    )


def test_ppo_loss_matches_clipped_objective_by_hand() -> None:
    config = PPOConfig()
    losses = ppo_losses(
        new_log_probabilities=torch.log(torch.tensor([1.5, 0.5])),
        old_log_probabilities=torch.zeros(2),
        advantages=torch.tensor([2.0, -2.0]),
        current_values=torch.tensor([1.0, 3.0]),
        value_targets=torch.tensor([2.0, 1.0]),
        entropies=torch.tensor([0.4, 0.6]),
        config=config,
    )

    # min(1.5*2, 1.2*2)=2.4 and min(0.5*-2, 0.8*-2)=-1.6.
    torch.testing.assert_close(losses.policy, torch.tensor(-0.4))
    torch.testing.assert_close(losses.value, torch.tensor(2.5))
    torch.testing.assert_close(losses.entropy, torch.tensor(0.5))
    torch.testing.assert_close(losses.total, torch.tensor(0.845))


def test_ppo_update_changes_weights_and_obeys_epoch_limit() -> None:
    torch.manual_seed(404)
    policy = SmallPolicyNetwork()
    collector = RolloutCollector(
        [InvestigationEnv() for _ in range(2)],
        seed=505,
    )
    rollout = collector.collect(policy, rollout_steps_per_env=4)
    rollout.compute_advantages(gamma=1.0, gae_lambda=0.95)
    config = PPOConfig(
        num_envs=2,
        rollout_steps_per_env=4,
        minibatch_size=4,
        update_epochs=2,
        total_transitions=8,
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    before = [parameter.detach().clone() for parameter in policy.parameters()]

    stats = update_policy(policy, optimizer, rollout, config, shuffle_seed=606)

    assert 1 <= stats.epochs_completed <= config.update_epochs
    assert stats.optimizer_steps <= 4
    assert stats.approximate_kl >= 0.0
    assert stats.maximum_unclipped_gradient_norm >= 0.0
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, policy.parameters(), strict=True)
    )


def test_ppo_update_stops_remaining_epochs_after_target_kl() -> None:
    torch.manual_seed(707)
    policy = SmallPolicyNetwork()
    collector = RolloutCollector(
        [InvestigationEnv() for _ in range(2)],
        seed=808,
    )
    rollout = collector.collect(policy, rollout_steps_per_env=4)
    rollout.compute_advantages(gamma=1.0, gae_lambda=0.95)
    config = PPOConfig(
        learning_rate=0.1,
        target_kl=1e-12,
        num_envs=2,
        rollout_steps_per_env=4,
        minibatch_size=4,
        update_epochs=4,
        total_transitions=8,
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)

    stats = update_policy(policy, optimizer, rollout, config, shuffle_seed=909)

    assert stats.stopped_early
    assert stats.epochs_completed == 1
    assert stats.optimizer_steps == 2
    assert stats.approximate_kl > config.target_kl


def test_checkpoint_exactly_restores_next_rollout(tmp_path: Path) -> None:
    torch.manual_seed(111)
    config = PPOConfig(
        num_envs=1,
        rollout_steps_per_env=2,
        minibatch_size=2,
        total_transitions=2,
    )
    policy = SmallPolicyNetwork()
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    collector = RolloutCollector([InvestigationEnv()], seed=222)
    collector.collect(policy, rollout_steps_per_env=2)
    manager = CheckpointManager(tmp_path, lock_path=PROJECT_ROOT / "uv.lock")
    latest, best = manager.save(
        policy=policy,
        optimizer=optimizer,
        config=config,
        completed_transitions=2,
        collector=collector,
        validation_score=0.25,
    )
    expected = collector.collect(policy, rollout_steps_per_env=2)

    restored_policy = SmallPolicyNetwork()
    restored_optimizer = torch.optim.Adam(
        restored_policy.parameters(), lr=config.learning_rate
    )
    restored_collector = RolloutCollector([InvestigationEnv()], seed=999)
    resume = load_checkpoint(
        latest,
        policy=restored_policy,
        optimizer=restored_optimizer,
        config=config,
        collector=restored_collector,
        exact=True,
        lock_path=PROJECT_ROOT / "uv.lock",
    )
    actual = restored_collector.collect(restored_policy, rollout_steps_per_env=2)

    assert best == manager.best_path
    assert latest != best
    assert resume.completed_transitions == 2
    assert resume.best_validation_score == 0.25
    assert resume.continuation == "exact_rollout_boundary"
    assert torch.equal(actual.actions, expected.actions)
    assert torch.equal(actual.rewards, expected.rewards)
    assert actual.case_ids == expected.case_ids
    for key in actual.observations:
        assert torch.equal(actual.observations[key], expected.observations[key])


def test_checkpoint_can_declare_non_identical_continuation(tmp_path: Path) -> None:
    config = PPOConfig(
        num_envs=1,
        rollout_steps_per_env=1,
        minibatch_size=1,
        total_transitions=1,
    )
    policy = SmallPolicyNetwork()
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    collector = RolloutCollector([InvestigationEnv()], seed=333)
    manager = CheckpointManager(tmp_path, lock_path=PROJECT_ROOT / "uv.lock")
    latest, _ = manager.save(
        policy=policy,
        optimizer=optimizer,
        config=config,
        completed_transitions=0,
        collector=collector,
    )

    resume = load_checkpoint(
        latest,
        policy=policy,
        optimizer=optimizer,
        config=config,
        collector=None,
        exact=False,
        lock_path=PROJECT_ROOT / "uv.lock",
    )

    assert resume.continuation == "non_identical_fresh_incidents"


def test_evaluation_rng_streams_do_not_consume_training_rngs() -> None:
    np.random.seed(444)
    torch.manual_seed(555)
    expected_numpy = np.random.random()
    expected_torch = torch.rand(1)
    np.random.seed(444)
    torch.manual_seed(555)

    evaluation_numpy, evaluation_torch = evaluation_seed_streams(666)
    evaluation_numpy.random(10)
    torch.rand(10, generator=evaluation_torch)

    assert np.random.random() == expected_numpy
    assert torch.equal(torch.rand(1), expected_torch)
