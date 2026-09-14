"""Proximal policy optimization configuration and updates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
import yaml

from rca_sim.model import SmallPolicyNetwork
from rca_sim.rollout import RolloutBatch


@dataclass(frozen=True)
class PPOConfig:
    """Validated settings for the first CPU PPO pilot."""

    algorithm: str = "ppo"
    device: str = "cpu"
    seed: int = 101
    gamma: float = 1.0
    gae_lambda: float = 0.95
    learning_rate: float = 3e-4
    clip_ratio: float = 0.2
    value_loss_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_gradient_norm: float = 0.5
    target_kl: float = 0.02
    num_envs: int = 8
    rollout_steps_per_env: int = 128
    minibatch_size: int = 256
    update_epochs: int = 4
    total_transitions: int = 102_400
    evaluate_every_updates: int = 10
    torch_num_threads: int = 1

    def __post_init__(self) -> None:
        if self.algorithm != "ppo":
            raise ValueError("algorithm must be 'ppo'")
        if self.device != "cpu":
            raise ValueError("the ppo_v0 pilot supports only the cpu device")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")
        for name in ("gamma", "gae_lambda"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between zero and one")
        for name in (
            "learning_rate",
            "clip_ratio",
            "value_loss_coefficient",
            "entropy_coefficient",
            "max_gradient_norm",
            "target_kl",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "num_envs",
            "rollout_steps_per_env",
            "minibatch_size",
            "update_epochs",
            "total_transitions",
            "evaluate_every_updates",
            "torch_num_threads",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.transitions_per_update % self.minibatch_size:
            raise ValueError("minibatch_size must divide transitions_per_update")
        if self.total_transitions % self.transitions_per_update:
            raise ValueError("total_transitions must contain complete rollouts")

    @property
    def transitions_per_update(self) -> int:
        return self.num_envs * self.rollout_steps_per_env

    @property
    def total_updates(self) -> int:
        return self.total_transitions // self.transitions_per_update

    @property
    def minibatches_per_epoch(self) -> int:
        return self.transitions_per_update // self.minibatch_size

    @property
    def maximum_optimizer_steps_per_update(self) -> int:
        return self.minibatches_per_epoch * self.update_epochs

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_ppo_config(path: str | Path) -> PPOConfig:
    """Load a PPO YAML file and reject missing or unknown settings."""
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("PPO configuration must be a YAML mapping")
    expected = set(PPOConfig.__dataclass_fields__)
    supplied = set(payload)
    missing = expected - supplied
    unknown = supplied - expected
    if missing:
        raise ValueError(f"PPO configuration is missing: {sorted(missing)}")
    if unknown:
        raise ValueError(f"PPO configuration has unknown keys: {sorted(unknown)}")
    return PPOConfig(**payload)


def configure_torch_runtime(config: PPOConfig) -> torch.device:
    """Apply the documented CPU thread count and return the training device."""
    torch.set_num_threads(config.torch_num_threads)
    return torch.device(config.device)


@dataclass(frozen=True, slots=True)
class PPOLosses:
    """Scalar terms in the PPO objective."""

    total: Tensor
    policy: Tensor
    value: Tensor
    entropy: Tensor
    approximate_kl: Tensor


@dataclass(frozen=True, slots=True)
class PPOUpdateStats:
    """Auditable summary of one rollout update."""

    policy_loss: float
    value_loss: float
    entropy: float
    approximate_kl: float
    optimizer_steps: int
    epochs_completed: int
    stopped_early: bool
    maximum_unclipped_gradient_norm: float


def ppo_losses(
    *,
    new_log_probabilities: Tensor,
    old_log_probabilities: Tensor,
    advantages: Tensor,
    current_values: Tensor,
    value_targets: Tensor,
    entropies: Tensor,
    config: PPOConfig,
) -> PPOLosses:
    """Calculate the clipped policy, critic, and entropy objective."""
    log_ratio = new_log_probabilities - old_log_probabilities
    ratio = log_ratio.exp()
    unclipped = ratio * advantages
    clipped = ratio.clamp(1.0 - config.clip_ratio, 1.0 + config.clip_ratio)
    policy_loss = -torch.minimum(unclipped, clipped * advantages).mean()
    value_loss = torch.square(current_values - value_targets).mean()
    entropy = entropies.mean()
    total = (
        policy_loss
        + config.value_loss_coefficient * value_loss
        - config.entropy_coefficient * entropy
    )
    approximate_kl = ((ratio - 1.0) - log_ratio).mean()
    return PPOLosses(
        total=total,
        policy=policy_loss,
        value=value_loss,
        entropy=entropy,
        approximate_kl=approximate_kl,
    )


def update_policy(
    policy: SmallPolicyNetwork,
    optimizer: torch.optim.Optimizer,
    rollout: RolloutBatch,
    config: PPOConfig,
    *,
    shuffle_seed: int,
) -> PPOUpdateStats:
    """Consume one rollout with PPO minibatches and a rollout-wide KL gate."""
    if rollout.advantages is None or rollout.value_targets is None:
        raise ValueError("compute rollout advantages before the PPO update")
    if rollout.size % config.minibatch_size:
        raise ValueError("minibatch_size must divide the collected rollout")

    device = torch.device(config.device)
    policy.train()
    observations = {
        name: value.to(device) for name, value in rollout.observations.items()
    }
    actions = rollout.actions.to(device)
    old_log_probabilities = rollout.old_log_probabilities.to(device)
    advantages = rollout.advantages.to(device)
    value_targets = rollout.value_targets.to(device)
    generator = torch.Generator(device="cpu").manual_seed(shuffle_seed)

    policies: list[float] = []
    values: list[float] = []
    entropies: list[float] = []
    maximum_gradient_norm = 0.0
    optimizer_steps = 0
    epochs_completed = 0
    stopped_early = False
    approximate_kl = 0.0

    for _ in range(config.update_epochs):
        permutation = torch.randperm(rollout.size, generator=generator)
        for start in range(0, rollout.size, config.minibatch_size):
            indices = permutation[start : start + config.minibatch_size].to(device)
            minibatch_observation = {
                name: value[indices] for name, value in observations.items()
            }
            new_log_probabilities, entropy, current_values = (
                policy.evaluate_actions(minibatch_observation, actions[indices])
            )
            losses = ppo_losses(
                new_log_probabilities=new_log_probabilities,
                old_log_probabilities=old_log_probabilities[indices],
                advantages=advantages[indices],
                current_values=current_values,
                value_targets=value_targets[indices],
                entropies=entropy,
                config=config,
            )
            if not torch.isfinite(losses.total):
                raise FloatingPointError("PPO loss is not finite")
            optimizer.zero_grad(set_to_none=True)
            losses.total.backward()
            gradient_norm = nn.utils.clip_grad_norm_(
                policy.parameters(),
                config.max_gradient_norm,
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("PPO gradient norm is not finite")
            optimizer.step()
            optimizer_steps += 1
            maximum_gradient_norm = max(
                maximum_gradient_norm,
                float(gradient_norm.detach().cpu()),
            )
            policies.append(float(losses.policy.detach().cpu()))
            values.append(float(losses.value.detach().cpu()))
            entropies.append(float(losses.entropy.detach().cpu()))

        epochs_completed += 1
        with torch.no_grad():
            new_log_probabilities, _, _ = policy.evaluate_actions(
                observations,
                actions,
            )
            log_ratio = new_log_probabilities - old_log_probabilities
            ratio = log_ratio.exp()
            approximate_kl = float(
                (((ratio - 1.0) - log_ratio).mean()).cpu()
            )
        if approximate_kl > config.target_kl:
            stopped_early = True
            break

    if not policies:  # pragma: no cover
        raise RuntimeError("PPO update performed no optimizer steps")
    return PPOUpdateStats(
        policy_loss=sum(policies) / len(policies),
        value_loss=sum(values) / len(values),
        entropy=sum(entropies) / len(entropies),
        approximate_kl=approximate_kl,
        optimizer_steps=optimizer_steps,
        epochs_completed=epochs_completed,
        stopped_early=stopped_early,
        maximum_unclipped_gradient_norm=maximum_gradient_norm,
    )
