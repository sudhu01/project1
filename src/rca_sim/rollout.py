"""On-policy trajectory collection and storage."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

from rca_sim.environment import InvestigationEnv
from rca_sim.model import SmallPolicyNetwork
from rca_sim.observation import Observation
from rca_sim.tools import STOP_ACTION_INDEX, decode_probe_action


OBSERVATION_KEYS = (
    "node_features",
    "node_mask",
    "adjacency",
    "global_features",
    "action_mask",
)


@dataclass(frozen=True, slots=True)
class SemanticAction:
    """Auditable meaning of one stored discrete action."""

    kind: str
    service_slot: int | None
    template: str | None


@dataclass(slots=True)
class RolloutBatch:
    """One disposable rollout collected under frozen policy parameters."""

    observations: dict[str, Tensor]
    action_masks: Tensor
    actions: Tensor
    semantic_actions: tuple[SemanticAction, ...]
    old_log_probabilities: Tensor
    old_values: Tensor
    rewards: Tensor
    terminations: Tensor
    next_values: Tensor
    episode_boundaries: Tensor
    case_ids: tuple[str, ...]
    num_envs: int
    rollout_steps_per_env: int
    advantages: Tensor | None = None
    value_targets: Tensor | None = None

    def __post_init__(self) -> None:
        size = self.num_envs * self.rollout_steps_per_env
        if set(self.observations) != set(OBSERVATION_KEYS):
            raise ValueError("rollout observations do not match sim_v0")
        if any(value.shape[0] != size for value in self.observations.values()):
            raise ValueError("every observation tensor must match rollout size")
        for name in (
            "action_masks",
            "actions",
            "old_log_probabilities",
            "old_values",
            "rewards",
            "terminations",
            "next_values",
            "episode_boundaries",
        ):
            if getattr(self, name).shape[0] != size:
                raise ValueError(f"{name} must match rollout size")
        if len(self.semantic_actions) != size or len(self.case_ids) != size:
            raise ValueError("rollout audit fields must match rollout size")

    @property
    def size(self) -> int:
        return self.num_envs * self.rollout_steps_per_env

    def compute_advantages(
        self,
        *,
        gamma: float,
        gae_lambda: float,
        normalization_epsilon: float = 1e-8,
    ) -> "GAEResult":
        """Compute and attach normalized advantages and unscaled value targets."""
        result = compute_gae(
            rewards=self.rewards,
            old_values=self.old_values,
            next_values=self.next_values,
            terminations=self.terminations,
            num_envs=self.num_envs,
            rollout_steps_per_env=self.rollout_steps_per_env,
            gamma=gamma,
            gae_lambda=gae_lambda,
            normalization_epsilon=normalization_epsilon,
        )
        self.advantages = result.normalized_advantages
        self.value_targets = result.value_targets
        return result


@dataclass(frozen=True, slots=True)
class GAEResult:
    """Raw and normalized GAE plus critic regression targets."""

    raw_advantages: Tensor
    normalized_advantages: Tensor
    value_targets: Tensor


def compute_gae(
    *,
    rewards: Tensor,
    old_values: Tensor,
    next_values: Tensor,
    terminations: Tensor,
    num_envs: int,
    rollout_steps_per_env: int,
    gamma: float,
    gae_lambda: float,
    normalization_epsilon: float = 1e-8,
) -> GAEResult:
    """Compute GAE without crossing true episode or rollout boundaries."""
    size = num_envs * rollout_steps_per_env
    tensors = (rewards, old_values, next_values, terminations)
    if any(tensor.ndim != 1 or tensor.shape[0] != size for tensor in tensors):
        raise ValueError("GAE inputs must be flat tensors matching rollout size")
    if not 0.0 <= gamma <= 1.0 or not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gamma and gae_lambda must be between zero and one")
    if normalization_epsilon <= 0.0:
        raise ValueError("normalization_epsilon must be positive")

    rewards_grid = rewards.reshape(rollout_steps_per_env, num_envs)
    values_grid = old_values.reshape(rollout_steps_per_env, num_envs)
    next_values_grid = next_values.reshape(rollout_steps_per_env, num_envs)
    terminal_grid = terminations.reshape(rollout_steps_per_env, num_envs)
    advantages_grid = torch.zeros_like(values_grid)
    advantage_next = torch.zeros(
        num_envs,
        dtype=values_grid.dtype,
        device=values_grid.device,
    )
    for step in range(rollout_steps_per_env - 1, -1, -1):
        continues = (~terminal_grid[step]).to(dtype=values_grid.dtype)
        delta = (
            rewards_grid[step]
            + gamma * next_values_grid[step]
            - values_grid[step]
        )
        advantage_next = (
            delta + gamma * gae_lambda * continues * advantage_next
        )
        advantages_grid[step] = advantage_next

    raw_advantages = advantages_grid.reshape(size)
    value_targets = raw_advantages + old_values
    standard_deviation = raw_advantages.std(unbiased=False)
    normalized = (raw_advantages - raw_advantages.mean()) / (
        standard_deviation + normalization_epsilon
    )
    return GAEResult(
        raw_advantages=raw_advantages,
        normalized_advantages=normalized,
        value_targets=value_targets,
    )


class RolloutCollector:
    """Step logical environments serially after one batched policy decision."""

    def __init__(
        self,
        environments: list[InvestigationEnv],
        *,
        seed: int,
        device: torch.device | str = "cpu",
    ) -> None:
        if not environments:
            raise ValueError("at least one environment is required")
        self.environments = environments
        self.device = torch.device(device)
        seed_sequence = np.random.SeedSequence(seed)
        self._episode_rngs = [
            np.random.default_rng(child)
            for child in seed_sequence.spawn(len(environments))
        ]
        self._episode_numbers = [0] * len(environments)
        self._observations: list[Observation] = []
        self._case_ids: list[str] = []
        for env_index in range(len(environments)):
            observation, info = self._reset_environment(env_index)
            self._observations.append(observation)
            self._case_ids.append(self._case_id(env_index, info))

    def collect(
        self,
        policy: SmallPolicyNetwork,
        *,
        rollout_steps_per_env: int,
    ) -> RolloutBatch:
        """Collect fresh transitions without changing policy parameters."""
        if rollout_steps_per_env <= 0:
            raise ValueError("rollout_steps_per_env must be positive")
        was_training = policy.training
        policy.eval()
        records: list[dict[str, Any]] = []
        try:
            with torch.no_grad():
                for _ in range(rollout_steps_per_env):
                    policy_observation = observations_to_tensors(
                        self._observations,
                        device=self.device,
                    )
                    decision = policy.act(policy_observation)
                    next_observations: list[Observation] = []
                    step_records: list[dict[str, Any]] = []
                    for env_index, environment in enumerate(self.environments):
                        action = int(decision.actions[env_index].item())
                        next_observation, reward, terminated, truncated, info = (
                            environment.step(action)
                        )
                        if truncated:
                            raise RuntimeError(
                                "collector requires true terminations, not truncations"
                            )
                        next_observations.append(next_observation)
                        step_records.append(
                            {
                                "observation": self._observations[env_index],
                                "action_mask": np.array(
                                    self._observations[env_index]["action_mask"],
                                    copy=True,
                                ),
                                "action": action,
                                "semantic_action": _semantic_action(action),
                                "old_log_probability": decision.log_probabilities[
                                    env_index
                                ].cpu(),
                                "old_value": decision.values[env_index].cpu(),
                                "reward": float(reward),
                                "termination": bool(terminated),
                                "episode_boundary": bool(terminated),
                                "case_id": self._case_ids[env_index],
                            }
                        )

                    next_tensors = observations_to_tensors(
                        next_observations,
                        device=self.device,
                    )
                    _, next_values = policy(next_tensors)
                    for env_index, record in enumerate(step_records):
                        terminated = record["termination"]
                        record["next_value"] = (
                            torch.tensor(0.0)
                            if terminated
                            else next_values[env_index].cpu()
                        )
                        records.append(record)
                        if terminated:
                            observation, info = self._reset_environment(env_index)
                            self._observations[env_index] = observation
                            self._case_ids[env_index] = self._case_id(env_index, info)
                        else:
                            self._observations[env_index] = next_observations[env_index]
        finally:
            policy.train(was_training)
        return _build_rollout_batch(
            records,
            num_envs=len(self.environments),
            rollout_steps_per_env=rollout_steps_per_env,
        )

    def state_dict(self) -> dict[str, Any]:
        """Capture active worlds, resource state, ledgers, and collector RNGs."""
        return {
            "version": 1,
            "episode_rng_states": [
                copy.deepcopy(rng.bit_generator.state) for rng in self._episode_rngs
            ],
            "episode_numbers": list(self._episode_numbers),
            "case_ids": list(self._case_ids),
            "environments": [
                environment.checkpoint_state()
                for environment in self.environments
            ],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore an exact rollout-boundary collector state."""
        if state.get("version") != 1:
            raise ValueError("unsupported collector checkpoint version")
        environment_states = state.get("environments")
        rng_states = state.get("episode_rng_states")
        episode_numbers = state.get("episode_numbers")
        case_ids = state.get("case_ids")
        count = len(self.environments)
        if any(
            not isinstance(values, list) or len(values) != count
            for values in (
                environment_states,
                rng_states,
                episode_numbers,
                case_ids,
            )
        ):
            raise ValueError("collector checkpoint does not match environment count")
        for rng, rng_state in zip(self._episode_rngs, rng_states, strict=True):
            rng.bit_generator.state = copy.deepcopy(rng_state)
        for environment, environment_state in zip(
            self.environments,
            environment_states,
            strict=True,
        ):
            environment.restore_checkpoint_state(environment_state)
        self._episode_numbers = [int(value) for value in episode_numbers]
        self._case_ids = [str(value) for value in case_ids]
        self._observations = [
            environment.current_observation for environment in self.environments
        ]
        if any(observation is None for observation in self._observations):
            raise RuntimeError("restored collector contains an uninitialized environment")

    def _reset_environment(
        self,
        env_index: int,
    ) -> tuple[Observation, dict[str, Any]]:
        incident_seed = int(
            self._episode_rngs[env_index].integers(0, np.iinfo(np.int32).max)
        )
        self._episode_numbers[env_index] += 1
        return self.environments[env_index].reset(seed=incident_seed)

    def _case_id(self, env_index: int, info: dict[str, Any]) -> str:
        return str(
            info.get(
                "case_id",
                f"generated-env-{env_index}-episode-{self._episode_numbers[env_index]}",
            )
        )


def observations_to_tensors(
    observations: list[Observation],
    *,
    device: torch.device | str = "cpu",
) -> dict[str, Tensor]:
    """Convert only public sim_v0 observation fields into a policy batch."""
    if not observations:
        raise ValueError("at least one observation is required")
    result: dict[str, Tensor] = {}
    for key in OBSERVATION_KEYS:
        array = np.stack([observation[key] for observation in observations])
        tensor = torch.from_numpy(array)
        if key in ("node_mask", "action_mask"):
            tensor = tensor.to(dtype=torch.bool)
        else:
            tensor = tensor.to(dtype=torch.float32)
        result[key] = tensor.to(device)
    return result


def _semantic_action(action: int) -> SemanticAction:
    if action == STOP_ACTION_INDEX:
        return SemanticAction(kind="stop", service_slot=None, template=None)
    service_slot, template = decode_probe_action(action)
    return SemanticAction(
        kind="probe",
        service_slot=service_slot,
        template=template.name,
    )


def _build_rollout_batch(
    records: list[dict[str, Any]],
    *,
    num_envs: int,
    rollout_steps_per_env: int,
) -> RolloutBatch:
    observations = {
        key: torch.from_numpy(
            np.stack([record["observation"][key] for record in records])
        )
        for key in OBSERVATION_KEYS
    }
    observations = {
        key: value.to(
            dtype=torch.bool
            if key in ("node_mask", "action_mask")
            else torch.float32
        )
        for key, value in observations.items()
    }
    return RolloutBatch(
        observations=observations,
        action_masks=torch.from_numpy(
            np.stack([record["action_mask"] for record in records])
        ).to(dtype=torch.bool),
        actions=torch.tensor([record["action"] for record in records]),
        semantic_actions=tuple(record["semantic_action"] for record in records),
        old_log_probabilities=torch.stack(
            [record["old_log_probability"] for record in records]
        ),
        old_values=torch.stack([record["old_value"] for record in records]),
        rewards=torch.tensor([record["reward"] for record in records]),
        terminations=torch.tensor(
            [record["termination"] for record in records], dtype=torch.bool
        ),
        next_values=torch.stack([record["next_value"] for record in records]),
        episode_boundaries=torch.tensor(
            [record["episode_boundary"] for record in records], dtype=torch.bool
        ),
        case_ids=tuple(record["case_id"] for record in records),
        num_envs=num_envs,
        rollout_steps_per_env=rollout_steps_per_env,
    )
