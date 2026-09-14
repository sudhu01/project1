"""Training checkpoint and resume support for the small simulator."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rca_sim.model import SmallPolicyNetwork
from rca_sim.observation import OBSERVATION_SCHEMA_VERSION, OBSERVATION_SHAPES
from rca_sim.ppo import PPOConfig
from rca_sim.rollout import RolloutCollector
from rca_sim.world import GENERATOR_VERSION


CHECKPOINT_VERSION = 1


@dataclass(frozen=True, slots=True)
class ResumeResult:
    """Training counters and the declared continuation semantics."""

    completed_transitions: int
    best_validation_score: float | None
    continuation: str


class CheckpointManager:
    """Write independent latest and best-validation training checkpoints."""

    def __init__(self, directory: str | Path, *, lock_path: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock_path = Path(lock_path)
        self.best_validation_score: float | None = None

    @property
    def latest_path(self) -> Path:
        return self.directory / "latest.pt"

    @property
    def best_path(self) -> Path:
        return self.directory / "best-validation.pt"

    def save(
        self,
        *,
        policy: SmallPolicyNetwork,
        optimizer: torch.optim.Optimizer,
        config: PPOConfig,
        completed_transitions: int,
        collector: RolloutCollector,
        validation_score: float | None = None,
    ) -> tuple[Path, Path | None]:
        """Save latest and update best only when validation improves."""
        if completed_transitions < 0:
            raise ValueError("completed_transitions must be nonnegative")
        if validation_score is not None and not np.isfinite(validation_score):
            raise ValueError("validation_score must be finite")
        candidate_best = self.best_validation_score
        is_best = validation_score is not None and (
            candidate_best is None or validation_score > candidate_best
        )
        if is_best:
            candidate_best = float(validation_score)
        payload = _checkpoint_payload(
            policy=policy,
            optimizer=optimizer,
            config=config,
            completed_transitions=completed_transitions,
            collector=collector,
            best_validation_score=candidate_best,
            lock_path=self.lock_path,
        )
        torch.save(payload, self.latest_path)
        best_path = None
        if is_best:
            torch.save(payload, self.best_path)
            self.best_validation_score = candidate_best
            best_path = self.best_path
        return self.latest_path, best_path


def load_checkpoint(
    path: str | Path,
    *,
    policy: SmallPolicyNetwork,
    optimizer: torch.optim.Optimizer,
    config: PPOConfig,
    collector: RolloutCollector | None,
    exact: bool,
    lock_path: str | Path,
) -> ResumeResult:
    """Restore a checkpoint exactly or declare a fresh-incident continuation."""
    payload = torch.load(Path(path), map_location=config.device, weights_only=False)
    if payload.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError("unsupported checkpoint version")
    if payload.get("config") != config.as_dict():
        raise ValueError("checkpoint PPO configuration does not match")
    expected_lock_hash = _file_sha256(Path(lock_path))
    if payload.get("dependency_lock_sha256") != expected_lock_hash:
        raise ValueError("dependency lock hash does not match checkpoint")
    if payload.get("feature_schema") != _feature_schema():
        raise ValueError("feature schema does not match checkpoint")
    if payload.get("generator_version") != GENERATOR_VERSION:
        raise ValueError("generator version does not match checkpoint")

    policy.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    _restore_rng_states(payload["rng_states"])
    if exact:
        if collector is None:
            raise ValueError("exact resume requires a rollout collector")
        collector.load_state_dict(payload["collector_state"])
        continuation = "exact_rollout_boundary"
    else:
        continuation = "non_identical_fresh_incidents"
    return ResumeResult(
        completed_transitions=int(payload["completed_transitions"]),
        best_validation_score=payload.get("best_validation_score"),
        continuation=continuation,
    )


def evaluation_seed_streams(seed: int) -> tuple[np.random.Generator, torch.Generator]:
    """Create evaluation-only RNG streams independent of training state."""
    sequence = np.random.SeedSequence([seed, 0x4556414C])
    numpy_child, torch_child = sequence.spawn(2)
    numpy_rng = np.random.default_rng(numpy_child)
    torch_seed = int(torch_child.generate_state(1, dtype=np.uint64)[0])
    torch_rng = torch.Generator(device="cpu").manual_seed(torch_seed)
    return numpy_rng, torch_rng


def _checkpoint_payload(
    *,
    policy: SmallPolicyNetwork,
    optimizer: torch.optim.Optimizer,
    config: PPOConfig,
    completed_transitions: int,
    collector: RolloutCollector,
    best_validation_score: float | None,
    lock_path: Path,
) -> dict[str, Any]:
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "model_state": policy.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": config.as_dict(),
        "completed_transitions": completed_transitions,
        "rng_states": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
        },
        "feature_schema": _feature_schema(),
        "generator_version": GENERATOR_VERSION,
        "dependency_lock_sha256": _file_sha256(lock_path),
        "collector_state": collector.state_dict(),
        "best_validation_score": best_validation_score,
        "continuation": "exact_rollout_boundary",
    }


def _restore_rng_states(states: dict[str, Any]) -> None:
    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.set_rng_state(states["torch_cpu"])


def _feature_schema() -> dict[str, Any]:
    return {
        "version": OBSERVATION_SCHEMA_VERSION,
        "shapes": dict(OBSERVATION_SHAPES),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
