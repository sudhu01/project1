"""Training checkpoint and resume support for the small simulator."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import numpy as np
import torch
import yaml

from rca_sim.config import load_simulator_config
from rca_sim.environment import InvestigationEnv
from rca_sim.fixtures import FixtureEnv, fixture_catalog
from rca_sim.model import SmallPolicyNetwork
from rca_sim.observation import OBSERVATION_SCHEMA_VERSION, OBSERVATION_SHAPES
from rca_sim.ppo import PPOConfig, configure_torch_runtime, load_ppo_config, update_policy
from rca_sim.rollout import RolloutCollector, observations_to_tensors
from rca_sim.world import GENERATOR_VERSION


CHECKPOINT_VERSION = 1
CHECKPOINT_RETURN_TIE_TOLERANCE = 0.005


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


def run_training(
    *,
    sim_config_path: Path,
    ppo_config_path: Path,
    seed: int,
    total_transitions: int,
    run_directory: Path,
    fixture_name: str | None = None,
    train_cases: Path | None = None,
    validation_cases: Path | None = None,
    lambda_cost_override: float | None = None,
    budget_credits_override: int | None = None,
) -> dict[str, Any]:
    """Train one diagnostic PPO run and write auditable learning artifacts."""
    base_config = load_ppo_config(ppo_config_path)
    config = replace(base_config, seed=seed, total_transitions=total_transitions)
    device = configure_torch_runtime(config)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if fixture_name is not None:
        try:
            definition = fixture_catalog()[fixture_name]
        except KeyError as error:
            raise ValueError(f"unknown fixture: {fixture_name}") from error
        if fixture_name == "one_perfect_probe":
            catalog = fixture_catalog()
            curriculum = (
                catalog["known_answer"],
                catalog["one_perfect_probe"],
                catalog["useless_evidence"],
            )
        else:
            curriculum = (definition,)
        environments = [
            FixtureEnv(
                curriculum[index % len(curriculum)],
                initial_budget=8,
                max_probes=6,
                lambda_cost=0.05,
            )
            for index in range(config.num_envs)
        ]
        case_sources = None
        evaluation = lambda policy: _evaluate_fixture(policy, definition, seed + 70_000)
    else:
        sim_config = load_simulator_config(sim_config_path)
        environment_kwargs = sim_config.environment_kwargs()
        if lambda_cost_override is not None:
            if not 0.0 <= lambda_cost_override <= 0.20:
                raise ValueError("lambda cost override must be between 0 and 0.20")
            environment_kwargs["lambda_cost"] = float(lambda_cost_override)
        if budget_credits_override is not None:
            if not 1 <= budget_credits_override <= 16:
                raise ValueError("budget override must be between 1 and 16")
            environment_kwargs["initial_budget"] = int(budget_credits_override)
        environments = [
            InvestigationEnv(**environment_kwargs)
            for _ in range(config.num_envs)
        ]
        case_sources = _case_paths(train_cases) if train_cases is not None else None
        evaluation_cases = (
            _case_paths(validation_cases)
            if validation_cases is not None
            else case_sources or _streaming_evaluation_cases(seed, 64)
        )
        evaluation = lambda policy: _evaluate_incidents(
            policy, environment_kwargs, evaluation_cases
        )

    collector = RolloutCollector(
        environments,
        seed=seed + 10_000,
        device=device,
        case_sources=case_sources,
        permute_case_services=case_sources is not None,
    )
    policy = SmallPolicyNetwork().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    run_directory.mkdir(parents=True, exist_ok=True)
    (run_directory / "resolved_ppo_config.json").write_text(
        json.dumps(config.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    resolved_sim = yaml.safe_load(sim_config_path.read_text(encoding="utf-8"))
    if fixture_name is None:
        resolved_sim["lambda_cost"] = environment_kwargs["lambda_cost"]
        resolved_sim["budget_credits"] = environment_kwargs["initial_budget"]
    (run_directory / "resolved_sim_config.yaml").write_text(
        yaml.safe_dump(resolved_sim, sort_keys=False), encoding="utf-8"
    )

    started = perf_counter()
    metrics: list[dict[str, Any]] = []
    initial_eval = evaluation(policy)
    best_score = float(initial_eval["mean_return"])
    best_cost = float(initial_eval.get("mean_credits", 0.0))
    validation_metrics: list[dict[str, Any]] = [
        {"update": 0, "completed_transitions": 0, **initial_eval}
    ]
    _save_run_checkpoint(
        run_directory / "best.pt", policy, optimizer, config, 0, best_score
    )
    for update in range(1, config.total_updates + 1):
        rollout = collector.collect(
            policy, rollout_steps_per_env=config.rollout_steps_per_env
        )
        gae = rollout.compute_advantages(gamma=config.gamma, gae_lambda=config.gae_lambda)
        stats = update_policy(
            policy, optimizer, rollout, config, shuffle_seed=seed + update
        )
        completed = update * config.transitions_per_update
        row: dict[str, Any] = {
            "update": update,
            "completed_transitions": completed,
            "mean_batch_reward": float(rollout.rewards.mean()),
            "raw_advantage_mean": float(gae.raw_advantages.mean()),
            **asdict(stats),
        }
        if update % config.evaluate_every_updates == 0 or update == config.total_updates:
            result = evaluation(policy)
            validation_metrics.append(
                {"update": update, "completed_transitions": completed, **result}
            )
            row.update({f"evaluation_{key}": value for key, value in result.items()})
            score = float(result["mean_return"])
            cost = float(result.get("mean_credits", 0.0))
            if score > best_score + CHECKPOINT_RETURN_TIE_TOLERANCE or (
                abs(score - best_score) <= CHECKPOINT_RETURN_TIE_TOLERANCE
                and cost < best_cost
            ):
                best_score = score
                best_cost = cost
                _save_run_checkpoint(
                    run_directory / "best.pt", policy, optimizer, config, completed, score
                )
        metrics.append(row)
    elapsed = perf_counter() - started
    _save_run_checkpoint(
        run_directory / "latest.pt", policy, optimizer, config,
        config.total_transitions, best_score,
    )
    _write_metrics(run_directory / "training_metrics.csv", metrics)
    _write_metrics(run_directory / "validation_metrics.csv", validation_metrics)
    final_eval = evaluation(policy)
    result = {
        "run_id": run_directory.name,
        "fixture": fixture_name,
        "train_cases": None if train_cases is None else str(train_cases),
        "validation_cases": None if validation_cases is None else str(validation_cases),
        "seed": seed,
        "total_transitions": config.total_transitions,
        "total_updates": config.total_updates,
        "initial_evaluation": initial_eval,
        "final_evaluation": final_eval,
        "best_evaluation_return": best_score,
        "best_evaluation_mean_credits": best_cost,
        "checkpoint_return_tie_tolerance": CHECKPOINT_RETURN_TIE_TOLERANCE,
        "lambda_cost": None if fixture_name is not None else environment_kwargs["lambda_cost"],
        "budget_credits": None if fixture_name is not None else environment_kwargs["initial_budget"],
        "max_probes": None if fixture_name is not None else environment_kwargs["max_probes"],
        "wall_seconds": elapsed,
        "transitions_per_second": config.total_transitions / elapsed,
    }
    (run_directory / "metadata.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for environment in environments:
        environment.close()
    return result


def _case_paths(path: Path) -> tuple[Path, ...]:
    cases = tuple(sorted(path.glob("case_*.json")))
    if not cases:
        raise ValueError(f"training case directory contains no cases: {path}")
    return cases


def _streaming_evaluation_cases(seed: int, count: int) -> tuple[dict[str, object], ...]:
    sequence = np.random.SeedSequence([seed, 0x4556414C])
    seeds = sequence.generate_state(count, dtype=np.uint64)
    return tuple(
        {
            "case_id": f"stream_eval_{index:04d}",
            "incident_seed": int(value),
            "n_services": 8,
            "extra_edge_probability": 0.15,
        }
        for index, value in enumerate(seeds)
    )


def _evaluate_incidents(
    policy: SmallPolicyNetwork,
    environment_kwargs: dict[str, int | float],
    cases: Sequence[Path | dict[str, object]],
) -> dict[str, float]:
    environment = InvestigationEnv(**environment_kwargs)
    returns: list[float] = []
    correct: list[bool] = []
    credits: list[int] = []
    was_training = policy.training
    policy.eval()
    try:
        with torch.no_grad():
            for case in cases:
                observation, _ = environment.reset(options={"case": case})
                while not environment.terminated:
                    tensors = observations_to_tensors([observation])
                    action = int(policy.act(tensors, deterministic=True).actions[0])
                    observation, _, terminated, _, info = environment.step(action)
                    if terminated:
                        returns.append(float(info["episode_return"]))
                        correct.append(bool(info["diagnosis_correct"]))
                        credits.append(int(info["credits_spent"]))
    finally:
        policy.train(was_training)
        environment.close()
    return {
        "mean_return": float(np.mean(returns)),
        "accuracy": float(np.mean(correct)),
        "mean_credits": float(np.mean(credits)),
    }


def _evaluate_fixture(
    policy: SmallPolicyNetwork, definition: Any, seed: int, episodes: int = 256
) -> dict[str, float]:
    environment = FixtureEnv(definition, initial_budget=8, max_probes=6, lambda_cost=0.05)
    returns: list[float] = []
    correct: list[bool] = []
    first_optimal: list[bool] = []
    was_training = policy.training
    policy.eval()
    try:
        with torch.no_grad():
            for episode in range(episodes):
                observation, _ = environment.reset(seed=seed + episode)
                optimal_actions = {
                    action for action, value in environment.decision_values(lookahead=6).items()
                    if np.isclose(value, environment.optimal_value(lookahead=6))
                }
                first = True
                while True:
                    tensors = observations_to_tensors([observation])
                    action = int(policy.act(tensors, deterministic=True).actions[0])
                    if first:
                        first_optimal.append(action in optimal_actions)
                        first = False
                    observation, _, terminated, _, info = environment.step(action)
                    if terminated:
                        returns.append(float(environment.episode_return))
                        correct.append(bool(info["diagnosis_correct"]))
                        break
    finally:
        policy.train(was_training)
        environment.close()
    return {
        "mean_return": float(np.mean(returns)),
        "accuracy": float(np.mean(correct)),
        "optimal_first_action_rate": float(np.mean(first_optimal)),
    }


def _save_run_checkpoint(
    path: Path,
    policy: SmallPolicyNetwork,
    optimizer: torch.optim.Optimizer,
    config: PPOConfig,
    completed_transitions: int,
    score: float,
) -> None:
    torch.save({
        "checkpoint_version": CHECKPOINT_VERSION,
        "model_state": policy.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": config.as_dict(),
        "completed_transitions": completed_transitions,
        "validation_score": score,
        "feature_schema": _feature_schema(),
        "generator_version": GENERATOR_VERSION,
        "dependency_lock_sha256": _file_sha256(Path("uv.lock")),
        "continuation": "boundary_weights_only",
    }, path)


def _write_metrics(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-config", type=Path, required=True)
    parser.add_argument("--ppo-config", type=Path, required=True)
    parser.add_argument("--fixture", choices=tuple(fixture_catalog()))
    parser.add_argument("--train-cases", type=Path)
    parser.add_argument("--validation-cases", type=Path)
    parser.add_argument("--lambda-cost", type=float)
    parser.add_argument("--budget-credits", type=int)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--total-transitions", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.fixture is not None and args.train_cases is not None:
        raise ValueError("fixture and train-cases are mutually exclusive")
    result = run_training(
        sim_config_path=args.sim_config,
        ppo_config_path=args.ppo_config,
        seed=args.seed,
        total_transitions=args.total_transitions,
        run_directory=Path("runs") / args.run_id,
        fixture_name=args.fixture,
        train_cases=args.train_cases,
        validation_cases=args.validation_cases,
        lambda_cost_override=args.lambda_cost,
        budget_credits_override=args.budget_credits,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
