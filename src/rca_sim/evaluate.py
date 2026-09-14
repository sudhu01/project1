"""Evaluate non-learning policies on the same frozen incident cases."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from rca_sim.baselines import (
    BaselinePolicy,
    ImmediateStopPolicy,
    RandomAcquisitionPolicy,
    RandomIncludingStopPolicy,
    ScriptedInvestigatorPolicy,
)
from rca_sim.environment import InvestigationEnv
from rca_sim.world import HiddenIncidentWorld


CaseSource = HiddenIncidentWorld | Mapping[str, object] | str | Path
EnvironmentFactory = Callable[[], InvestigationEnv]

DEFAULT_PROBE_BUDGETS = (1, 2, 4, 6)
DEFAULT_ACTION_SEEDS = (9101, 9102, 9103, 9104, 9105)
DEFAULT_SCRIPT_THRESHOLDS = (0.60, 0.75, 0.90, 0.99)


@dataclass(frozen=True, slots=True)
class EvaluationRow:
    """One terminal episode result."""

    method: str
    case_id: str
    action_seed: int | None
    diagnosis_correct: bool
    episode_return: float
    credits_spent: int
    probes_taken: int
    predicted_service: int
    predicted_fault: str
    termination_reason: str
    actions: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class AggregateRow:
    """Aggregate for one method across cases and action seeds."""

    method: str
    episodes: int
    cases: int
    action_seeds: int
    accuracy: float
    mean_return: float
    mean_credits_spent: float
    mean_probes_taken: float


def evaluate_episode(
    environment: InvestigationEnv,
    case: CaseSource,
    policy: BaselinePolicy,
    *,
    fallback_case_id: str,
) -> EvaluationRow:
    """Run one policy on one frozen incident until true termination."""
    observation, reset_info = environment.reset(options={"case": case})
    policy.reset()
    actions: list[int] = []
    terminal_info: dict[str, Any] | None = None

    while not environment.terminated:
        action = policy.select_action(observation)
        actions.append(action)
        observation, _, terminated, truncated, info = environment.step(action)
        if truncated:
            raise RuntimeError("baseline evaluation does not allow truncation")
        if terminated:
            terminal_info = info

    if terminal_info is None or terminal_info.get("diagnosis_correct") is None:
        raise RuntimeError("episode terminated without a scored diagnosis")

    case_id = str(reset_info.get("case_id", fallback_case_id))
    return EvaluationRow(
        method=policy.name,
        case_id=case_id,
        action_seed=policy.action_seed,
        diagnosis_correct=bool(terminal_info["diagnosis_correct"]),
        episode_return=float(terminal_info["episode_return"]),
        credits_spent=int(terminal_info["credits_spent"]),
        probes_taken=int(terminal_info["probes_taken"]),
        predicted_service=int(terminal_info["predicted_service"]),
        predicted_fault=str(terminal_info["predicted_fault"]),
        termination_reason=str(terminal_info["termination_reason"]),
        actions=tuple(actions),
    )


def evaluate_baselines(
    cases: Iterable[CaseSource],
    *,
    probe_budgets: Sequence[int] = DEFAULT_PROBE_BUDGETS,
    action_seeds: Sequence[int] = DEFAULT_ACTION_SEEDS,
    script_thresholds: Sequence[float] = DEFAULT_SCRIPT_THRESHOLDS,
    include_stop: bool = True,
    include_random: bool = True,
    include_random_smoke: bool = True,
    include_script: bool = True,
    environment_factory: EnvironmentFactory = InvestigationEnv,
) -> tuple[EvaluationRow, ...]:
    """Evaluate baseline variants while reusing every case across methods."""
    frozen_cases = tuple(cases)
    if not frozen_cases:
        raise ValueError("cases cannot be empty")
    budgets = _unique_nonnegative(probe_budgets, "probe_budgets", positive=True)
    seeds = _unique_nonnegative(action_seeds, "action_seeds")
    if (include_random or include_random_smoke) and not seeds:
        raise ValueError("stochastic baselines require at least one action seed")

    policies: list[BaselinePolicy] = []
    if include_stop:
        policies.append(ImmediateStopPolicy())
    if include_random:
        policies.extend(
            RandomAcquisitionPolicy(probe_budget=budget, seed=seed)
            for budget in budgets
            for seed in seeds
        )
    if include_random_smoke:
        policies.extend(RandomIncludingStopPolicy(seed=seed) for seed in seeds)
    if include_script:
        policies.extend(
            ScriptedInvestigatorPolicy(confidence_threshold=threshold)
            for threshold in _unique_thresholds(script_thresholds)
        )
    if not policies:
        raise ValueError("at least one baseline method must be enabled")

    rows: list[EvaluationRow] = []
    for policy in policies:
        environment = environment_factory()
        try:
            for case_index, case in enumerate(frozen_cases):
                rows.append(
                    evaluate_episode(
                        environment,
                        case,
                        policy,
                        fallback_case_id=f"case_{case_index:04d}",
                    )
                )
        finally:
            environment.close()
    return tuple(rows)


def aggregate_results(rows: Iterable[EvaluationRow]) -> tuple[AggregateRow, ...]:
    """Summarize episode rows by named method variant."""
    grouped: dict[str, list[EvaluationRow]] = defaultdict(list)
    for row in rows:
        if not isinstance(row, EvaluationRow):
            raise TypeError("rows must contain EvaluationRow values")
        grouped[row.method].append(row)
    if not grouped:
        raise ValueError("rows cannot be empty")

    summaries: list[AggregateRow] = []
    for method in sorted(grouped):
        method_rows = grouped[method]
        summaries.append(
            AggregateRow(
                method=method,
                episodes=len(method_rows),
                cases=len({row.case_id for row in method_rows}),
                action_seeds=len(
                    {
                        row.action_seed
                        for row in method_rows
                        if row.action_seed is not None
                    }
                ),
                accuracy=float(np.mean([row.diagnosis_correct for row in method_rows])),
                mean_return=float(np.mean([row.episode_return for row in method_rows])),
                mean_credits_spent=float(
                    np.mean([row.credits_spent for row in method_rows])
                ),
                mean_probes_taken=float(
                    np.mean([row.probes_taken for row in method_rows])
                ),
            )
        )
    return tuple(summaries)


def _environment_factory_from_config(path: Path | None) -> EnvironmentFactory:
    if path is None:
        return InvestigationEnv
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("simulator config must contain a mapping")
    values = {
        "n_services": payload.get("services", 8),
        "initial_budget": payload.get("budget_credits", 8),
        "max_probes": payload.get("max_probes", 6),
        "lambda_cost": payload.get("lambda_cost", 0.05),
        "extra_edge_probability": payload.get("extra_edge_probability", 0.15),
    }
    return lambda: InvestigationEnv(**values)


def _case_paths(path: Path) -> tuple[Path, ...]:
    if path.is_file():
        return (path,)
    if not path.is_dir():
        raise ValueError(f"case path does not exist: {path}")
    cases = tuple(sorted(path.rglob("*.json")))
    if not cases:
        raise ValueError(f"case directory contains no JSON files: {path}")
    return cases


def _write_results(
    output: Path,
    rows: Sequence[EvaluationRow],
    summaries: Sequence[AggregateRow],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with (output / "episodes.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), sort_keys=True) + "\n")
    (output / "summary.json").write_text(
        json.dumps([asdict(summary) for summary in summaries], indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _unique_nonnegative(
    values: Sequence[int],
    name: str,
    *,
    positive: bool = False,
) -> tuple[int, ...]:
    result: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must contain integers")
        if value < int(positive):
            bound = "positive" if positive else "nonnegative"
            raise ValueError(f"{name} must contain {bound} integers")
        if value not in result:
            result.append(value)
    return tuple(result)


def _unique_thresholds(values: Sequence[float]) -> tuple[float, ...]:
    result: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("script_thresholds must contain real numbers")
        threshold = float(value)
        if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("script_thresholds must be between 0 and 1")
        if threshold not in result:
            result.append(threshold)
    if not result:
        raise ValueError("script_thresholds cannot be empty")
    return tuple(result)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("stop", "random", "random_stop", "script"),
        default=("stop", "random", "random_stop", "script"),
    )
    parser.add_argument(
        "--probe-budgets", nargs="+", type=int, default=DEFAULT_PROBE_BUDGETS
    )
    parser.add_argument(
        "--action-seeds", nargs="+", type=int, default=DEFAULT_ACTION_SEEDS
    )
    parser.add_argument(
        "--script-thresholds",
        nargs="+",
        type=float,
        default=DEFAULT_SCRIPT_THRESHOLDS,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    methods = set(args.methods)
    rows = evaluate_baselines(
        _case_paths(args.cases),
        probe_budgets=args.probe_budgets,
        action_seeds=args.action_seeds,
        script_thresholds=args.script_thresholds,
        include_stop="stop" in methods,
        include_random="random" in methods,
        include_random_smoke="random_stop" in methods,
        include_script="script" in methods,
        environment_factory=_environment_factory_from_config(args.config),
    )
    summaries = aggregate_results(rows)
    _write_results(args.output, rows, summaries)
    print(json.dumps([asdict(summary) for summary in summaries], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
