"""Profile valid random simulator transitions by major component."""

from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import tracemalloc
from pathlib import Path
from time import perf_counter
from typing import Sequence

import numpy as np

from rca_sim.config import load_simulator_config
from rca_sim.environment import InvestigationEnv


def profile_transitions(*, config_path: Path, transitions: int, seed: int) -> dict[str, object]:
    if transitions < 1:
        raise ValueError("transitions must be positive")
    config = load_simulator_config(config_path)
    environment = InvestigationEnv(**config.environment_kwargs())
    action_rng = np.random.default_rng(seed)
    episode_rng = np.random.default_rng(seed + 1)
    profiler = cProfile.Profile()
    tracemalloc.start()
    started = perf_counter()
    episodes = 1
    try:
        profiler.enable()
        observation, _ = environment.reset(
            seed=int(episode_rng.integers(0, np.iinfo(np.int32).max))
        )
        for index in range(transitions):
            valid_actions = np.flatnonzero(observation["action_mask"])
            action = int(action_rng.choice(valid_actions))
            observation, _, terminated, truncated, _ = environment.step(action)
            if truncated:
                raise RuntimeError("the base simulator unexpectedly truncated")
            if terminated and index + 1 < transitions:
                observation, _ = environment.reset(
                    seed=int(episode_rng.integers(0, np.iinfo(np.int32).max))
                )
                episodes += 1
        profiler.disable()
        elapsed = perf_counter() - started
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        profiler.disable()
        tracemalloc.stop()
        environment.close()

    stats = pstats.Stats(profiler)
    component_functions = {
        "world_generation_seconds": "generate_incident_world",
        "inference_seconds": "update",
        "observation_construction_seconds": "build_observation",
    }
    components: dict[str, float] = {}
    for output_name, function_name in component_functions.items():
        components[output_name] = float(sum(
            values[3]
            for key, values in stats.stats.items()
            if key[2] == function_name
            and (function_name != "update" or key[0].endswith("belief.py"))
        ))
    return {
        "seed": seed,
        "transitions": transitions,
        "episodes": episodes,
        "wall_seconds": elapsed,
        "transitions_per_second": transitions / elapsed,
        "peak_traced_memory_bytes": peak_bytes,
        "memory_measurement": "Python allocations measured by tracemalloc",
        **components,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--transitions", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = profile_transitions(
        config_path=args.config, transitions=args.transitions, seed=args.seed
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
