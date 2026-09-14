"""Non-learning policies for the small simulator baselines."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from typing import Protocol

import numpy as np

from rca_sim.observation import Observation
from rca_sim.tools import ACTION_SPACE_SIZE, STOP_ACTION_INDEX


class BaselinePolicy(Protocol):
    """The policy interface used by the baseline evaluator."""

    @property
    def name(self) -> str:
        """Return a stable method name for result rows."""

    @property
    def action_seed(self) -> int | None:
        """Return the independent policy RNG seed, when one exists."""

    def reset(self) -> None:
        """Clear episode-local policy state without rewinding its RNG."""

    def select_action(self, observation: Observation) -> int:
        """Select one action using only the public observation."""


@dataclass(slots=True)
class ImmediateStopPolicy:
    """Stop at the initial belief without acquiring evidence."""

    @property
    def name(self) -> str:
        return "stop"

    @property
    def action_seed(self) -> None:
        return None

    def reset(self) -> None:
        pass

    def select_action(self, observation: Observation) -> int:
        _valid_action_mask(observation)
        return STOP_ACTION_INDEX


@dataclass(slots=True)
class RandomAcquisitionPolicy:
    """Acquire uniform feasible probes, then stop at a fixed probe count."""

    probe_budget: int
    seed: int
    _rng: np.random.Generator = field(init=False, repr=False)
    _probes_selected: int = field(init=False, default=0, repr=False)

    def __post_init__(self) -> None:
        self.probe_budget = _positive_integer(self.probe_budget, "probe_budget")
        self.seed = _nonnegative_integer(self.seed, "seed")
        self._rng = np.random.default_rng(self.seed)

    @property
    def name(self) -> str:
        return f"random_{self.probe_budget}_probes"

    @property
    def action_seed(self) -> int:
        return self.seed

    def reset(self) -> None:
        self._probes_selected = 0

    def select_action(self, observation: Observation) -> int:
        mask = _valid_action_mask(observation)
        if self._probes_selected >= self.probe_budget:
            return STOP_ACTION_INDEX

        feasible_probes = np.flatnonzero(mask[:STOP_ACTION_INDEX])
        if feasible_probes.size == 0:
            return STOP_ACTION_INDEX

        action = int(self._rng.choice(feasible_probes))
        self._probes_selected += 1
        return action


@dataclass(slots=True)
class RandomIncludingStopPolicy:
    """Smoke-test policy that samples STOP along with feasible probes."""

    seed: int
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.seed = _nonnegative_integer(self.seed, "seed")
        self._rng = np.random.default_rng(self.seed)

    @property
    def name(self) -> str:
        return "random_including_stop"

    @property
    def action_seed(self) -> int:
        return self.seed

    def reset(self) -> None:
        pass

    def select_action(self, observation: Observation) -> int:
        mask = _valid_action_mask(observation)
        feasible_actions = np.flatnonzero(mask)
        if feasible_actions.size == 0:
            raise ValueError("observation has no feasible actions")
        return int(self._rng.choice(feasible_actions))


def _valid_action_mask(observation: Observation) -> np.ndarray:
    if not isinstance(observation, dict) or "action_mask" not in observation:
        raise TypeError("observation must contain an action_mask")
    mask = np.asarray(observation["action_mask"])
    if mask.shape != (ACTION_SPACE_SIZE,) or mask.dtype != np.bool_:
        raise ValueError(
            f"action_mask must be a boolean array with shape ({ACTION_SPACE_SIZE},)"
        )
    return mask


def _positive_integer(value: object, name: str) -> int:
    value = _nonnegative_integer(value, name)
    if value == 0:
        raise ValueError(f"{name} must be positive")
    return value


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value
