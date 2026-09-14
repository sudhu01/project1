"""Non-learning policies for the small simulator baselines."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from numbers import Real
from typing import Protocol

import numpy as np

from rca_sim.observation import (
    GLOBAL_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
    OBSERVATION_SHAPES,
    Observation,
)
from rca_sim.tools import (
    ACTION_SPACE_SIZE,
    STOP_ACTION_INDEX,
    probe_action_index,
)


_ENTRY_FEATURE = NODE_FEATURE_NAMES.index("is_entry_service")
_METRICS_QUICK_EXECUTED_FEATURE = NODE_FEATURE_NAMES.index(
    "template_executed.metrics.quick"
)
_CONFIDENCE_FEATURE = GLOBAL_FEATURE_NAMES.index("largest_service_probability")


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


@dataclass(slots=True)
class ScriptedInvestigatorPolicy:
    """Probe quick metrics in deterministic dependency breadth-first order."""

    confidence_threshold: float

    def __post_init__(self) -> None:
        if isinstance(self.confidence_threshold, bool) or not isinstance(
            self.confidence_threshold, Real
        ):
            raise TypeError("confidence_threshold must be a real number")
        threshold = float(self.confidence_threshold)
        if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        self.confidence_threshold = threshold

    @property
    def name(self) -> str:
        return f"script_{self.confidence_threshold:.2f}"

    @property
    def action_seed(self) -> None:
        return None

    def reset(self) -> None:
        pass

    def select_action(self, observation: Observation) -> int:
        mask = _valid_action_mask(observation)
        node_features, node_mask, adjacency, global_features = _script_inputs(
            observation
        )
        order = _dependency_breadth_first_order(
            node_features,
            node_mask,
            adjacency,
        )
        quick_executed = node_features[:, _METRICS_QUICK_EXECUTED_FEATURE] > 0.5

        entry_service = order[0]
        entry_action = probe_action_index(entry_service, "metrics.quick")
        if not quick_executed[entry_service] and mask[entry_action]:
            return entry_action

        acquired_quick_metrics = bool(np.any(quick_executed[node_mask]))
        confidence = float(global_features[_CONFIDENCE_FEATURE])
        if acquired_quick_metrics and confidence >= self.confidence_threshold:
            return STOP_ACTION_INDEX

        for service_id in order[1:]:
            action = probe_action_index(service_id, "metrics.quick")
            if not quick_executed[service_id] and mask[action]:
                return action
        return STOP_ACTION_INDEX


def _valid_action_mask(observation: Observation) -> np.ndarray:
    if not isinstance(observation, dict) or "action_mask" not in observation:
        raise TypeError("observation must contain an action_mask")
    mask = np.asarray(observation["action_mask"])
    if mask.shape != (ACTION_SPACE_SIZE,) or mask.dtype != np.bool_:
        raise ValueError(
            f"action_mask must be a boolean array with shape ({ACTION_SPACE_SIZE},)"
        )
    return mask


def _script_inputs(
    observation: Observation,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    required = ("node_features", "node_mask", "adjacency", "global_features")
    missing = [name for name in required if name not in observation]
    if missing:
        raise TypeError(f"observation is missing {', '.join(missing)}")

    node_features = np.asarray(observation["node_features"])
    node_mask = np.asarray(observation["node_mask"])
    adjacency = np.asarray(observation["adjacency"])
    global_features = np.asarray(observation["global_features"])
    expected = {
        "node_features": OBSERVATION_SHAPES["node_features"],
        "node_mask": OBSERVATION_SHAPES["node_mask"],
        "adjacency": OBSERVATION_SHAPES["adjacency"],
        "global_features": OBSERVATION_SHAPES["global_features"],
    }
    actual = {
        "node_features": node_features.shape,
        "node_mask": node_mask.shape,
        "adjacency": adjacency.shape,
        "global_features": global_features.shape,
    }
    for name, shape in actual.items():
        if shape != expected[name]:
            raise ValueError(f"{name} must have shape {expected[name]}")
    if node_mask.dtype != np.bool_:
        raise ValueError("node_mask must be a boolean array")
    if not np.any(node_mask):
        raise ValueError("observation has no active services")
    return node_features, node_mask, adjacency, global_features


def _dependency_breadth_first_order(
    node_features: np.ndarray,
    node_mask: np.ndarray,
    adjacency: np.ndarray,
) -> tuple[int, ...]:
    active = np.flatnonzero(node_mask)
    entries = [
        int(service_id)
        for service_id in active
        if node_features[service_id, _ENTRY_FEATURE] > 0.5
    ]
    if len(entries) != 1:
        raise ValueError("observation must identify exactly one entry service")

    order = [entries[0]]
    seen = {entries[0]}
    next_index = 0
    while next_index < len(order):
        caller = order[next_index]
        next_index += 1
        for dependency in np.flatnonzero(adjacency[caller] > 0.5):
            dependency_id = int(dependency)
            if node_mask[dependency_id] and dependency_id not in seen:
                seen.add(dependency_id)
                order.append(dependency_id)

    if len(order) != len(active):
        raise ValueError("not every active service is reachable from the entry service")
    return tuple(order)


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
