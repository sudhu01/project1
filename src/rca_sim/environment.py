"""Gymnasium lifecycle for a diagnostic investigation episode."""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping
from numbers import Integral, Real
from os import PathLike
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from rca_sim.belief import ExactBeliefEstimator
from rca_sim.contracts import EvidenceRecord
from rca_sim.evidence import EvidenceLedger
from rca_sim.observation import (
    OBSERVATION_SCHEMA_VERSION,
    Observation,
    build_observation,
    make_observation_space,
)
from rca_sim.tools import (
    MAX_SERVICE_SLOTS,
    STOP_ACTION_INDEX,
    ProbeTool,
    construct_action_mask,
    execute_probe_action,
    is_stop_action,
)
from rca_sim.world import (
    GENERATOR_VERSION,
    HiddenIncidentWorld,
    generate_incident_world,
)
from rca_sim.graph import DependencyGraph


DEFAULT_SERVICES = 8
DEFAULT_INITIAL_BUDGET = 8
DEFAULT_MAX_PROBES = 6
DEFAULT_LAMBDA_COST = 0.05
DEFAULT_EXTRA_EDGE_PROBABILITY = 0.15

TERMINATION_BUDGET = "budget"
TERMINATION_HORIZON = "horizon"
TERMINATION_NO_PROBE = "no_probe_available"
TERMINATION_STOP = "stop"


class InvestigationEnv(gym.Env[Observation, int]):
    """Small exact RCA simulator with explicit probe and STOP transitions."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        n_services: int = DEFAULT_SERVICES,
        initial_budget: int = DEFAULT_INITIAL_BUDGET,
        max_probes: int = DEFAULT_MAX_PROBES,
        lambda_cost: float = DEFAULT_LAMBDA_COST,
        extra_edge_probability: float = DEFAULT_EXTRA_EDGE_PROBABILITY,
        available_tools: Mapping[int, Collection[ProbeTool]] | None = None,
        trace_enabled: bool = False,
    ) -> None:
        super().__init__()
        self._configured_n_services = _bounded_positive_integer(
            n_services,
            "n_services",
            upper=MAX_SERVICE_SLOTS,
        )
        self.initial_budget = _bounded_positive_integer(
            initial_budget,
            "initial_budget",
            upper=16,
        )
        self.max_probes = _positive_integer(max_probes, "max_probes")
        self.lambda_cost = _bounded_real(
            lambda_cost,
            "lambda_cost",
            lower=0.0,
            upper=0.20,
        )
        self.extra_edge_probability = _bounded_real(
            extra_edge_probability,
            "extra_edge_probability",
            lower=0.0,
            upper=1.0,
        )
        self.available_tools = _copy_available_tools(available_tools)
        if not isinstance(trace_enabled, (bool, np.bool_)):
            raise TypeError("trace_enabled must be boolean")
        self.trace_enabled = bool(trace_enabled)

        self.action_space = gym.spaces.Discrete(STOP_ACTION_INDEX + 1)
        self.observation_space = make_observation_space()

        self._world: HiddenIncidentWorld | None = None
        self._ledger: EvidenceLedger | None = None
        self._belief: ExactBeliefEstimator | None = None
        self._observation: Observation | None = None
        self._last_pre_action_observation: Observation | None = None
        self._executed_probe_actions: list[int] = []
        self._remaining_credits = self.initial_budget
        self._probes_taken = 0
        self._terminated = False
        self._termination_reason: str | None = None
        self._episode_return = 0.0
        self._case_id: str | None = None
        self._trace_events: list[dict[str, Any]] = []

    @property
    def remaining_credits(self) -> int:
        return self._remaining_credits

    @property
    def configured_n_services(self) -> int:
        return self._configured_n_services

    @property
    def probes_taken(self) -> int:
        return self._probes_taken

    @property
    def terminated(self) -> bool:
        return self._terminated

    @property
    def termination_reason(self) -> str | None:
        return self._termination_reason

    @property
    def episode_return(self) -> float:
        return self._episode_return

    @property
    def current_observation(self) -> Observation | None:
        """Return a copy of the live state for a collection cutoff.

        Reading this property does not terminate, truncate, reset, or score
        the episode. A rollout collector can bootstrap from this observation
        and continue the same incident in its next batch.
        """
        if self._observation is None:
            return None
        return _copy_observation(self._observation)

    @property
    def evidence_records(self) -> tuple[EvidenceRecord, ...]:
        """Return the acquired public evidence in first-seen order."""
        return () if self._ledger is None else self._ledger.records

    @property
    def public_graph(self) -> DependencyGraph:
        """Return the immutable public dependency graph for exact planning."""
        if self._world is None:
            raise RuntimeError("call reset before reading the public graph")
        return self._world.graph

    @property
    def belief_posterior(self) -> np.ndarray:
        """Return the inferred public posterior without hidden incident data."""
        if self._belief is None:
            raise RuntimeError("call reset before reading the belief posterior")
        return self._belief.posterior

    @property
    def executed_probe_actions(self) -> tuple[int, ...]:
        return tuple(self._executed_probe_actions)

    @property
    def last_pre_action_observation(self) -> Observation | None:
        """Return a fresh copy of the observation saved before the last probe."""
        if self._last_pre_action_observation is None:
            return None
        return _copy_observation(self._last_pre_action_observation)

    @property
    def trace_events(self) -> tuple[dict[str, Any], ...]:
        """Return copies of public trace events in episode order."""
        return tuple(dict(event) for event in self._trace_events)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Observation, dict[str, Any]]:
        """Start a generated incident or load an exact evaluation case."""
        super().reset(seed=seed)
        resolved_options = _validate_reset_options(options)
        case_keys = tuple(
            key for key in ("case", "case_path", "world") if key in resolved_options
        )
        if len(case_keys) > 1:
            raise ValueError("reset options may specify only one exact case source")
        case = resolved_options.get(case_keys[0]) if case_keys else None

        if case is None:
            incident_seed = int(
                self.np_random.integers(0, np.iinfo(np.int64).max, dtype=np.int64)
            )
            world = generate_incident_world(
                n_services=self._configured_n_services,
                incident_seed=incident_seed,
                extra_edge_probability=self.extra_edge_probability,
            )
            case_id = None
        else:
            world, case_id = load_exact_case(
                case,
                default_n_services=self._configured_n_services,
                default_edge_probability=self.extra_edge_probability,
            )

        self._world = world
        self._ledger = EvidenceLedger()
        self._belief = ExactBeliefEstimator(world.graph)
        self._executed_probe_actions = []
        self._remaining_credits = self.initial_budget
        self._probes_taken = 0
        self._terminated = False
        self._termination_reason = None
        self._episode_return = 0.0
        self._last_pre_action_observation = None
        self._case_id = case_id
        self._observation = self._build_observation()

        self._trace_events = []
        self._record_trace("reset", action=None, reward=0.0)

        return _copy_observation(self._observation), self._public_info()

    def step(
        self,
        action: int,
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        """Execute one probe or stop and score the current diagnosis."""
        world, ledger, belief, observation = self._require_active_episode()
        action_index = _action_index(action)
        if is_stop_action(action_index):
            return self._step_stop(observation=observation)
        if not bool(observation["action_mask"][action_index]):
            raise ValueError(f"action {action_index} is invalid for the current state")

        pre_action_observation = _copy_observation(observation)
        result = execute_probe_action(
            world,
            action_index,
            remaining_credits=self._remaining_credits,
            seen_evidence_ids=ledger.evidence_ids,
            available_tools=self.available_tools,
        )

        self._last_pre_action_observation = pre_action_observation
        self._remaining_credits -= result.cost.credits
        unseen_records = ledger.add(result.evidence)
        belief.update(unseen_records)
        self._probes_taken += 1
        self._executed_probe_actions.append(action_index)

        self._termination_reason = self._probe_termination_reason()
        self._terminated = self._termination_reason is not None
        reward, diagnosis_correct = self._score_transition(
            step_cost=result.cost.credits,
            terminal=self._terminated,
        )
        self._observation = self._build_observation()
        info = self._public_info()
        info.update(
            {
                "action_type": "probe",
                "probe": result.as_dict(),
                "new_evidence_ids": tuple(
                    record.evidence_id for record in unseen_records
                ),
                "step_cost": result.cost.credits,
                "diagnosis_correct": diagnosis_correct,
                "episode_return": self._episode_return,
            }
        )
        self._record_trace("probe", action=action_index, reward=float(reward))
        return (
            _copy_observation(self._observation),
            float(reward),
            self._terminated,
            False,
            info,
        )

    def _step_stop(
        self,
        *,
        observation: Observation,
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        self._last_pre_action_observation = _copy_observation(observation)
        self._termination_reason = TERMINATION_STOP
        self._terminated = True
        reward, diagnosis_correct = self._score_transition(
            step_cost=0,
            terminal=True,
        )
        self._observation = self._build_observation()
        info = self._public_info()
        info.update(
            {
                "action_type": "stop",
                "new_evidence_ids": (),
                "step_cost": 0,
                "diagnosis_correct": diagnosis_correct,
                "episode_return": self._episode_return,
            }
        )
        self._record_trace("stop", action=STOP_ACTION_INDEX, reward=float(reward))
        return (
            _copy_observation(self._observation),
            float(reward),
            True,
            False,
            info,
        )

    def _score_transition(
        self,
        *,
        step_cost: int,
        terminal: bool,
    ) -> tuple[float, bool | None]:
        world, _, belief, _ = self._require_initialized_episode(
            require_observation=False
        )
        diagnosis_correct = None
        terminal_utility = 0.0
        if terminal:
            diagnosis_correct = (
                belief.predicted_service == world.hypothesis.cause_service
            )
            terminal_utility = float(diagnosis_correct)
        reward = -self.lambda_cost * step_cost + terminal_utility
        self._episode_return += reward
        if terminal:
            expected_return = (
                terminal_utility
                - self.lambda_cost
                * (self.initial_budget - self._remaining_credits)
            )
            if not np.isclose(self._episode_return, expected_return, atol=1e-12):
                raise RuntimeError("episode reward accounting is inconsistent")
            lower_bound = -self.lambda_cost * self.initial_budget
            if not lower_bound - 1e-12 <= self._episode_return <= 1.0 + 1e-12:
                raise RuntimeError("episode return is outside configured bounds")
        return float(reward), diagnosis_correct

    def _probe_termination_reason(self) -> str | None:
        if self._remaining_credits == 0:
            return TERMINATION_BUDGET
        if self._probes_taken >= self.max_probes:
            return TERMINATION_HORIZON
        world, ledger, _, _ = self._require_initialized_episode()
        mask = construct_action_mask(
            n_services=world.graph.n_services,
            remaining_credits=self._remaining_credits,
            seen_evidence_ids=ledger.evidence_ids,
            available_tools=self.available_tools,
        )
        if not bool(mask[:STOP_ACTION_INDEX].any()):
            return TERMINATION_NO_PROBE
        return None

    def _build_observation(self) -> Observation:
        world, ledger, belief, _ = self._require_initialized_episode(
            require_observation=False
        )
        return build_observation(
            graph=world.graph,
            belief=belief,
            evidence_records=ledger.records,
            executed_probe_actions=self._executed_probe_actions,
            initial_budget=self.initial_budget,
            remaining_credits=self._remaining_credits,
            probes_taken=self._probes_taken,
            max_probes=self.max_probes,
            lambda_cost=self.lambda_cost,
            available_tools=self.available_tools,
            terminated=self._terminated,
        )

    def _public_info(self) -> dict[str, Any]:
        world, _, belief, _ = self._require_initialized_episode(
            require_observation=False
        )
        info: dict[str, Any] = {
            "schema_version": OBSERVATION_SCHEMA_VERSION,
            "generator_version": world.generator_version,
            "n_services": world.graph.n_services,
            "entry_service": world.graph.entry_service,
            "initial_budget": self.initial_budget,
            "remaining_credits": self._remaining_credits,
            "credits_spent": self.initial_budget - self._remaining_credits,
            "max_probes": self.max_probes,
            "probes_taken": self._probes_taken,
            "predicted_service": belief.predicted_service,
            "predicted_fault": belief.predicted_fault.name.lower(),
            "termination_reason": self._termination_reason,
        }
        if self._case_id is not None:
            info["case_id"] = self._case_id
        return info

    def _record_trace(
        self,
        event: str,
        *,
        action: int | None,
        reward: float,
    ) -> None:
        if not self.trace_enabled:
            return
        _, ledger, belief, _ = self._require_initialized_episode(
            require_observation=False
        )
        self._trace_events.append(
            {
                "event": event,
                "action": action,
                "reward": reward,
                "remaining_credits": self._remaining_credits,
                "probes_taken": self._probes_taken,
                "evidence_count": len(ledger),
                "predicted_service": belief.predicted_service,
                "predicted_fault": belief.predicted_fault.name.lower(),
                "terminated": self._terminated,
                "termination_reason": self._termination_reason,
            }
        )

    def _require_active_episode(
        self,
    ) -> tuple[
        HiddenIncidentWorld,
        EvidenceLedger,
        ExactBeliefEstimator,
        Observation,
    ]:
        world, ledger, belief, observation = self._require_initialized_episode()
        if self._terminated:
            raise RuntimeError("episode has terminated; call reset before step")
        assert observation is not None
        return world, ledger, belief, observation

    def _require_initialized_episode(
        self,
        *,
        require_observation: bool = True,
    ) -> tuple[
        HiddenIncidentWorld,
        EvidenceLedger,
        ExactBeliefEstimator,
        Observation | None,
    ]:
        if self._world is None or self._ledger is None or self._belief is None:
            raise RuntimeError("call reset before using the environment")
        if require_observation and self._observation is None:
            raise RuntimeError("call reset before using the environment")
        return self._world, self._ledger, self._belief, self._observation


def _copy_observation(observation: Observation) -> Observation:
    return {
        key: np.array(value, copy=True)
        for key, value in observation.items()
    }


def _validate_reset_options(options: dict[str, Any] | None) -> dict[str, Any]:
    if options is None:
        return {}
    if not isinstance(options, dict):
        raise TypeError("reset options must be a dictionary")
    unknown = set(options) - {"case", "case_path", "world"}
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"unknown reset option: {names}")
    return options


def load_exact_case(
    case: object,
    *,
    default_n_services: int,
    default_edge_probability: float,
) -> tuple[HiddenIncidentWorld, str | None]:
    """Load an evaluator-only frozen incident from a supported case source."""
    if isinstance(case, HiddenIncidentWorld):
        if case.graph.n_services > MAX_SERVICE_SLOTS:
            raise ValueError("evaluation case exceeds the observation service limit")
        if case.generator_version != GENERATOR_VERSION:
            raise ValueError(
                f"evaluation case requires unsupported generator "
                f"{case.generator_version!r}"
            )
        return case, None
    if isinstance(case, (str, PathLike)):
        path = Path(case)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"could not load evaluation case {path}") from error
        if not isinstance(payload, dict):
            raise ValueError("evaluation case JSON must contain an object")
        case_id = str(payload.get("case_id", path.stem))
        descriptor: Mapping[str, object] = payload
    elif isinstance(case, Mapping):
        descriptor = case
        raw_case_id = descriptor.get("case_id")
        case_id = None if raw_case_id is None else str(raw_case_id)
    else:
        raise TypeError(
            "reset option 'case' must be a world, mapping, or JSON path"
        )

    nested_world = descriptor.get("world")
    if isinstance(nested_world, Mapping):
        descriptor = nested_world
    generator_version = descriptor.get("generator_version", GENERATOR_VERSION)
    if generator_version != GENERATOR_VERSION:
        raise ValueError(
            f"evaluation case requires unsupported generator {generator_version!r}"
        )
    if "incident_seed" not in descriptor:
        raise ValueError("evaluation case must contain incident_seed")
    incident_seed = _nonnegative_integer(
        descriptor["incident_seed"],
        "incident_seed",
    )
    raw_n_services = descriptor.get(
        "n_services",
        descriptor.get("services", default_n_services),
    )
    n_services = _bounded_positive_integer(
        raw_n_services,
        "case n_services",
        upper=MAX_SERVICE_SLOTS,
    )
    edge_probability = _bounded_real(
        descriptor.get("extra_edge_probability", default_edge_probability),
        "case extra_edge_probability",
        lower=0.0,
        upper=1.0,
    )
    return (
        generate_incident_world(
            n_services=n_services,
            incident_seed=incident_seed,
            extra_edge_probability=edge_probability,
        ),
        case_id,
    )


def _action_index(action: int) -> int:
    if isinstance(action, bool) or not isinstance(action, Integral):
        raise TypeError("action must be an integer")
    action = int(action)
    if not 0 <= action <= STOP_ACTION_INDEX:
        raise ValueError(f"action must be between 0 and {STOP_ACTION_INDEX}")
    return action


def _copy_available_tools(
    available_tools: Mapping[int, Collection[ProbeTool]] | None,
) -> dict[int, frozenset[ProbeTool]] | None:
    if available_tools is None:
        return None
    if not isinstance(available_tools, Mapping):
        raise TypeError("available_tools must be a mapping")
    copied: dict[int, frozenset[ProbeTool]] = {}
    for service_id, tools in available_tools.items():
        if isinstance(tools, (str, bytes)) or not isinstance(tools, Collection):
            raise TypeError("available tool values must be collections")
        copied[service_id] = frozenset(tools)
    return copied


def _bounded_positive_integer(value: object, name: str, *, upper: int) -> int:
    value = _positive_integer(value, name)
    if value > upper:
        raise ValueError(f"{name} must be at most {upper}")
    return value


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


def _bounded_real(
    value: object,
    name: str,
    *,
    lower: float,
    upper: float,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not np.isfinite(value) or not lower <= value <= upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")
    return value
