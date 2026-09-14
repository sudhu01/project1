"""Finite diagnostic fixtures with exact beliefs and decision values."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any, Mapping

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from rca_sim.belief import DiagnosticQuantities, InconsistentEvidenceError
from rca_sim.contracts import (
    EvidenceRecord,
    FaultType,
    MetricCategory,
    MetricEvidenceRecord,
    Workload,
)
from rca_sim.evidence import EvidenceConflictError, EvidenceLedger
from rca_sim.graph import DependencyGraph
from rca_sim.observation import Observation, build_observation, make_observation_space
from rca_sim.tools import (
    ACTION_SPACE_SIZE,
    STOP_ACTION_INDEX,
    ProbeAction,
    ProbeCost,
    ProbeCoverage,
    ProbePreset,
    ProbeResult,
    ProbeStatus,
    ProbeTool,
    decode_probe_action,
    probe_action_index,
)


FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class FixtureHypothesis:
    """One valid latent state and its scored diagnosis label."""

    state_id: str
    cause_service: int
    fault_type: FaultType = FaultType.CPU
    workload: Workload = Workload.NORMAL

    def __post_init__(self) -> None:
        if not isinstance(self.state_id, str) or not self.state_id:
            raise ValueError("state_id must be a nonempty string")
        if isinstance(self.cause_service, bool) or not isinstance(
            self.cause_service,
            Integral,
        ):
            raise TypeError("cause_service must be an integer")
        if self.cause_service < 0:
            raise ValueError("cause_service must be nonnegative")
        if not isinstance(self.fault_type, FaultType):
            raise TypeError("fault_type must be a FaultType")
        if not isinstance(self.workload, Workload):
            raise TypeError("workload must be a Workload")
        object.__setattr__(self, "cause_service", int(self.cause_service))


@dataclass(frozen=True, slots=True)
class FixtureEvidence:
    """A binary public record and its true probability in each latent state."""

    service_id: int
    metric: MetricCategory
    reading_index: int
    true_probabilities: tuple[float, ...]

    def __post_init__(self) -> None:
        prototype = MetricEvidenceRecord(
            self.service_id,
            self.metric,
            self.reading_index,
            False,
        )
        probabilities = tuple(float(value) for value in self.true_probabilities)
        if not probabilities:
            raise ValueError("true_probabilities cannot be empty")
        if any(not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError("evidence probabilities must be between zero and one")
        object.__setattr__(self, "service_id", prototype.service_id)
        object.__setattr__(self, "metric", prototype.metric)
        object.__setattr__(self, "reading_index", prototype.reading_index)
        object.__setattr__(self, "true_probabilities", probabilities)

    @property
    def evidence_id(self) -> str:
        return self.record(False).evidence_id

    def record(self, value: bool) -> MetricEvidenceRecord:
        return MetricEvidenceRecord(
            self.service_id,
            self.metric,
            self.reading_index,
            value,
        )


@dataclass(frozen=True, slots=True)
class FixtureProbe:
    """One enabled action with fixture-specific cost and evidence coverage."""

    action_index: int
    cost_credits: int
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        service_id, _ = decode_probe_action(self.action_index)
        del service_id
        if isinstance(self.cost_credits, bool) or not isinstance(
            self.cost_credits,
            Integral,
        ):
            raise TypeError("cost_credits must be an integer")
        if self.cost_credits <= 0:
            raise ValueError("cost_credits must be positive")
        if not self.evidence_ids or any(
            not isinstance(item, str) or not item for item in self.evidence_ids
        ):
            raise ValueError("evidence_ids must contain nonempty strings")
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("a probe cannot repeat an evidence ID")
        object.__setattr__(self, "cost_credits", int(self.cost_credits))


@dataclass(frozen=True, slots=True)
class FixtureDefinition:
    """A complete finite fixture model."""

    name: str
    graph: DependencyGraph
    hypotheses: tuple[FixtureHypothesis, ...]
    evidence: tuple[FixtureEvidence, ...]
    probes: tuple[FixtureProbe, ...]
    prior: tuple[float, ...] | None = None
    initial_evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("fixture name must be nonempty")
        if not isinstance(self.graph, DependencyGraph):
            raise TypeError("graph must be a DependencyGraph")
        if not self.hypotheses:
            raise ValueError("a fixture needs at least one valid hypothesis")
        if len({item.state_id for item in self.hypotheses}) != len(self.hypotheses):
            raise ValueError("fixture hypothesis state IDs must be unique")
        if any(item.cause_service >= self.graph.n_services for item in self.hypotheses):
            raise ValueError("fixture cause service is not present in the graph")
        if any(
            len(item.true_probabilities) != len(self.hypotheses)
            for item in self.evidence
        ):
            raise ValueError("each evidence probability row must match the hypotheses")
        evidence_ids = tuple(item.evidence_id for item in self.evidence)
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("fixture evidence IDs must be unique")
        if any(item.service_id >= self.graph.n_services for item in self.evidence):
            raise ValueError("fixture evidence targets an inactive service")
        evidence_set = set(evidence_ids)
        if any(item not in evidence_set for item in self.initial_evidence_ids):
            raise ValueError("initial evidence is not defined by the fixture")
        if any(
            evidence_id not in evidence_set
            for probe in self.probes
            for evidence_id in probe.evidence_ids
        ):
            raise ValueError("probe coverage contains unknown evidence")
        if len({item.action_index for item in self.probes}) != len(self.probes):
            raise ValueError("fixture probe action indices must be unique")
        if any(
            decode_probe_action(item.action_index)[0] >= self.graph.n_services
            for item in self.probes
        ):
            raise ValueError("fixture probe targets an inactive service")

        if self.prior is None:
            prior = tuple(np.full(len(self.hypotheses), 1.0 / len(self.hypotheses)))
        else:
            prior = tuple(float(value) for value in self.prior)
            if len(prior) != len(self.hypotheses):
                raise ValueError("fixture prior must match the hypotheses")
            if any(not np.isfinite(value) or value < 0.0 for value in prior):
                raise ValueError("fixture prior must be finite and nonnegative")
            total = float(sum(prior))
            if total <= 0.0:
                raise ValueError("fixture prior must have positive mass")
            prior = tuple(value / total for value in prior)
        object.__setattr__(self, "prior", prior)

    @property
    def evidence_by_id(self) -> Mapping[str, FixtureEvidence]:
        return MappingProxyType({item.evidence_id: item for item in self.evidence})

    @property
    def probes_by_action(self) -> Mapping[int, FixtureProbe]:
        return MappingProxyType({item.action_index: item for item in self.probes})


class FixtureBelief:
    """Exact posterior over a fixture's declared latent states."""

    def __init__(self, definition: FixtureDefinition) -> None:
        self._definition = definition
        self._weights = np.asarray(definition.prior, dtype=np.float64)
        self._records: dict[str, EvidenceRecord] = {}

    @property
    def posterior(self) -> FloatArray:
        result = np.array(self._weights, copy=True)
        result.flags.writeable = False
        return result

    def update(self, records: Iterable[EvidenceRecord]) -> tuple[EvidenceRecord, ...]:
        batch = tuple(records)
        pending: dict[str, EvidenceRecord] = {}
        for record in batch:
            if not isinstance(record, MetricEvidenceRecord):
                raise TypeError("fixture evidence must contain metric records")
            try:
                self._definition.evidence_by_id[record.evidence_id]
            except KeyError as error:
                raise ValueError("record is not defined by this fixture") from error
            existing = pending.get(record.evidence_id, self._records.get(record.evidence_id))
            if existing is not None and existing != record:
                raise EvidenceConflictError(
                    f"evidence ID {record.evidence_id!r} has conflicting values"
                )
            if existing is None:
                pending[record.evidence_id] = record
        if not pending:
            return ()

        updated = np.array(self._weights, copy=True)
        for record in pending.values():
            probabilities = np.asarray(
                self._definition.evidence_by_id[record.evidence_id].true_probabilities,
                dtype=np.float64,
            )
            updated *= probabilities if record.value else 1.0 - probabilities
        total = float(updated.sum())
        if total <= 0.0 or not np.isfinite(total):
            raise InconsistentEvidenceError(
                "fixture evidence is impossible under every valid hypothesis"
            )
        self._weights = updated / total
        self._records.update(pending)
        return tuple(pending.values())

    def diagnostic_quantities(self) -> DiagnosticQuantities:
        service_fault = np.zeros(
            (self._definition.graph.n_services, len(FaultType)),
            dtype=np.float64,
        )
        busy = 0.0
        for weight, hypothesis in zip(
            self._weights,
            self._definition.hypotheses,
            strict=True,
        ):
            service_fault[hypothesis.cause_service, hypothesis.fault_type] += weight
            if hypothesis.workload is Workload.BUSY:
                busy += float(weight)
        service = service_fault.sum(axis=1)
        predicted_service = int(np.argmax(service))
        predicted_fault = FaultType(int(np.argmax(service_fault[predicted_service])))
        service_fault.flags.writeable = False
        service.flags.writeable = False
        return DiagnosticQuantities(
            belief_service_fault=service_fault,
            belief_service=service,
            belief_busy=busy,
            predicted_service=predicted_service,
            predicted_fault=predicted_fault,
        )


class FixtureEnv(gym.Env[Observation, int]):
    """Gymnasium environment for finite fixtures using the simulator schemas."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        definition: FixtureDefinition,
        *,
        initial_budget: int = 8,
        max_probes: int = 6,
        lambda_cost: float = 0.05,
    ) -> None:
        super().__init__()
        if not isinstance(definition, FixtureDefinition):
            raise TypeError("definition must be a FixtureDefinition")
        self.definition = definition
        self.initial_budget = _positive_integer(initial_budget, "initial_budget")
        self.max_probes = _positive_integer(max_probes, "max_probes")
        self.lambda_cost = _bounded_cost(lambda_cost)
        self.action_space = gym.spaces.Discrete(ACTION_SPACE_SIZE)
        self.observation_space = make_observation_space()
        self._hidden_index: int | None = None
        self._values: dict[str, bool] = {}
        self._ledger: EvidenceLedger | None = None
        self._belief: FixtureBelief | None = None
        self._remaining_credits = self.initial_budget
        self._probes_taken = 0
        self._actions: list[int] = []
        self._action_costs: list[int] = []
        self._terminated = False
        self._episode_return = 0.0
        self._observation: Observation | None = None

    @property
    def remaining_credits(self) -> int:
        return self._remaining_credits

    @property
    def evidence_records(self) -> tuple[EvidenceRecord, ...]:
        return () if self._ledger is None else self._ledger.records

    @property
    def posterior(self) -> FloatArray:
        if self._belief is None:
            raise RuntimeError("call reset before reading the posterior")
        return self._belief.posterior

    @property
    def episode_return(self) -> float:
        return self._episode_return

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Observation, dict[str, Any]]:
        super().reset(seed=seed)
        options = {} if options is None else dict(options)
        unknown = set(options) - {"hidden_index"}
        if unknown:
            raise ValueError(f"unknown reset option: {', '.join(sorted(unknown))}")
        if "hidden_index" in options:
            hidden_index = _nonnegative_integer(options["hidden_index"], "hidden_index")
            if hidden_index >= len(self.definition.hypotheses):
                raise ValueError("hidden_index is outside the fixture hypothesis list")
        else:
            hidden_index = int(
                self.np_random.choice(
                    len(self.definition.hypotheses),
                    p=self.definition.prior,
                )
            )
        self._hidden_index = hidden_index
        self._values = {
            evidence.evidence_id: bool(
                self.np_random.random()
                < evidence.true_probabilities[hidden_index]
            )
            for evidence in self.definition.evidence
        }
        self._ledger = EvidenceLedger()
        self._belief = FixtureBelief(self.definition)
        self._remaining_credits = self.initial_budget
        self._probes_taken = 0
        self._actions = []
        self._action_costs = []
        self._terminated = False
        self._episode_return = 0.0
        initial_records = self._records_for(self.definition.initial_evidence_ids)
        self._belief.update(self._ledger.add(initial_records))
        self._observation = self._build_observation()
        return _copy_observation(self._observation), self._info()

    def step(
        self,
        action: int,
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        ledger, belief = self._require_active()
        action = _action_index(action)
        if action == STOP_ACTION_INDEX:
            self._terminated = True
            reward, correct = self._score(step_cost=0, terminal=True)
            self._observation = self._build_observation()
            info = self._info()
            info.update({"action_type": "stop", "diagnosis_correct": correct})
            return _copy_observation(self._observation), reward, True, False, info
        if not bool(self._observation["action_mask"][action]):
            raise ValueError(f"action {action} is invalid for the fixture state")

        probe = self.definition.probes_by_action[action]
        records = self._records_for(probe.evidence_ids)
        unseen = ledger.add(records)
        belief.update(unseen)
        self._remaining_credits -= probe.cost_credits
        self._probes_taken += 1
        self._actions.append(action)
        self._action_costs.append(probe.cost_credits)
        terminal = (
            self._remaining_credits == 0
            or self._probes_taken >= self.max_probes
            or not self._probe_mask().any()
        )
        self._terminated = terminal
        reward, correct = self._score(step_cost=probe.cost_credits, terminal=terminal)
        self._observation = self._build_observation()
        service_id, template = decode_probe_action(action)
        result = ProbeResult(
            action=ProbeAction(
                template.tool,
                f"service-{service_id}",
                template.preset,
            ),
            status=ProbeStatus.OK,
            evidence=records,
            cost=ProbeCost(probe.cost_credits),
            coverage=ProbeCoverage(True),
        )
        info = self._info()
        info.update(
            {
                "action_type": "probe",
                "probe": result.as_dict(),
                "new_evidence_ids": tuple(item.evidence_id for item in unseen),
                "diagnosis_correct": correct,
            }
        )
        return _copy_observation(self._observation), reward, terminal, False, info

    def decision_values(self, *, lookahead: int = 1) -> dict[int, float]:
        """Return exact expected values for STOP and each feasible probe."""
        if self._belief is None or self._ledger is None or self._terminated:
            raise RuntimeError("decision values require an active fixture episode")
        lookahead = _positive_integer(lookahead, "lookahead")
        return self._decision_values(
            self._belief.posterior,
            frozenset(self._ledger.evidence_ids),
            self._remaining_credits,
            self.max_probes - self._probes_taken,
            lookahead,
        )

    def optimal_action(self, *, lookahead: int = 1) -> int:
        """Choose the exact best action, favoring STOP on a value tie."""
        values = self.decision_values(lookahead=lookahead)
        best_action = STOP_ACTION_INDEX
        best_value = values[STOP_ACTION_INDEX]
        for action in sorted(set(values) - {STOP_ACTION_INDEX}):
            if values[action] > best_value + 1e-12:
                best_action = action
                best_value = values[action]
        return best_action

    def collect_action(self, action: int) -> tuple[EvidenceRecord, ...]:
        """Read the fixture backend without changing the ledger or belief."""
        action = _action_index(action)
        try:
            probe = self.definition.probes_by_action[action]
        except KeyError as error:
            raise ValueError("action is not enabled by this fixture") from error
        return self._records_for(probe.evidence_ids)

    def _decision_values(
        self,
        weights: FloatArray,
        seen: frozenset[str],
        remaining_credits: int,
        probes_left: int,
        lookahead: int,
    ) -> dict[int, float]:
        stop_value = float(self._cause_probabilities(weights).max())
        values = {STOP_ACTION_INDEX: stop_value}
        feasible = self._feasible_probes(seen, remaining_credits, probes_left)
        for probe in feasible:
            unseen = tuple(item for item in probe.evidence_ids if item not in seen)
            expected_after = 0.0
            for outcome, probability, posterior in self._outcomes(weights, unseen):
                del outcome
                next_seen = seen.union(unseen)
                next_budget = remaining_credits - probe.cost_credits
                next_probes = probes_left - 1
                next_feasible = self._feasible_probes(
                    next_seen,
                    next_budget,
                    next_probes,
                )
                if lookahead == 1 or not next_feasible:
                    continuation = float(self._cause_probabilities(posterior).max())
                else:
                    continuation = max(
                        self._decision_values(
                            posterior,
                            frozenset(next_seen),
                            next_budget,
                            next_probes,
                            lookahead - 1,
                        ).values()
                    )
                expected_after += probability * continuation
            values[probe.action_index] = (
                expected_after - self.lambda_cost * probe.cost_credits
            )
        return values

    def _outcomes(
        self,
        weights: FloatArray,
        evidence_ids: tuple[str, ...],
    ) -> Iterable[tuple[tuple[bool, ...], float, FloatArray]]:
        for encoded in range(1 << len(evidence_ids)):
            outcome = tuple(bool(encoded & (1 << index)) for index in range(len(evidence_ids)))
            likelihood = np.ones(len(weights), dtype=np.float64)
            for evidence_id, value in zip(evidence_ids, outcome, strict=True):
                probabilities = np.asarray(
                    self.definition.evidence_by_id[evidence_id].true_probabilities,
                    dtype=np.float64,
                )
                likelihood *= probabilities if value else 1.0 - probabilities
            predictive = float(np.dot(weights, likelihood))
            if predictive > 0.0:
                yield outcome, predictive, weights * likelihood / predictive

    def _cause_probabilities(self, weights: FloatArray) -> FloatArray:
        result = np.zeros(self.definition.graph.n_services, dtype=np.float64)
        for weight, hypothesis in zip(weights, self.definition.hypotheses, strict=True):
            result[hypothesis.cause_service] += weight
        return result

    def _feasible_probes(
        self,
        seen: frozenset[str],
        remaining_credits: int,
        probes_left: int,
    ) -> tuple[FixtureProbe, ...]:
        if probes_left <= 0:
            return ()
        return tuple(
            probe
            for probe in self.definition.probes
            if probe.cost_credits <= remaining_credits
            and not set(probe.evidence_ids).issubset(seen)
        )

    def _probe_mask(self) -> NDArray[np.bool_]:
        mask = np.zeros(STOP_ACTION_INDEX, dtype=np.bool_)
        if self._terminated or self._ledger is None:
            return mask
        seen = frozenset(self._ledger.evidence_ids)
        for probe in self._feasible_probes(
            seen,
            self._remaining_credits,
            self.max_probes - self._probes_taken,
        ):
            mask[probe.action_index] = True
        return mask

    def _build_observation(self) -> Observation:
        if self._belief is None or self._ledger is None:
            raise RuntimeError("call reset before building an observation")
        observation = build_observation(
            graph=self.definition.graph,
            belief=self._belief.diagnostic_quantities(),
            evidence_records=self._ledger.records,
            executed_probe_actions=self._actions,
            executed_probe_costs=self._action_costs,
            initial_budget=self.initial_budget,
            remaining_credits=self._remaining_credits,
            probes_taken=self._probes_taken,
            max_probes=self.max_probes,
            lambda_cost=self.lambda_cost,
            available_tools=self._available_tools(),
            terminated=self._terminated,
        )
        action_mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.bool_)
        if not self._terminated:
            action_mask[:STOP_ACTION_INDEX] = self._probe_mask()
            action_mask[STOP_ACTION_INDEX] = True
        observation["action_mask"] = action_mask
        return observation

    def _available_tools(self) -> dict[int, frozenset[ProbeTool]]:
        tools = {service: set() for service in range(self.definition.graph.n_services)}
        for probe in self.definition.probes:
            service, template = decode_probe_action(probe.action_index)
            tools[service].add(template.tool)
        return {service: frozenset(values) for service, values in tools.items()}

    def _records_for(self, evidence_ids: Iterable[str]) -> tuple[EvidenceRecord, ...]:
        return tuple(
            self.definition.evidence_by_id[evidence_id].record(self._values[evidence_id])
            for evidence_id in evidence_ids
        )

    def _score(self, *, step_cost: int, terminal: bool) -> tuple[float, bool | None]:
        if self._belief is None or self._hidden_index is None:
            raise RuntimeError("call reset before scoring")
        correct = None
        utility = 0.0
        if terminal:
            predicted = self._belief.diagnostic_quantities().predicted_service
            actual = self.definition.hypotheses[self._hidden_index].cause_service
            correct = predicted == actual
            utility = float(correct)
        reward = utility - self.lambda_cost * step_cost
        self._episode_return += reward
        return reward, correct

    def _info(self) -> dict[str, Any]:
        if self._belief is None:
            raise RuntimeError("call reset before reading info")
        diagnostics = self._belief.diagnostic_quantities()
        return {
            "fixture": self.definition.name,
            "n_services": self.definition.graph.n_services,
            "remaining_credits": self._remaining_credits,
            "credits_spent": self.initial_budget - self._remaining_credits,
            "probes_taken": self._probes_taken,
            "predicted_service": diagnostics.predicted_service,
            "predicted_fault": diagnostics.predicted_fault.name.lower(),
            "episode_return": self._episode_return,
        }

    def _require_active(self) -> tuple[EvidenceLedger, FixtureBelief]:
        if self._ledger is None or self._belief is None or self._observation is None:
            raise RuntimeError("call reset before using the fixture environment")
        if self._terminated:
            raise RuntimeError("fixture episode has terminated; call reset before step")
        return self._ledger, self._belief


def known_answer_fixture() -> FixtureDefinition:
    evidence = _binary_evidence((1.0, 0.0))
    return _two_cause_fixture(
        "known_answer",
        evidence=(evidence,),
        probes=(),
        initial_evidence_ids=(evidence.evidence_id,),
    )


def one_perfect_probe_fixture() -> FixtureDefinition:
    evidence = _binary_evidence((1.0, 0.0))
    distractor = FixtureEvidence(
        1,
        MetricCategory.CPU_HIGH,
        0,
        (0.5, 0.5),
    )
    return _two_cause_fixture(
        "one_perfect_probe",
        evidence=(evidence, distractor),
        probes=(
            FixtureProbe(
                probe_action_index(0, "metrics.quick"),
                1,
                (evidence.evidence_id,),
            ),
            FixtureProbe(
                probe_action_index(1, "metrics.quick"),
                1,
                (distractor.evidence_id,),
            ),
        ),
    )


def useless_evidence_fixture() -> FixtureDefinition:
    evidence = _binary_evidence((0.5, 0.5))
    return _two_cause_fixture(
        "useless_evidence",
        evidence=(evidence,),
        probes=(FixtureProbe(probe_action_index(0, "metrics.quick"), 1, (evidence.evidence_id,)),),
    )


def duplicate_evidence_fixture() -> FixtureDefinition:
    evidence = _binary_evidence((0.8, 0.2))
    return _two_cause_fixture(
        "duplicate_evidence",
        evidence=(evidence,),
        probes=(
            FixtureProbe(probe_action_index(0, "metrics.quick"), 1, (evidence.evidence_id,)),
            FixtureProbe(probe_action_index(0, "metrics.detailed"), 2, (evidence.evidence_id,)),
        ),
    )


def expensive_discriminator_fixture(*, cost_credits: int) -> FixtureDefinition:
    evidence = _binary_evidence((1.0, 0.0))
    return _two_cause_fixture(
        "expensive_discriminator",
        evidence=(evidence,),
        probes=(FixtureProbe(probe_action_index(0, "metrics.quick"), cost_credits, (evidence.evidence_id,)),),
    )


def scope_choice_fixture(*, prior: tuple[float, float] = (0.5, 0.5)) -> FixtureDefinition:
    quick = _binary_evidence((0.75, 0.25), reading_index=0)
    extra = _binary_evidence((0.90, 0.10), reading_index=1)
    return _two_cause_fixture(
        "scope_choice",
        evidence=(quick, extra),
        probes=(
            FixtureProbe(probe_action_index(0, "metrics.quick"), 1, (quick.evidence_id,)),
            FixtureProbe(
                probe_action_index(0, "metrics.detailed"),
                2,
                (quick.evidence_id, extra.evidence_id),
            ),
        ),
        prior=prior,
    )


def last_credit_evidence_fixture() -> FixtureDefinition:
    evidence = _binary_evidence((1.0, 0.0))
    return _two_cause_fixture(
        "last_credit_evidence",
        evidence=(evidence,),
        probes=(FixtureProbe(probe_action_index(0, "metrics.quick"), 1, (evidence.evidence_id,)),),
    )


def no_affordable_probe_fixture() -> FixtureDefinition:
    evidence = _binary_evidence((1.0, 0.0))
    return _two_cause_fixture(
        "no_affordable_probe",
        evidence=(evidence,),
        probes=(FixtureProbe(probe_action_index(0, "metrics.detailed"), 2, (evidence.evidence_id,)),),
    )


def complementarity_fixture() -> FixtureDefinition:
    graph = _two_service_graph()
    hypotheses = (
        FixtureHypothesis("equal-00", 0),
        FixtureHypothesis("equal-11", 0),
        FixtureHypothesis("different-01", 1),
        FixtureHypothesis("different-10", 1),
    )
    first = FixtureEvidence(0, MetricCategory.CPU_HIGH, 0, (0.0, 1.0, 0.0, 1.0))
    second = FixtureEvidence(1, MetricCategory.CPU_HIGH, 0, (0.0, 1.0, 1.0, 0.0))
    return FixtureDefinition(
        name="two_probe_complementarity",
        graph=graph,
        hypotheses=hypotheses,
        evidence=(first, second),
        probes=(
            FixtureProbe(probe_action_index(0, "metrics.quick"), 1, (first.evidence_id,)),
            FixtureProbe(probe_action_index(1, "metrics.quick"), 1, (second.evidence_id,)),
        ),
    )


def fixture_catalog() -> Mapping[str, FixtureDefinition]:
    """Return the standard step 8.3 fixtures with stable names."""
    fixtures = (
        known_answer_fixture(),
        one_perfect_probe_fixture(),
        useless_evidence_fixture(),
        duplicate_evidence_fixture(),
        expensive_discriminator_fixture(cost_credits=5),
        scope_choice_fixture(),
        last_credit_evidence_fixture(),
        no_affordable_probe_fixture(),
        complementarity_fixture(),
    )
    return MappingProxyType({item.name: item for item in fixtures})


def _two_cause_fixture(
    name: str,
    *,
    evidence: tuple[FixtureEvidence, ...],
    probes: tuple[FixtureProbe, ...],
    prior: tuple[float, float] = (0.5, 0.5),
    initial_evidence_ids: tuple[str, ...] = (),
) -> FixtureDefinition:
    return FixtureDefinition(
        name=name,
        graph=_two_service_graph(),
        hypotheses=(
            FixtureHypothesis("cause-0", 0),
            FixtureHypothesis("cause-1", 1),
        ),
        evidence=evidence,
        probes=probes,
        prior=prior,
        initial_evidence_ids=initial_evidence_ids,
    )


def _binary_evidence(
    probabilities: tuple[float, float],
    *,
    reading_index: int = 0,
) -> FixtureEvidence:
    return FixtureEvidence(
        0,
        MetricCategory.CPU_HIGH,
        reading_index,
        probabilities,
    )


def _two_service_graph() -> DependencyGraph:
    return DependencyGraph.from_edges(2, [(0, 1)], entry_service=0)


def _copy_observation(observation: Observation) -> Observation:
    return {key: np.array(value, copy=True) for key, value in observation.items()}


def _action_index(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("action must be an integer")
    value = int(value)
    if not 0 <= value < ACTION_SPACE_SIZE:
        raise ValueError(f"action must be between 0 and {ACTION_SPACE_SIZE - 1}")
    return value


def _positive_integer(value: object, name: str) -> int:
    return_value = _nonnegative_integer(value, name)
    if return_value == 0:
        raise ValueError(f"{name} must be positive")
    return return_value


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _bounded_cost(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("lambda_cost must be a real number")
    value = float(value)
    if not np.isfinite(value) or not 0.0 <= value <= 0.20:
        raise ValueError("lambda_cost must be between 0 and 0.20")
    return value
