"""Construction and schema for policy-visible simulator observations."""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
from numbers import Integral, Real
from types import MappingProxyType
from typing import TypeAlias

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from rca_sim.belief import DiagnosticQuantities, ExactBeliefEstimator
from rca_sim.contracts import (
    EvidenceRecord,
    FaultType,
    LogCategory,
    LogEvidenceRecord,
    MetricCategory,
    MetricEvidenceRecord,
)
from rca_sim.graph import DependencyGraph
from rca_sim.tools import (
    ACTION_SPACE_SIZE,
    MAX_SERVICE_SLOTS,
    PROBE_TEMPLATE_NAMES,
    ProbeTemplate,
    ProbeTool,
    construct_action_mask,
    decode_probe_action,
)


Float32Array = NDArray[np.float32]
BoolArray = NDArray[np.bool_]
Observation: TypeAlias = dict[str, Float32Array | BoolArray]

OBSERVATION_SCHEMA_VERSION = "sim_v0"
NODE_FEATURE_COUNT = 42
GLOBAL_FEATURE_COUNT = 10

NODE_FEATURE_NAMES = (
    "is_entry_service",
    "in_degree_div_9",
    "out_degree_div_9",
    "distance_from_entry_div_9",
    "cause_service_probability",
    *(f"fault_probability_given_service.{fault.name.lower()}" for fault in FaultType),
    *(
        f"metric_value.{metric.evidence_name}.{reading_index}"
        for metric in MetricCategory
        for reading_index in range(2)
    ),
    *(
        f"metric_present.{metric.evidence_name}.{reading_index}"
        for metric in MetricCategory
        for reading_index in range(2)
    ),
    *(
        f"log_value.{reading_index}.{category.name.lower()}"
        for reading_index in range(3)
        for category in LogCategory
    ),
    *(f"log_present.{reading_index}" for reading_index in range(3)),
    "tool_available.metrics",
    "tool_available.logs",
    *(f"template_executed.{name}" for name in PROBE_TEMPLATE_NAMES),
    "credits_spent_div_initial_budget",
)

GLOBAL_FEATURE_NAMES = (
    "initial_budget_div_16",
    "remaining_credits_div_16",
    "remaining_credits_div_initial_budget",
    "remaining_probes_div_max_probes",
    "service_fault_entropy_div_log_hypothesis_count",
    "largest_service_probability",
    "service_probability_top_two_gap",
    "busy_workload_probability",
    "active_service_count_div_10",
    "lambda_cost_div_0_20",
)

OBSERVATION_SHAPES = MappingProxyType(
    {
        "node_features": (MAX_SERVICE_SLOTS, NODE_FEATURE_COUNT),
        "node_mask": (MAX_SERVICE_SLOTS,),
        "adjacency": (MAX_SERVICE_SLOTS, MAX_SERVICE_SLOTS),
        "global_features": (GLOBAL_FEATURE_COUNT,),
        "action_mask": (ACTION_SPACE_SIZE,),
    }
)

OBSERVATION_SCALES = MappingProxyType(
    {
        "degree": 9.0,
        "distance_from_entry": 9.0,
        "initial_budget": 16.0,
        "remaining_credits": 16.0,
        "active_service_count": 10.0,
        "lambda_cost": 0.20,
    }
)

if len(NODE_FEATURE_NAMES) != NODE_FEATURE_COUNT:  # pragma: no cover
    raise RuntimeError("node feature schema does not contain 42 columns")
if len(GLOBAL_FEATURE_NAMES) != GLOBAL_FEATURE_COUNT:  # pragma: no cover
    raise RuntimeError("global feature schema does not contain 10 columns")


def make_observation_space() -> gym.spaces.Dict:
    """Return the fixed Gymnasium observation space for schema ``sim_v0``."""
    return gym.spaces.Dict(
        {
            "node_features": gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=OBSERVATION_SHAPES["node_features"],
                dtype=np.float32,
            ),
            "node_mask": gym.spaces.Box(
                low=0,
                high=1,
                shape=OBSERVATION_SHAPES["node_mask"],
                dtype=np.bool_,
            ),
            "adjacency": gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=OBSERVATION_SHAPES["adjacency"],
                dtype=np.float32,
            ),
            "global_features": gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=OBSERVATION_SHAPES["global_features"],
                dtype=np.float32,
            ),
            "action_mask": gym.spaces.Box(
                low=0,
                high=1,
                shape=OBSERVATION_SHAPES["action_mask"],
                dtype=np.bool_,
            ),
        }
    )


def build_observation(
    *,
    graph: DependencyGraph,
    belief: ExactBeliefEstimator | DiagnosticQuantities,
    evidence_records: Iterable[EvidenceRecord],
    executed_probe_actions: Iterable[int],
    initial_budget: int,
    remaining_credits: int,
    probes_taken: int,
    max_probes: int,
    lambda_cost: float,
    available_tools: Mapping[int, Collection[ProbeTool]] | None = None,
    terminated: bool = False,
) -> Observation:
    """Build a fresh policy observation from public episode state.

    Hidden incident fields are deliberately absent. Evidence coverage comes
    only from records in ``evidence_records``. Template flags and per-service
    spend come only from actions that the environment says it executed.
    """
    if not isinstance(graph, DependencyGraph):
        raise TypeError("graph must be a DependencyGraph")
    if graph.n_services > MAX_SERVICE_SLOTS:
        raise ValueError(f"graph cannot contain more than {MAX_SERVICE_SLOTS} services")

    diagnostics = _resolve_diagnostics(belief, graph.n_services)
    records = _validate_evidence_records(evidence_records, graph.n_services)
    action_history = _validate_action_history(
        executed_probe_actions,
        graph.n_services,
    )
    initial_budget = _positive_integer(initial_budget, "initial_budget")
    remaining_credits = _nonnegative_integer(
        remaining_credits,
        "remaining_credits",
    )
    if remaining_credits > initial_budget:
        raise ValueError("remaining_credits cannot exceed initial_budget")
    probes_taken = _nonnegative_integer(probes_taken, "probes_taken")
    max_probes = _positive_integer(max_probes, "max_probes")
    if probes_taken > max_probes:
        raise ValueError("probes_taken cannot exceed max_probes")
    if probes_taken != len(action_history):
        raise ValueError("probes_taken must equal the executed probe action count")
    lambda_cost = _bounded_real(lambda_cost, "lambda_cost", upper=0.20)
    terminated = _boolean(terminated, "terminated")

    tools_by_service = _resolve_available_tools(graph.n_services, available_tools)
    node_features = np.zeros(OBSERVATION_SHAPES["node_features"], dtype=np.float32)
    node_mask = np.zeros(OBSERVATION_SHAPES["node_mask"], dtype=np.bool_)
    adjacency = np.zeros(OBSERVATION_SHAPES["adjacency"], dtype=np.float32)

    node_mask[: graph.n_services] = True
    adjacency[: graph.n_services, : graph.n_services] = graph.adjacency

    metric_values = np.zeros((graph.n_services, len(MetricCategory), 2), dtype=np.float32)
    metric_present = np.zeros_like(metric_values)
    log_values = np.zeros((graph.n_services, 3, len(LogCategory)), dtype=np.float32)
    log_present = np.zeros((graph.n_services, 3), dtype=np.float32)
    for record in records:
        if isinstance(record, MetricEvidenceRecord):
            metric_values[record.service_id, record.metric, record.reading_index] = float(
                record.value
            )
            metric_present[record.service_id, record.metric, record.reading_index] = 1.0
        else:
            log_values[
                record.service_id,
                record.reading_index,
                record.value,
            ] = 1.0
            log_present[record.service_id, record.reading_index] = 1.0

    executed_flags = np.zeros(
        (graph.n_services, len(PROBE_TEMPLATE_NAMES)),
        dtype=np.float32,
    )
    credits_spent = np.zeros(graph.n_services, dtype=np.float32)
    for service_id, template in action_history:
        template_index = PROBE_TEMPLATE_NAMES.index(template.name)
        executed_flags[service_id, template_index] = 1.0
        credits_spent[service_id] += template.cost_credits

    for service_id in range(graph.n_services):
        service_probability = float(diagnostics.belief_service[service_id])
        conditional_fault = np.zeros(len(FaultType), dtype=np.float64)
        if service_probability > 0.0:
            conditional_fault = (
                diagnostics.belief_service_fault[service_id] / service_probability
            )

        row = node_features[service_id]
        row[0] = float(service_id == graph.entry_service)
        row[1] = graph.in_degree[service_id] / OBSERVATION_SCALES["degree"]
        row[2] = graph.out_degree[service_id] / OBSERVATION_SCALES["degree"]
        row[3] = (
            graph.distances[graph.entry_service, service_id]
            / OBSERVATION_SCALES["distance_from_entry"]
        )
        row[4] = service_probability
        row[5:8] = conditional_fault
        row[8:14] = metric_values[service_id].reshape(-1)
        row[14:20] = metric_present[service_id].reshape(-1)
        row[20:32] = log_values[service_id].reshape(-1)
        row[32:35] = log_present[service_id]
        row[35] = float(ProbeTool.METRICS in tools_by_service[service_id])
        row[36] = float(ProbeTool.LOGS in tools_by_service[service_id])
        row[37:41] = executed_flags[service_id]
        row[41] = credits_spent[service_id] / initial_budget

    service_fault = np.asarray(diagnostics.belief_service_fault, dtype=np.float64)
    positive = service_fault[service_fault > 0.0]
    entropy = -float(np.sum(positive * np.log(positive)))
    hypothesis_count = graph.n_services * len(FaultType)
    normalized_entropy = entropy / np.log(hypothesis_count)
    sorted_service = np.sort(np.asarray(diagnostics.belief_service, dtype=np.float64))
    largest_service = float(sorted_service[-1])
    top_two_gap = largest_service - float(sorted_service[-2]) if graph.n_services > 1 else largest_service

    global_features = np.asarray(
        [
            initial_budget / OBSERVATION_SCALES["initial_budget"],
            remaining_credits / OBSERVATION_SCALES["remaining_credits"],
            remaining_credits / initial_budget,
            (max_probes - probes_taken) / max_probes,
            normalized_entropy,
            largest_service,
            top_two_gap,
            diagnostics.belief_busy,
            graph.n_services / OBSERVATION_SCALES["active_service_count"],
            lambda_cost / OBSERVATION_SCALES["lambda_cost"],
        ],
        dtype=np.float32,
    )

    seen_ids = tuple(record.evidence_id for record in records)
    action_mask = construct_action_mask(
        n_services=graph.n_services,
        remaining_credits=remaining_credits,
        seen_evidence_ids=seen_ids,
        available_tools=available_tools,
        terminated=terminated,
    )
    observation: Observation = {
        "node_features": node_features,
        "node_mask": node_mask,
        "adjacency": adjacency,
        "global_features": global_features,
        "action_mask": np.array(action_mask, dtype=np.bool_, copy=True),
    }
    if not make_observation_space().contains(observation):
        raise ValueError(
            "episode values exceed observation schema sim_v0 bounds; "
            "revise the schema version instead of clipping them"
        )
    return observation


def _resolve_diagnostics(
    belief: ExactBeliefEstimator | DiagnosticQuantities,
    n_services: int,
) -> DiagnosticQuantities:
    if isinstance(belief, ExactBeliefEstimator):
        diagnostics = belief.diagnostic_quantities()
    elif isinstance(belief, DiagnosticQuantities):
        diagnostics = belief
    else:
        raise TypeError("belief must be an ExactBeliefEstimator or DiagnosticQuantities")
    if diagnostics.belief_service.shape != (n_services,):
        raise ValueError("belief service count does not match graph")
    if diagnostics.belief_service_fault.shape != (n_services, len(FaultType)):
        raise ValueError("service/fault belief shape does not match graph")
    return diagnostics


def _validate_evidence_records(
    records: Iterable[EvidenceRecord],
    n_services: int,
) -> tuple[EvidenceRecord, ...]:
    try:
        batch = tuple(records)
    except TypeError as error:
        raise TypeError("evidence_records must be iterable") from error
    by_id: dict[str, EvidenceRecord] = {}
    for record in batch:
        if not isinstance(record, (MetricEvidenceRecord, LogEvidenceRecord)):
            raise TypeError("evidence_records must contain only evidence records")
        if record.service_id >= n_services:
            raise ValueError("evidence record refers to an inactive service")
        existing = by_id.get(record.evidence_id)
        if existing is not None and existing != record:
            raise ValueError("evidence_records contains a conflicting evidence ID")
        by_id.setdefault(record.evidence_id, record)
    return tuple(by_id.values())


def _validate_action_history(
    actions: Iterable[int],
    n_services: int,
) -> tuple[tuple[int, ProbeTemplate], ...]:
    try:
        action_indices = tuple(actions)
    except TypeError as error:
        raise TypeError("executed_probe_actions must be iterable") from error
    decoded = []
    for action_index in action_indices:
        service_id, template = decode_probe_action(action_index)
        if service_id >= n_services:
            raise ValueError("executed probe action targets an inactive service")
        decoded.append((service_id, template))
    return tuple(decoded)


def _resolve_available_tools(
    n_services: int,
    available_tools: Mapping[int, Collection[ProbeTool]] | None,
) -> dict[int, frozenset[ProbeTool]]:
    if available_tools is None:
        all_tools = frozenset(ProbeTool)
        return {service_id: all_tools for service_id in range(n_services)}
    construct_action_mask(
        n_services=n_services,
        remaining_credits=0,
        seen_evidence_ids=(),
        available_tools=available_tools,
    )
    return {
        service_id: frozenset(available_tools.get(service_id, ()))
        for service_id in range(n_services)
    }


def _positive_integer(value: int, name: str) -> int:
    value = _nonnegative_integer(value, name)
    if value == 0:
        raise ValueError(f"{name} must be positive")
    return value


def _nonnegative_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _bounded_real(value: float, name: str, *, upper: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not np.isfinite(value) or not 0.0 <= value <= upper:
        raise ValueError(f"{name} must be between 0 and {upper}")
    return value


def _boolean(value: bool, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be boolean")
    return bool(value)
