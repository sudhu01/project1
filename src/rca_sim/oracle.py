"""Exact planners that use the simulator's public probability model."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from itertools import product
from numbers import Real
from types import MappingProxyType

import numpy as np

from rca_sim.belief import record_likelihoods
from rca_sim.contracts import (
    FaultType,
    LogCategory,
    LogEvidenceRecord,
    MetricCategory,
    MetricEvidenceRecord,
    Workload,
)
from rca_sim.graph import DependencyGraph
from rca_sim.tools import (
    ACTION_SPACE_SIZE,
    STOP_ACTION_INDEX,
    ProbeTemplate,
    ProbeTool,
    decode_probe_action,
)


@dataclass(frozen=True, slots=True)
class OneStepPlan:
    """Exact STOP and probe values for one public investigation state."""

    stop_value: float
    action_values: Mapping[int, float]
    outcome_counts: Mapping[int, int]
    best_action: int


def one_step_plan(
    *,
    graph: DependencyGraph,
    posterior: np.ndarray,
    seen_evidence_ids: Collection[str],
    action_mask: np.ndarray,
    lambda_cost: float,
) -> OneStepPlan:
    """Enumerate every unseen output for each feasible probe exactly."""
    if not isinstance(graph, DependencyGraph):
        raise TypeError("graph must be a DependencyGraph")
    weights = _validated_posterior(posterior, graph.n_services)
    seen = _validated_evidence_ids(seen_evidence_ids)
    mask = np.asarray(action_mask)
    if mask.shape != (ACTION_SPACE_SIZE,) or mask.dtype != np.bool_:
        raise ValueError(
            f"action_mask must be a boolean array with shape ({ACTION_SPACE_SIZE},)"
        )
    if not mask[STOP_ACTION_INDEX]:
        raise ValueError("STOP must be feasible in an active planning state")
    if isinstance(lambda_cost, bool) or not isinstance(lambda_cost, Real):
        raise TypeError("lambda_cost must be a real number")
    cost_weight = float(lambda_cost)
    if not np.isfinite(cost_weight) or cost_weight < 0.0:
        raise ValueError("lambda_cost must be finite and nonnegative")

    stop_value = float(weights.sum(axis=(1, 2)).max())
    values: dict[int, float] = {STOP_ACTION_INDEX: stop_value}
    outcome_counts: dict[int, int] = {STOP_ACTION_INDEX: 1}
    best_action = STOP_ACTION_INDEX
    best_value = stop_value

    for action in np.flatnonzero(mask[:STOP_ACTION_INDEX]):
        action_index = int(action)
        service_id, template = decode_probe_action(action_index)
        outcomes = _unseen_record_outcomes(service_id, template, seen)
        if not outcomes:
            raise ValueError("feasible probe does not reveal a new record")

        expected_accuracy = 0.0
        outcome_count = 0
        for records in product(*outcomes):
            likelihood = np.ones_like(weights)
            for record in records:
                likelihood *= record_likelihoods(graph, record)
            predictive_probability = float(np.sum(weights * likelihood))
            if predictive_probability <= 0.0:
                continue
            next_posterior = weights * likelihood / predictive_probability
            next_accuracy = float(next_posterior.sum(axis=(1, 2)).max())
            expected_accuracy += predictive_probability * next_accuracy
            outcome_count += 1

        value = expected_accuracy - cost_weight * template.cost_credits
        values[action_index] = value
        outcome_counts[action_index] = outcome_count
        if value > best_value + 1e-12:
            best_action = action_index
            best_value = value

    return OneStepPlan(
        stop_value=stop_value,
        action_values=MappingProxyType(values),
        outcome_counts=MappingProxyType(outcome_counts),
        best_action=best_action,
    )


def _unseen_record_outcomes(
    service_id: int,
    template: ProbeTemplate,
    seen: frozenset[str],
) -> tuple[tuple[MetricEvidenceRecord | LogEvidenceRecord, ...], ...]:
    outcomes: list[tuple[MetricEvidenceRecord | LogEvidenceRecord, ...]] = []
    if template.tool is ProbeTool.METRICS:
        for reading_index in template.reading_indices:
            for metric in MetricCategory:
                evidence_id = MetricEvidenceRecord(
                    service_id,
                    metric,
                    reading_index,
                    False,
                ).evidence_id
                if evidence_id not in seen:
                    outcomes.append(
                        tuple(
                            MetricEvidenceRecord(
                                service_id,
                                metric,
                                reading_index,
                                value,
                            )
                            for value in (False, True)
                        )
                    )
    else:
        for reading_index in template.reading_indices:
            evidence_id = LogEvidenceRecord(
                service_id,
                reading_index,
                LogCategory.NO_RELEVANT_ERROR,
            ).evidence_id
            if evidence_id not in seen:
                outcomes.append(
                    tuple(
                        LogEvidenceRecord(service_id, reading_index, category)
                        for category in LogCategory
                    )
                )
    return tuple(outcomes)


def _validated_posterior(posterior: np.ndarray, n_services: int) -> np.ndarray:
    weights = np.asarray(posterior, dtype=np.float64)
    expected_shape = (n_services, len(FaultType), len(Workload))
    if weights.shape != expected_shape:
        raise ValueError(f"posterior must have shape {expected_shape}")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("posterior must be finite and nonnegative")
    total = float(weights.sum())
    if not np.isclose(total, 1.0, atol=1e-12):
        raise ValueError("posterior must sum to one")
    return np.array(weights, copy=True)


def _validated_evidence_ids(values: Collection[str]) -> frozenset[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Collection):
        raise TypeError("seen_evidence_ids must be a collection of strings")
    if any(not isinstance(value, str) for value in values):
        raise TypeError("seen_evidence_ids must contain strings")
    return frozenset(values)
