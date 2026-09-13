"""Public evidence likelihood calculations shared with world generation."""

from __future__ import annotations

from numbers import Integral

import numpy as np
from numpy.typing import NDArray

from rca_sim.contracts import (
    FaultType,
    LogCategory,
    MetricCategory,
    RelationshipKind,
    ServiceRelationship,
    Workload,
)


FloatArray = NDArray[np.float64]

_CAUSE_METRIC_PROBABILITIES: dict[FaultType, FloatArray] = {
    FaultType.CPU: np.array([0.85, 0.15, 0.75], dtype=np.float64),
    FaultType.MEMORY: np.array([0.15, 0.85, 0.75], dtype=np.float64),
    FaultType.NETWORK_DELAY: np.array([0.15, 0.15, 0.90], dtype=np.float64),
}
_UNRELATED_METRIC_PROBABILITIES = np.array(
    [0.10, 0.10, 0.15], dtype=np.float64
)
_BUSY_METRIC_INCREMENTS = np.array([0.10, 0.10, 0.05], dtype=np.float64)

_RESOURCE_CAUSE_LOG_PROBABILITIES = np.array(
    [0.10, 0.65, 0.15, 0.10], dtype=np.float64
)
_NETWORK_CAUSE_LOG_PROBABILITIES = np.array(
    [0.10, 0.10, 0.70, 0.10], dtype=np.float64
)
_DIRECT_CALLER_LOG_PROBABILITIES = np.array(
    [0.25, 0.10, 0.55, 0.10], dtype=np.float64
)
_UNRELATED_LOG_PROBABILITIES = np.array(
    [0.75, 0.05, 0.10, 0.10], dtype=np.float64
)
_BUSY_BACKGROUND_LOG_PROBABILITIES = np.array(
    [0.25, 0.35, 0.30, 0.10], dtype=np.float64
)


def metric_high_probabilities(
    relationship: ServiceRelationship,
    fault_type: FaultType,
    workload: Workload,
) -> FloatArray:
    """Return CPU, memory, and latency-high probabilities."""
    relationship = _require_relationship(relationship)
    fault_type = _coerce_fault_type(fault_type)
    workload = _coerce_workload(workload)

    if relationship.kind is RelationshipKind.CAUSE:
        probabilities = _CAUSE_METRIC_PROBABILITIES[fault_type].copy()
    elif relationship.kind is RelationshipKind.AFFECTED_CALLER:
        distance = _affected_distance(relationship)
        latency_probability = 0.15 + 0.55 * 0.7 ** (distance - 1)
        probabilities = np.array(
            [0.10, 0.10, latency_probability], dtype=np.float64
        )
    else:
        probabilities = _UNRELATED_METRIC_PROBABILITIES.copy()

    if workload is Workload.BUSY:
        probabilities = np.minimum(probabilities + _BUSY_METRIC_INCREMENTS, 0.95)

    probabilities.flags.writeable = False
    return probabilities


def metric_high_probability(
    relationship: ServiceRelationship,
    fault_type: FaultType,
    workload: Workload,
    metric: MetricCategory,
) -> float:
    """Return the Bernoulli probability for one metric category."""
    if isinstance(metric, bool) or not isinstance(metric, Integral):
        raise TypeError("metric category must be an integer encoding")
    try:
        metric = MetricCategory(int(metric))
    except (TypeError, ValueError) as error:
        raise ValueError("metric category is not supported") from error
    return float(metric_high_probabilities(relationship, fault_type, workload)[metric])


def log_category_probabilities(
    relationship: ServiceRelationship,
    fault_type: FaultType,
    workload: Workload,
) -> FloatArray:
    """Return probabilities for the four log categories."""
    relationship = _require_relationship(relationship)
    fault_type = _coerce_fault_type(fault_type)
    workload = _coerce_workload(workload)

    if relationship.kind is RelationshipKind.CAUSE:
        if fault_type is FaultType.NETWORK_DELAY:
            probabilities = _NETWORK_CAUSE_LOG_PROBABILITIES.copy()
        else:
            probabilities = _RESOURCE_CAUSE_LOG_PROBABILITIES.copy()
    elif relationship.kind is RelationshipKind.AFFECTED_CALLER:
        distance = _affected_distance(relationship)
        caller_weight = 0.7 ** (distance - 1)
        probabilities = (
            caller_weight * _DIRECT_CALLER_LOG_PROBABILITIES
            + (1.0 - caller_weight) * _UNRELATED_LOG_PROBABILITIES
        )
    else:
        probabilities = _UNRELATED_LOG_PROBABILITIES.copy()

    if workload is Workload.BUSY:
        probabilities = (
            0.85 * probabilities + 0.15 * _BUSY_BACKGROUND_LOG_PROBABILITIES
        )

    probabilities.flags.writeable = False
    return probabilities


def log_category_probability(
    relationship: ServiceRelationship,
    fault_type: FaultType,
    workload: Workload,
    category: LogCategory,
) -> float:
    """Return the categorical probability for one log value."""
    if isinstance(category, bool) or not isinstance(category, Integral):
        raise TypeError("log category must be an integer encoding")
    try:
        category = LogCategory(int(category))
    except (TypeError, ValueError) as error:
        raise ValueError("log category is not supported") from error
    return float(log_category_probabilities(relationship, fault_type, workload)[category])


def _require_relationship(relationship: ServiceRelationship) -> ServiceRelationship:
    if not isinstance(relationship, ServiceRelationship):
        raise TypeError("relationship must be a ServiceRelationship")
    return relationship


def _coerce_fault_type(fault_type: FaultType) -> FaultType:
    if isinstance(fault_type, bool) or not isinstance(fault_type, Integral):
        raise TypeError("fault type must be an integer encoding")
    try:
        return FaultType(int(fault_type))
    except (TypeError, ValueError) as error:
        raise ValueError("fault type is not supported") from error


def _coerce_workload(workload: Workload) -> Workload:
    if isinstance(workload, bool) or not isinstance(workload, Integral):
        raise TypeError("workload must be an integer encoding")
    try:
        return Workload(int(workload))
    except (TypeError, ValueError) as error:
        raise ValueError("workload is not supported") from error


def _affected_distance(relationship: ServiceRelationship) -> int:
    if relationship.distance is None:
        raise ValueError("affected caller relationship has no distance")
    return relationship.distance
