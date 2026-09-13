"""Hidden incident state and generation."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.typing import NDArray

from rca_sim.contracts import (
    FaultType,
    LOG_READINGS_PER_SERVICE,
    LogCategory,
    METRIC_READINGS_PER_CATEGORY,
    MetricCategory,
    RelationshipKind,
    ServiceRelationship,
    Workload,
)
from rca_sim.graph import DependencyGraph, classify_service_relationship
from rca_sim.likelihoods import (
    log_category_probabilities,
    metric_high_probabilities,
)


BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class HiddenHypothesis:
    """The private service, fault, and workload state for one incident."""

    cause_service: int
    fault_type: FaultType
    workload: Workload

    def __post_init__(self) -> None:
        if isinstance(self.cause_service, bool) or not isinstance(
            self.cause_service, Integral
        ):
            raise TypeError("cause_service must be an integer")
        if self.cause_service < 0:
            raise ValueError("cause_service must be nonnegative")
        if isinstance(self.fault_type, bool) or not isinstance(
            self.fault_type, Integral
        ):
            raise TypeError("fault_type must be an integer encoding")
        if isinstance(self.workload, bool) or not isinstance(self.workload, Integral):
            raise TypeError("workload must be an integer encoding")

        object.__setattr__(self, "cause_service", int(self.cause_service))
        try:
            object.__setattr__(self, "fault_type", FaultType(int(self.fault_type)))
        except (TypeError, ValueError) as error:
            raise ValueError("fault_type is not supported") from error
        try:
            object.__setattr__(self, "workload", Workload(int(self.workload)))
        except (TypeError, ValueError) as error:
            raise ValueError("workload is not supported") from error


@dataclass(frozen=True, slots=True)
class SyntheticObservations:
    """Persistent metric and log samples for one hidden incident."""

    metric_readings: BoolArray
    log_readings: IntArray

    def __post_init__(self) -> None:
        metrics = np.asarray(self.metric_readings)
        logs = np.asarray(self.log_readings)
        expected_metric_tail = (len(MetricCategory), METRIC_READINGS_PER_CATEGORY)
        expected_log_width = LOG_READINGS_PER_SERVICE

        if metrics.ndim != 3 or metrics.shape[1:] != expected_metric_tail:
            raise ValueError("metric_readings must have shape (services, 3, 2)")
        if metrics.shape[0] < 1:
            raise ValueError("observations must contain at least one service")
        if metrics.dtype != np.bool_:
            raise TypeError("metric_readings must use boolean values")
        if logs.ndim != 2 or logs.shape[1] != expected_log_width:
            raise ValueError("log_readings must have shape (services, 3)")
        if logs.shape[0] != metrics.shape[0]:
            raise ValueError("metric and log service counts must match")
        if not np.issubdtype(logs.dtype, np.integer):
            raise TypeError("log_readings must use integer category values")
        if np.any(logs < 0) or np.any(logs >= len(LogCategory)):
            raise ValueError("log_readings contain an unsupported category")

        metrics = np.array(metrics, dtype=np.bool_, copy=True)
        logs = np.array(logs, dtype=np.int64, copy=True)
        metrics.flags.writeable = False
        logs.flags.writeable = False
        object.__setattr__(self, "metric_readings", metrics)
        object.__setattr__(self, "log_readings", logs)

    @property
    def n_services(self) -> int:
        return int(self.metric_readings.shape[0])


def enumerate_hidden_hypotheses(n_services: int) -> tuple[HiddenHypothesis, ...]:
    """Return the complete uniform-prior hypothesis space in stable order."""
    if isinstance(n_services, bool) or not isinstance(n_services, Integral):
        raise TypeError("n_services must be an integer")
    n_services = int(n_services)
    if n_services < 1:
        raise ValueError("n_services must be at least 1")

    return tuple(
        HiddenHypothesis(cause_service, fault_type, workload)
        for cause_service in range(n_services)
        for fault_type in FaultType
        for workload in Workload
    )


def sample_hidden_hypothesis(
    graph: DependencyGraph,
    rng: np.random.Generator,
) -> HiddenHypothesis:
    """Sample service, fault, and workload independently from uniform priors."""
    if not isinstance(graph, DependencyGraph):
        raise TypeError("graph must be a DependencyGraph")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")

    return HiddenHypothesis(
        cause_service=int(rng.integers(graph.n_services)),
        fault_type=FaultType(int(rng.integers(len(FaultType)))),
        workload=Workload(int(rng.integers(len(Workload)))),
    )


def generate_synthetic_observations(
    graph: DependencyGraph,
    hypothesis: HiddenHypothesis,
    rng: np.random.Generator,
) -> SyntheticObservations:
    """Sample all metric and log records once for a hidden hypothesis."""
    if not isinstance(graph, DependencyGraph):
        raise TypeError("graph must be a DependencyGraph")
    if not isinstance(hypothesis, HiddenHypothesis):
        raise TypeError("hypothesis must be a HiddenHypothesis")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    if hypothesis.cause_service >= graph.n_services:
        raise ValueError("hypothesis cause_service is not present in graph")

    metrics = np.empty(
        (graph.n_services, len(MetricCategory), METRIC_READINGS_PER_CATEGORY),
        dtype=np.bool_,
    )
    logs = np.empty(
        (graph.n_services, LOG_READINGS_PER_SERVICE),
        dtype=np.int64,
    )

    for service in range(graph.n_services):
        relationship = classify_service_relationship(
            graph,
            observed_service=service,
            cause_service=hypothesis.cause_service,
        )
        metric_probabilities = metric_high_probabilities(
            relationship,
            hypothesis.fault_type,
            hypothesis.workload,
        )
        metrics[service] = rng.random(
            (len(MetricCategory), METRIC_READINGS_PER_CATEGORY)
        ) < metric_probabilities[:, np.newaxis]

        log_probabilities = log_category_probabilities(
            relationship,
            hypothesis.fault_type,
            hypothesis.workload,
        )
        logs[service] = rng.choice(
            len(LogCategory),
            size=LOG_READINGS_PER_SERVICE,
            p=log_probabilities,
        )

    return SyntheticObservations(metrics, logs)
