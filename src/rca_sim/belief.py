"""Exact diagnostic belief updates from acquired public evidence."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from rca_sim.contracts import (
    EvidenceRecord,
    FaultType,
    LogEvidenceRecord,
    MetricEvidenceRecord,
    Workload,
)
from rca_sim.evidence import EvidenceConflictError
from rca_sim.graph import (
    DependencyGraph,
    classify_service_relationship,
)
from rca_sim.likelihoods import (
    log_category_probability,
    metric_high_probability,
)


FloatArray = NDArray[np.float64]


class InconsistentEvidenceError(ValueError):
    """The acquired evidence has zero probability under every hypothesis."""


@dataclass(frozen=True, slots=True)
class DiagnosticQuantities:
    """Marginal beliefs and the diagnosis derived from a full posterior."""

    belief_service_fault: FloatArray
    belief_service: FloatArray
    belief_busy: float
    predicted_service: int
    predicted_fault: FaultType


class ExactBeliefEstimator:
    """Maintain the exact posterior over service, fault, and workload.

    The estimator sees the public dependency graph and evidence passed to
    :meth:`update`. It has no reference to the incident's sampled hypothesis or
    its unrevealed records.
    """

    def __init__(self, graph: DependencyGraph) -> None:
        if not isinstance(graph, DependencyGraph):
            raise TypeError("graph must be a DependencyGraph")

        self._graph = graph
        shape = (graph.n_services, len(FaultType), len(Workload))
        self._log_weights = np.full(
            shape,
            -np.log(float(np.prod(shape))),
            dtype=np.float64,
        )
        self._evidence_by_id: dict[str, EvidenceRecord] = {}

    @property
    def posterior(self) -> FloatArray:
        """Return an immutable snapshot indexed by service, fault, workload."""
        return _immutable_copy(_normalize_log_weights(self._log_weights))

    @property
    def log_weights(self) -> FloatArray:
        """Return the cumulative unnormalized log probabilities."""
        return _immutable_copy(self._log_weights)

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        """Return incorporated evidence IDs in first-seen order."""
        return tuple(self._evidence_by_id)

    def update(self, records: Iterable[EvidenceRecord]) -> tuple[EvidenceRecord, ...]:
        """Incorporate unseen records and return the records that changed belief.

        Exact repeats do nothing. A conflicting record or invalid service ID
        rejects the full batch before the posterior changes.
        """
        try:
            batch = tuple(records)
        except TypeError as error:
            raise TypeError("records must be an iterable of evidence records") from error

        pending: dict[str, EvidenceRecord] = {}
        for record in batch:
            _validate_record(record, self._graph.n_services)
            existing = pending.get(
                record.evidence_id,
                self._evidence_by_id.get(record.evidence_id),
            )
            if existing is not None and existing != record:
                raise EvidenceConflictError(
                    f"evidence ID {record.evidence_id!r} has conflicting values"
                )
            if existing is None:
                pending[record.evidence_id] = record

        if not pending:
            return ()

        log_likelihood = np.zeros_like(self._log_weights)
        for record in pending.values():
            log_likelihood += _record_log_likelihoods(self._graph, record)

        updated = self._log_weights + log_likelihood
        _normalize_log_weights(updated)

        self._log_weights = updated
        self._evidence_by_id.update(pending)
        return tuple(pending.values())

    def reference_posterior(self) -> FloatArray:
        """Recompute from the prior and complete deduplicated evidence ledger."""
        return recompute_posterior(
            self._graph,
            self._evidence_by_id.values(),
        )

    def diagnostic_quantities(self) -> DiagnosticQuantities:
        """Return the Section 6.2 marginals and deterministic diagnosis."""
        posterior = _normalize_log_weights(self._log_weights)
        service_fault = posterior.sum(axis=2)
        service = service_fault.sum(axis=1)
        busy = float(posterior[:, :, Workload.BUSY].sum())
        predicted_service = int(np.argmax(service))
        predicted_fault = FaultType(int(np.argmax(service_fault[predicted_service])))

        return DiagnosticQuantities(
            belief_service_fault=_immutable_copy(service_fault),
            belief_service=_immutable_copy(service),
            belief_busy=busy,
            predicted_service=predicted_service,
            predicted_fault=predicted_fault,
        )

    @property
    def belief_service_fault(self) -> FloatArray:
        return self.diagnostic_quantities().belief_service_fault

    @property
    def belief_service(self) -> FloatArray:
        return self.diagnostic_quantities().belief_service

    @property
    def belief_busy(self) -> float:
        return self.diagnostic_quantities().belief_busy

    @property
    def predicted_service(self) -> int:
        return self.diagnostic_quantities().predicted_service

    @property
    def predicted_fault(self) -> FaultType:
        return self.diagnostic_quantities().predicted_fault

def recompute_posterior(
    graph: DependencyGraph,
    records: Iterable[EvidenceRecord],
) -> FloatArray:
    """Recompute a posterior from the uniform prior and complete evidence."""
    if not isinstance(graph, DependencyGraph):
        raise TypeError("graph must be a DependencyGraph")
    try:
        batch = tuple(records)
    except TypeError as error:
        raise TypeError("records must be an iterable of evidence records") from error

    deduplicated: dict[str, EvidenceRecord] = {}
    for record in batch:
        _validate_record(record, graph.n_services)
        existing = deduplicated.get(record.evidence_id)
        if existing is not None and existing != record:
            raise EvidenceConflictError(
                f"evidence ID {record.evidence_id!r} has conflicting values"
            )
        if existing is None:
            deduplicated[record.evidence_id] = record

    shape = (graph.n_services, len(FaultType), len(Workload))
    log_weights = np.full(
        shape,
        -np.log(float(np.prod(shape))),
        dtype=np.float64,
    )
    for record in deduplicated.values():
        log_weights += _record_log_likelihoods(graph, record)
    return _immutable_copy(_normalize_log_weights(log_weights))


def _record_likelihoods(
    graph: DependencyGraph,
    record: EvidenceRecord,
) -> FloatArray:
    shape = (graph.n_services, len(FaultType), len(Workload))
    likelihoods = np.empty(shape, dtype=np.float64)
    for cause_service in range(graph.n_services):
        relationship = classify_service_relationship(
            graph,
            observed_service=record.service_id,
            cause_service=cause_service,
        )
        for fault_type in FaultType:
            for workload in Workload:
                if isinstance(record, MetricEvidenceRecord):
                    probability = metric_high_probability(
                        relationship,
                        fault_type,
                        workload,
                        record.metric,
                    )
                    likelihood = probability if record.value else 1.0 - probability
                else:
                    likelihood = log_category_probability(
                        relationship,
                        fault_type,
                        workload,
                        record.value,
                    )
                likelihoods[cause_service, fault_type, workload] = likelihood
    return likelihoods


def _record_log_likelihoods(
    graph: DependencyGraph,
    record: EvidenceRecord,
) -> FloatArray:
    likelihoods = _record_likelihoods(graph, record)
    if np.any(~np.isfinite(likelihoods)) or np.any(likelihoods < 0.0):
        raise ValueError("evidence likelihoods must be finite and nonnegative")
    result = np.full_like(likelihoods, -np.inf)
    np.log(likelihoods, out=result, where=likelihoods > 0.0)
    return result


def _normalize_log_weights(log_weights: FloatArray) -> FloatArray:
    maximum = float(np.max(log_weights))
    if not np.isfinite(maximum):
        raise InconsistentEvidenceError(
            "evidence is impossible under every supported hypothesis"
        )

    shifted = np.exp(log_weights - maximum)
    shifted_sum = float(shifted.sum())
    if not np.isfinite(shifted_sum) or shifted_sum <= 0.0:
        raise InconsistentEvidenceError(
            "evidence is impossible under every supported hypothesis"
        )
    return shifted / shifted_sum


def _validate_record(record: object, n_services: int) -> None:
    if not isinstance(record, (MetricEvidenceRecord, LogEvidenceRecord)):
        raise TypeError("records must contain only evidence records")
    if record.service_id >= n_services:
        raise ValueError("evidence service ID is not present in graph")


def _immutable_copy(array: FloatArray) -> FloatArray:
    result = np.array(array, dtype=np.float64, copy=True)
    result.flags.writeable = False
    return result
