"""Tests for exact belief updates."""

import numpy as np
import pytest

from rca_sim.belief import ExactBeliefEstimator
from rca_sim.contracts import (
    FaultType,
    LogCategory,
    LogEvidenceRecord,
    MetricCategory,
    MetricEvidenceRecord,
    Workload,
)
from rca_sim.evidence import EvidenceConflictError
from rca_sim.graph import DependencyGraph, classify_service_relationship
from rca_sim.likelihoods import (
    log_category_probability,
    metric_high_probability,
)


def _graph() -> DependencyGraph:
    return DependencyGraph.from_edges(
        2,
        [(0, 1)],
        entry_service=0,
    )


def _record_likelihood(
    graph: DependencyGraph,
    record: MetricEvidenceRecord | LogEvidenceRecord,
    service: int,
    fault: FaultType,
    workload: Workload,
) -> float:
    relationship = classify_service_relationship(
        graph,
        observed_service=record.service_id,
        cause_service=service,
    )
    if isinstance(record, MetricEvidenceRecord):
        high_probability = metric_high_probability(
            relationship,
            fault,
            workload,
            record.metric,
        )
        return high_probability if record.value else 1.0 - high_probability
    return log_category_probability(
        relationship,
        fault,
        workload,
        record.value,
    )


def test_initial_posterior_is_uniform_over_every_hypothesis() -> None:
    estimator = ExactBeliefEstimator(_graph())

    assert estimator.posterior.shape == (2, 3, 2)
    np.testing.assert_allclose(estimator.posterior, np.full((2, 3, 2), 1.0 / 12.0))
    assert estimator.posterior.sum() == pytest.approx(1.0)
    assert not estimator.posterior.flags.writeable


def test_batch_update_multiplies_likelihoods_with_workload_conditioned() -> None:
    graph = _graph()
    estimator = ExactBeliefEstimator(graph)
    records = (
        MetricEvidenceRecord(0, MetricCategory.CPU_HIGH, 0, True),
        LogEvidenceRecord(0, 0, LogCategory.RESOURCE_PRESSURE),
    )

    added = estimator.update(records)

    expected = np.empty((2, 3, 2), dtype=np.float64)
    for service in range(2):
        for fault in FaultType:
            for workload in Workload:
                expected[service, fault, workload] = np.prod(
                    [
                        _record_likelihood(
                            graph,
                            record,
                            service,
                            fault,
                            workload,
                        )
                        for record in records
                    ]
                )
    expected /= expected.sum()

    assert added == records
    np.testing.assert_allclose(estimator.posterior, expected)
    assert estimator.posterior.sum() == pytest.approx(1.0)
    assert np.all(estimator.posterior >= 0.0)


def test_false_metric_uses_one_minus_high_probability() -> None:
    graph = DependencyGraph.from_edges(1, [], entry_service=0)
    estimator = ExactBeliefEstimator(graph)
    record = MetricEvidenceRecord(0, MetricCategory.CPU_HIGH, 0, False)

    estimator.update([record])

    expected = np.array(
        [
            [
                [1.0 - metric_high_probability(
                    classify_service_relationship(
                        graph,
                        observed_service=0,
                        cause_service=0,
                    ),
                    fault,
                    workload,
                    MetricCategory.CPU_HIGH,
                ) for workload in Workload]
                for fault in FaultType
            ]
        ]
    )
    expected /= expected.sum()
    np.testing.assert_allclose(estimator.posterior, expected)


def test_exact_repeat_does_not_update_posterior_twice() -> None:
    estimator = ExactBeliefEstimator(_graph())
    record = MetricEvidenceRecord(0, MetricCategory.LATENCY_HIGH, 0, True)
    estimator.update([record])
    after_first = estimator.posterior

    added = estimator.update([record, record])

    assert added == ()
    assert estimator.evidence_ids == (record.evidence_id,)
    np.testing.assert_array_equal(estimator.posterior, after_first)


def test_conflicting_or_invalid_batch_is_atomic() -> None:
    estimator = ExactBeliefEstimator(_graph())
    original = MetricEvidenceRecord(0, MetricCategory.CPU_HIGH, 0, True)
    estimator.update([original])
    before = estimator.posterior
    new_record = LogEvidenceRecord(0, 0, LogCategory.TIMEOUT)
    conflict = MetricEvidenceRecord(0, MetricCategory.CPU_HIGH, 0, False)

    with pytest.raises(EvidenceConflictError, match="conflicting values"):
        estimator.update([new_record, conflict])

    assert estimator.evidence_ids == (original.evidence_id,)
    np.testing.assert_array_equal(estimator.posterior, before)

    with pytest.raises(ValueError, match="not present"):
        estimator.update([MetricEvidenceRecord(2, MetricCategory.CPU_HIGH, 0, True)])
    np.testing.assert_array_equal(estimator.posterior, before)


def test_diagnostic_quantities_are_exact_marginals() -> None:
    estimator = ExactBeliefEstimator(_graph())
    estimator.update(
        [MetricEvidenceRecord(1, MetricCategory.MEMORY_HIGH, 0, True)]
    )

    posterior = estimator.posterior
    diagnostics = estimator.diagnostic_quantities()

    np.testing.assert_allclose(
        diagnostics.belief_service_fault,
        posterior.sum(axis=2),
    )
    np.testing.assert_allclose(
        diagnostics.belief_service,
        posterior.sum(axis=(1, 2)),
    )
    assert diagnostics.belief_busy == pytest.approx(
        posterior[:, :, Workload.BUSY].sum()
    )
    assert diagnostics.predicted_service == int(
        np.argmax(diagnostics.belief_service)
    )
    assert diagnostics.predicted_fault is FaultType(
        int(
            np.argmax(
                diagnostics.belief_service_fault[diagnostics.predicted_service]
            )
        )
    )
    assert not diagnostics.belief_service_fault.flags.writeable
    assert not diagnostics.belief_service.flags.writeable


def test_uniform_ties_choose_lowest_service_and_fault_encoding() -> None:
    estimator = ExactBeliefEstimator(_graph())

    assert estimator.predicted_service == 0
    assert estimator.predicted_fault is FaultType.CPU
    np.testing.assert_allclose(estimator.belief_service, [0.5, 0.5])
    assert estimator.belief_busy == pytest.approx(0.5)


def test_estimator_rejects_non_graph_and_non_evidence_inputs() -> None:
    with pytest.raises(TypeError, match="DependencyGraph"):
        ExactBeliefEstimator(object())  # type: ignore[arg-type]

    estimator = ExactBeliefEstimator(_graph())
    with pytest.raises(TypeError, match="evidence records"):
        estimator.update([object()])  # type: ignore[list-item]
