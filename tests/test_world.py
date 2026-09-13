"""Tests for hidden incident hypothesis generation."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from rca_sim.graph import DependencyGraph
from rca_sim.world import (
    FaultType,
    HiddenHypothesis,
    RelationshipKind,
    ServiceRelationship,
    Workload,
    classify_service_relationship,
    enumerate_hidden_hypotheses,
    sample_hidden_hypothesis,
)


def _test_graph() -> DependencyGraph:
    return DependencyGraph.from_edges(
        8,
        [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (5, 6), (5, 7)],
        entry_service=0,
    )


def test_fault_and_workload_encodings_are_stable() -> None:
    assert list(FaultType) == [
        FaultType.CPU,
        FaultType.MEMORY,
        FaultType.NETWORK_DELAY,
    ]
    assert [int(fault) for fault in FaultType] == [0, 1, 2]
    assert list(Workload) == [Workload.NORMAL, Workload.BUSY]
    assert [int(workload) for workload in Workload] == [0, 1]


def test_eight_services_have_48_unique_hypotheses() -> None:
    hypotheses = enumerate_hidden_hypotheses(8)

    assert len(hypotheses) == 48
    assert len(set(hypotheses)) == 48
    assert {item.cause_service for item in hypotheses} == set(range(8))
    assert {item.fault_type for item in hypotheses} == set(FaultType)
    assert {item.workload for item in hypotheses} == set(Workload)


def test_hypothesis_is_immutable_and_normalizes_numpy_integer() -> None:
    hypothesis = HiddenHypothesis(
        np.int64(3),
        FaultType.MEMORY,
        Workload.BUSY,
    )

    assert hypothesis.cause_service == 3
    assert type(hypothesis.cause_service) is int
    with pytest.raises(FrozenInstanceError):
        hypothesis.cause_service = 4  # type: ignore[misc]


def test_sampler_is_reproducible() -> None:
    graph = _test_graph()
    first_rng = np.random.default_rng(2718)
    second_rng = np.random.default_rng(2718)

    first = [sample_hidden_hypothesis(graph, first_rng) for _ in range(100)]
    second = [sample_hidden_hypothesis(graph, second_rng) for _ in range(100)]

    assert first == second


def test_sampler_uses_independent_uniform_priors() -> None:
    graph = _test_graph()
    rng = np.random.default_rng(20260913)
    sample_count = 60_000
    cause_counts = np.zeros(graph.n_services, dtype=np.int64)
    fault_counts = np.zeros(len(FaultType), dtype=np.int64)
    workload_counts = np.zeros(len(Workload), dtype=np.int64)

    for _ in range(sample_count):
        hypothesis = sample_hidden_hypothesis(graph, rng)
        cause_counts[hypothesis.cause_service] += 1
        fault_counts[hypothesis.fault_type] += 1
        workload_counts[hypothesis.workload] += 1

    def assert_matches_uniform(counts: np.ndarray) -> None:
        probability = 1.0 / len(counts)
        standard_error = np.sqrt(probability * (1.0 - probability) / sample_count)
        observed = counts / sample_count
        np.testing.assert_allclose(
            observed,
            probability,
            atol=5.0 * standard_error + 0.002,
            rtol=0.0,
        )

    assert_matches_uniform(cause_counts)
    assert_matches_uniform(fault_counts)
    assert_matches_uniform(workload_counts)


def test_sampling_does_not_condition_on_entry_service() -> None:
    graph = _test_graph()
    rng = np.random.default_rng(99)
    sampled_causes = {
        sample_hidden_hypothesis(graph, rng).cause_service for _ in range(1_000)
    }

    assert sampled_causes == set(range(graph.n_services))


@pytest.mark.parametrize("n_services", [0, -1])
def test_enumerator_requires_at_least_one_service(n_services: int) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        enumerate_hidden_hypotheses(n_services)


@pytest.mark.parametrize(
    ("fault_type", "workload", "message"),
    [
        (3, Workload.NORMAL, "fault_type"),
        (FaultType.CPU, 2, "workload"),
    ],
)
def test_hypothesis_rejects_unknown_categories(
    fault_type: int | FaultType,
    workload: int | Workload,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        HiddenHypothesis(0, fault_type, workload)  # type: ignore[arg-type]


def test_hypothesis_rejects_boolean_category_encodings() -> None:
    with pytest.raises(TypeError, match="fault_type"):
        HiddenHypothesis(0, False, Workload.NORMAL)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="workload"):
        HiddenHypothesis(0, FaultType.CPU, True)  # type: ignore[arg-type]


def test_sampler_requires_graph_and_generator() -> None:
    graph = _test_graph()
    with pytest.raises(TypeError, match="graph"):
        sample_hidden_hypothesis(8, np.random.default_rng(0))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="rng"):
        sample_hidden_hypothesis(graph, 0)  # type: ignore[arg-type]


def test_chain_relationships_propagate_from_dependency_to_callers() -> None:
    graph = DependencyGraph.from_edges(
        4,
        [(0, 1), (1, 2), (2, 3)],
        entry_service=0,
    )

    assert classify_service_relationship(
        graph, observed_service=3, cause_service=3
    ) == ServiceRelationship(
        RelationshipKind.CAUSE,
        distance=0,
    )
    assert classify_service_relationship(
        graph, observed_service=0, cause_service=3
    ) == ServiceRelationship(
        RelationshipKind.AFFECTED_CALLER,
        distance=3,
    )
    assert classify_service_relationship(
        graph, observed_service=2, cause_service=1
    ) == ServiceRelationship(
        RelationshipKind.UNRELATED,
        distance=None,
    )


def test_branching_tree_keeps_sibling_branch_unrelated() -> None:
    graph = DependencyGraph.from_edges(
        5,
        [(0, 1), (0, 2), (1, 3), (1, 4)],
        entry_service=0,
    )

    assert classify_service_relationship(
        graph, observed_service=0, cause_service=3
    ) == ServiceRelationship(
        RelationshipKind.AFFECTED_CALLER,
        distance=2,
    )
    assert classify_service_relationship(
        graph, observed_service=1, cause_service=3
    ) == ServiceRelationship(
        RelationshipKind.AFFECTED_CALLER,
        distance=1,
    )
    assert classify_service_relationship(
        graph, observed_service=2, cause_service=3
    ) == ServiceRelationship(
        RelationshipKind.UNRELATED,
        distance=None,
    )
    assert classify_service_relationship(
        graph, observed_service=4, cause_service=3
    ) == ServiceRelationship(
        RelationshipKind.UNRELATED,
        distance=None,
    )


def test_shared_dependency_uses_the_shortest_caller_path() -> None:
    graph = DependencyGraph.from_edges(
        5,
        [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (0, 4)],
        entry_service=0,
    )

    assert classify_service_relationship(
        graph, observed_service=0, cause_service=4
    ) == ServiceRelationship(
        RelationshipKind.AFFECTED_CALLER,
        distance=1,
    )
    assert classify_service_relationship(
        graph, observed_service=1, cause_service=4
    ) == ServiceRelationship(
        RelationshipKind.AFFECTED_CALLER,
        distance=2,
    )
    assert classify_service_relationship(
        graph, observed_service=2, cause_service=4
    ) == ServiceRelationship(
        RelationshipKind.AFFECTED_CALLER,
        distance=2,
    )


def test_every_service_gets_exactly_one_relationship() -> None:
    graph = _test_graph()

    for cause_service in range(graph.n_services):
        relationships = [
            classify_service_relationship(
                graph,
                observed_service=observed_service,
                cause_service=cause_service,
            )
            for observed_service in range(graph.n_services)
        ]
        assert relationships[cause_service].kind is RelationshipKind.CAUSE
        assert sum(item.kind is RelationshipKind.CAUSE for item in relationships) == 1
        assert all(item.kind in RelationshipKind for item in relationships)


@pytest.mark.parametrize(
    ("observed_service", "cause_service"),
    [(-1, 0), (0, -1), (8, 0), (0, 8)],
)
def test_relationship_classifier_rejects_invalid_service_ids(
    observed_service: int,
    cause_service: int,
) -> None:
    with pytest.raises(ValueError, match="out of range"):
        classify_service_relationship(
            _test_graph(),
            observed_service=observed_service,
            cause_service=cause_service,
        )


@pytest.mark.parametrize(
    ("kind", "distance"),
    [
        (RelationshipKind.CAUSE, None),
        (RelationshipKind.AFFECTED_CALLER, 0),
        (RelationshipKind.UNRELATED, 1),
    ],
)
def test_relationship_record_enforces_distance_invariants(
    kind: RelationshipKind,
    distance: int | None,
) -> None:
    with pytest.raises(ValueError):
        ServiceRelationship(kind, distance)
