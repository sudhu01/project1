"""Tests for hidden incident hypothesis generation."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from rca_sim.contracts import MetricEvidenceRecord
from rca_sim.graph import DependencyGraph
from rca_sim.world import (
    FaultType,
    HiddenHypothesis,
    HiddenIncidentWorld,
    LogCategory,
    MetricCategory,
    RelationshipKind,
    ServiceRelationship,
    Workload,
    classify_service_relationship,
    enumerate_hidden_hypotheses,
    generate_incident_world,
    generate_synthetic_observations,
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


def test_synthetic_observation_shapes_types_and_reproducibility() -> None:
    graph = _test_graph()
    hypothesis = HiddenHypothesis(6, FaultType.NETWORK_DELAY, Workload.BUSY)
    first = generate_synthetic_observations(
        graph,
        hypothesis,
        np.random.default_rng(123),
    )
    second = generate_synthetic_observations(
        graph,
        hypothesis,
        np.random.default_rng(123),
    )

    assert first.n_services == 8
    assert first.metric_readings.shape == (8, 3, 2)
    assert first.metric_readings.dtype == np.bool_
    assert first.log_readings.shape == (8, 3)
    assert first.log_readings.dtype == np.int64
    assert np.all((0 <= first.log_readings) & (first.log_readings < 4))
    np.testing.assert_array_equal(first.metric_readings, second.metric_readings)
    np.testing.assert_array_equal(first.log_readings, second.log_readings)


def test_synthetic_observations_are_persistent_read_only_samples() -> None:
    observations = generate_synthetic_observations(
        _test_graph(),
        HiddenHypothesis(3, FaultType.CPU, Workload.NORMAL),
        np.random.default_rng(55),
    )

    with pytest.raises(ValueError):
        observations.metric_readings[0, MetricCategory.CPU_HIGH, 0] = False
    with pytest.raises(ValueError):
        observations.log_readings[0, 0] = LogCategory.OTHER_ERROR


def test_sample_frequencies_match_fixed_hypothesis_likelihoods() -> None:
    graph = _test_graph()
    hypothesis = HiddenHypothesis(3, FaultType.CPU, Workload.NORMAL)
    rng = np.random.default_rng(8675309)
    sample_count = 10_000
    metric_counts = np.zeros(len(MetricCategory), dtype=np.int64)
    log_counts = np.zeros(len(LogCategory), dtype=np.int64)

    for _ in range(sample_count):
        observations = generate_synthetic_observations(graph, hypothesis, rng)
        metric_counts += observations.metric_readings[3].sum(axis=1)
        log_counts += np.bincount(
            observations.log_readings[3],
            minlength=len(LogCategory),
        )

    metric_trials = sample_count * 2
    expected_metrics = np.array([0.85, 0.15, 0.75])
    metric_standard_errors = np.sqrt(
        expected_metrics * (1.0 - expected_metrics) / metric_trials
    )
    assert np.all(
        np.abs(metric_counts / metric_trials - expected_metrics)
        <= 5.0 * metric_standard_errors + 0.002
    )

    log_trials = sample_count * 3
    expected_logs = np.array([0.10, 0.65, 0.15, 0.10])
    log_standard_errors = np.sqrt(expected_logs * (1.0 - expected_logs) / log_trials)
    assert np.all(
        np.abs(log_counts / log_trials - expected_logs)
        <= 5.0 * log_standard_errors + 0.002
    )


def test_generation_rejects_cause_outside_graph() -> None:
    with pytest.raises(ValueError, match="not present"):
        generate_synthetic_observations(
            _test_graph(),
            HiddenHypothesis(8, FaultType.CPU, Workload.NORMAL),
            np.random.default_rng(0),
        )


def test_complete_incident_assigns_all_stable_evidence_ids() -> None:
    world = generate_incident_world(n_services=8, incident_seed=20260913)

    assert len(world.evidence_ids) == 72
    assert len(set(world.evidence_ids)) == 72
    assert "metrics/service-3/cpu/0" in world.evidence_by_id
    assert "metrics/service-3/memory/1" in world.evidence_by_id
    assert "metrics/service-3/latency/1" in world.evidence_by_id
    assert "logs/service-3/0" in world.evidence_by_id
    assert "logs/service-3/2" in world.evidence_by_id


def test_evidence_records_match_frozen_observation_arrays() -> None:
    world = generate_incident_world(n_services=8, incident_seed=314159)

    metric_record = world.get_evidence("metrics/service-3/latency/1")
    log_record = world.get_evidence("logs/service-3/2")

    assert metric_record.evidence_id == "metrics/service-3/latency/1"
    assert metric_record.kind == "latency_high"
    assert metric_record.value is bool(
        world.observations.metric_readings[3, MetricCategory.LATENCY_HIGH, 1]
    )
    assert log_record.evidence_id == "logs/service-3/2"
    assert log_record.kind == "log_category"
    assert log_record.value is LogCategory(world.observations.log_readings[3, 2])


def test_repeated_reads_return_the_same_record_without_mutation() -> None:
    world = generate_incident_world(n_services=8, incident_seed=271828)
    before_ids = world.evidence_ids

    first = world.get_evidence("metrics/service-5/cpu/0")
    world.get_evidence("logs/service-1/2")
    second = world.get_evidence("metrics/service-5/cpu/0")

    assert first is second
    assert world.evidence_ids == before_ids
    with pytest.raises(TypeError):
        world.evidence_by_id["new-id"] = first  # type: ignore[index]


def test_evidence_lookup_order_does_not_change_unseen_records() -> None:
    forward = generate_incident_world(n_services=8, incident_seed=161803)
    reverse = generate_incident_world(n_services=8, incident_seed=161803)

    for evidence_id in forward.evidence_ids:
        forward.get_evidence(evidence_id)
    for evidence_id in reversed(reverse.evidence_ids):
        reverse.get_evidence(evidence_id)

    assert forward.evidence_by_id == reverse.evidence_by_id
    np.testing.assert_array_equal(
        forward.observations.metric_readings,
        reverse.observations.metric_readings,
    )
    np.testing.assert_array_equal(
        forward.observations.log_readings,
        reverse.observations.log_readings,
    )


def test_incident_replays_exactly_from_stored_metadata() -> None:
    original = generate_incident_world(
        n_services=8,
        incident_seed=np.int64(424242),
        extra_edge_probability=0.35,
    )
    replayed = original.replay()

    assert original.incident_seed == 424242
    assert original.generator_version == "sim_v0"
    assert original.extra_edge_probability == pytest.approx(0.35)
    assert original.graph.entry_service == replayed.graph.entry_service
    np.testing.assert_array_equal(original.graph.adjacency, replayed.graph.adjacency)
    assert original.hypothesis == replayed.hypothesis
    assert original.evidence_by_id == replayed.evidence_by_id


def test_unknown_evidence_id_fails_without_changing_world() -> None:
    world = generate_incident_world(n_services=8, incident_seed=7)
    before = world.evidence_ids

    with pytest.raises(KeyError, match="unknown evidence ID"):
        world.get_evidence("metrics/service-99/cpu/0")

    assert world.evidence_ids == before


@pytest.mark.parametrize(
    ("seed", "probability", "error"),
    [
        (-1, 0.15, ValueError),
        (True, 0.15, TypeError),
        (1, -0.1, ValueError),
        (1, 1.1, ValueError),
        (1, float("nan"), ValueError),
    ],
)
def test_incident_generation_validates_replay_metadata(
    seed: int,
    probability: float,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        generate_incident_world(
            n_services=8,
            incident_seed=seed,
            extra_edge_probability=probability,
        )


def test_replay_rejects_unknown_generator_version() -> None:
    current = generate_incident_world(n_services=8, incident_seed=19)
    old = HiddenIncidentWorld(
        graph=current.graph,
        hypothesis=current.hypothesis,
        observations=current.observations,
        incident_seed=current.incident_seed,
        generator_version="sim_v999",
    )

    with pytest.raises(ValueError, match="cannot replay"):
        old.replay()


def test_metric_record_rejects_boolean_category_encoding() -> None:
    with pytest.raises(TypeError, match="metric"):
        MetricEvidenceRecord(
            service_id=0,
            metric=False,  # type: ignore[arg-type]
            reading_index=0,
            value=False,
        )
