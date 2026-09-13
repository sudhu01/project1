"""Tests for evidence likelihoods."""

import numpy as np
import pytest

from rca_sim.contracts import (
    FaultType,
    LogCategory,
    MetricCategory,
    RelationshipKind,
    ServiceRelationship,
    Workload,
)
from rca_sim.likelihoods import (
    log_category_probabilities,
    log_category_probability,
    metric_high_probabilities,
    metric_high_probability,
)


CAUSE = ServiceRelationship(RelationshipKind.CAUSE, 0)
DIRECT_CALLER = ServiceRelationship(RelationshipKind.AFFECTED_CALLER, 1)
DISTANT_CALLER = ServiceRelationship(RelationshipKind.AFFECTED_CALLER, 3)
UNRELATED = ServiceRelationship(RelationshipKind.UNRELATED, None)


@pytest.mark.parametrize(
    ("fault_type", "expected"),
    [
        (FaultType.CPU, [0.85, 0.15, 0.75]),
        (FaultType.MEMORY, [0.15, 0.85, 0.75]),
        (FaultType.NETWORK_DELAY, [0.15, 0.15, 0.90]),
    ],
)
def test_normal_cause_metric_probabilities(
    fault_type: FaultType,
    expected: list[float],
) -> None:
    actual = metric_high_probabilities(CAUSE, fault_type, Workload.NORMAL)

    np.testing.assert_allclose(actual, expected)


def test_caller_metric_latency_decays_with_distance() -> None:
    direct = metric_high_probabilities(
        DIRECT_CALLER,
        FaultType.CPU,
        Workload.NORMAL,
    )
    distant = metric_high_probabilities(
        DISTANT_CALLER,
        FaultType.NETWORK_DELAY,
        Workload.NORMAL,
    )
    unrelated = metric_high_probabilities(
        UNRELATED,
        FaultType.MEMORY,
        Workload.NORMAL,
    )

    np.testing.assert_allclose(direct, [0.10, 0.10, 0.70])
    np.testing.assert_allclose(distant, [0.10, 0.10, 0.15 + 0.55 * 0.7**2])
    np.testing.assert_allclose(unrelated, [0.10, 0.10, 0.15])
    assert direct[MetricCategory.LATENCY_HIGH] > distant[MetricCategory.LATENCY_HIGH]
    assert distant[MetricCategory.LATENCY_HIGH] > unrelated[MetricCategory.LATENCY_HIGH]


def test_busy_workload_adds_metric_background_and_caps_at_095() -> None:
    cpu_cause = metric_high_probabilities(CAUSE, FaultType.CPU, Workload.BUSY)
    network_cause = metric_high_probabilities(
        CAUSE,
        FaultType.NETWORK_DELAY,
        Workload.BUSY,
    )

    np.testing.assert_allclose(cpu_cause, [0.95, 0.25, 0.80])
    np.testing.assert_allclose(network_cause, [0.25, 0.25, 0.95])


def test_single_metric_probability_selects_fixed_category_order() -> None:
    probability = metric_high_probability(
        CAUSE,
        FaultType.MEMORY,
        Workload.NORMAL,
        MetricCategory.MEMORY_HIGH,
    )

    assert probability == pytest.approx(0.85)


@pytest.mark.parametrize(
    ("fault_type", "expected"),
    [
        (FaultType.CPU, [0.10, 0.65, 0.15, 0.10]),
        (FaultType.MEMORY, [0.10, 0.65, 0.15, 0.10]),
        (FaultType.NETWORK_DELAY, [0.10, 0.10, 0.70, 0.10]),
    ],
)
def test_normal_cause_log_probabilities(
    fault_type: FaultType,
    expected: list[float],
) -> None:
    actual = log_category_probabilities(CAUSE, fault_type, Workload.NORMAL)

    np.testing.assert_allclose(actual, expected)


def test_caller_log_distribution_mixes_toward_unrelated() -> None:
    direct = log_category_probabilities(
        DIRECT_CALLER,
        FaultType.CPU,
        Workload.NORMAL,
    )
    distant = log_category_probabilities(
        DISTANT_CALLER,
        FaultType.CPU,
        Workload.NORMAL,
    )
    unrelated = log_category_probabilities(
        UNRELATED,
        FaultType.CPU,
        Workload.NORMAL,
    )

    np.testing.assert_allclose(direct, [0.25, 0.10, 0.55, 0.10])
    np.testing.assert_allclose(
        distant,
        0.7**2 * np.array([0.25, 0.10, 0.55, 0.10])
        + (1.0 - 0.7**2) * np.array([0.75, 0.05, 0.10, 0.10]),
    )
    np.testing.assert_allclose(unrelated, [0.75, 0.05, 0.10, 0.10])
    assert direct[LogCategory.TIMEOUT] > distant[LogCategory.TIMEOUT]
    assert distant[LogCategory.TIMEOUT] > unrelated[LogCategory.TIMEOUT]


def test_busy_workload_mixes_log_background() -> None:
    normal = np.array([0.10, 0.65, 0.15, 0.10])
    busy_background = np.array([0.25, 0.35, 0.30, 0.10])
    actual = log_category_probabilities(CAUSE, FaultType.CPU, Workload.BUSY)

    np.testing.assert_allclose(actual, 0.85 * normal + 0.15 * busy_background)


@pytest.mark.parametrize("relationship", [CAUSE, DIRECT_CALLER, DISTANT_CALLER, UNRELATED])
@pytest.mark.parametrize("fault_type", list(FaultType))
@pytest.mark.parametrize("workload", list(Workload))
def test_all_probability_outputs_are_valid(
    relationship: ServiceRelationship,
    fault_type: FaultType,
    workload: Workload,
) -> None:
    metrics = metric_high_probabilities(relationship, fault_type, workload)
    logs = log_category_probabilities(relationship, fault_type, workload)

    assert np.all((0.0 <= metrics) & (metrics <= 1.0))
    assert np.all((0.0 <= logs) & (logs <= 1.0))
    assert logs.sum() == pytest.approx(1.0)
    assert not metrics.flags.writeable
    assert not logs.flags.writeable


def test_single_log_probability_selects_fixed_category_order() -> None:
    probability = log_category_probability(
        CAUSE,
        FaultType.NETWORK_DELAY,
        Workload.NORMAL,
        LogCategory.TIMEOUT,
    )

    assert probability == pytest.approx(0.70)


def test_probability_lookups_reject_boolean_category_encodings() -> None:
    with pytest.raises(TypeError, match="metric category"):
        metric_high_probability(
            CAUSE,
            FaultType.CPU,
            Workload.NORMAL,
            False,  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="log category"):
        log_category_probability(
            CAUSE,
            FaultType.CPU,
            Workload.NORMAL,
            True,  # type: ignore[arg-type]
        )
