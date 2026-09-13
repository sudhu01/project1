"""Tests for diagnostic tools."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from rca_sim.contracts import LogEvidenceRecord, MetricCategory, MetricEvidenceRecord
from rca_sim.tools import (
    PROBE_TEMPLATES,
    ProbePreset,
    ProbeTemplate,
    ProbeTool,
    collect_probe_evidence,
    get_probe_template,
)
from rca_sim.world import generate_incident_world


EXPECTED_TEMPLATES = {
    "metrics.quick": (1, 3),
    "metrics.detailed": (2, 6),
    "logs.quick": (2, 1),
    "logs.detailed": (4, 3),
}


def test_registry_has_exactly_four_templates_with_fixed_costs_and_coverage() -> None:
    assert {
        name: (template.cost_credits, len(template.evidence_ids(0)))
        for name, template in PROBE_TEMPLATES.items()
    } == EXPECTED_TEMPLATES


def test_registry_order_is_the_planned_template_index_order() -> None:
    assert tuple(PROBE_TEMPLATES) == (
        "metrics.quick",
        "metrics.detailed",
        "logs.quick",
        "logs.detailed",
    )


def test_metrics_quick_returns_reading_zero_for_every_metric() -> None:
    world = generate_incident_world(n_services=8, incident_seed=101)

    records = collect_probe_evidence(world, "metrics.quick", np.int64(3))

    assert [record.evidence_id for record in records] == [
        "metrics/service-3/cpu/0",
        "metrics/service-3/memory/0",
        "metrics/service-3/latency/0",
    ]
    assert all(isinstance(record, MetricEvidenceRecord) for record in records)


def test_metrics_detailed_returns_both_readings_for_every_metric() -> None:
    world = generate_incident_world(n_services=8, incident_seed=102)

    records = collect_probe_evidence(
        world,
        get_probe_template("metrics.detailed"),
        5,
    )

    assert [record.evidence_id for record in records] == [
        f"metrics/service-5/{metric.evidence_name}/{reading_index}"
        for reading_index in (0, 1)
        for metric in MetricCategory
    ]


@pytest.mark.parametrize(
    ("template_name", "expected_indices"),
    [("logs.quick", [0]), ("logs.detailed", [0, 1, 2])],
)
def test_log_templates_return_the_planned_records(
    template_name: str,
    expected_indices: list[int],
) -> None:
    world = generate_incident_world(n_services=8, incident_seed=103)

    records = collect_probe_evidence(world, template_name, 2)

    assert [record.evidence_id for record in records] == [
        f"logs/service-2/{index}" for index in expected_indices
    ]
    assert all(isinstance(record, LogEvidenceRecord) for record in records)


@pytest.mark.parametrize("tool", list(ProbeTool))
def test_detailed_coverage_contains_quick_coverage(tool: ProbeTool) -> None:
    quick = PROBE_TEMPLATES[f"{tool.value}.quick"]
    detailed = PROBE_TEMPLATES[f"{tool.value}.detailed"]

    assert set(quick.evidence_ids(4)) < set(detailed.evidence_ids(4))


def test_probe_reads_are_repeatable_and_do_not_change_the_world() -> None:
    world = generate_incident_world(n_services=8, incident_seed=104)
    before_ids = world.evidence_ids

    first = collect_probe_evidence(world, "metrics.detailed", 6)
    collect_probe_evidence(world, "logs.quick", 1)
    second = collect_probe_evidence(world, "metrics.detailed", 6)

    assert first == second
    assert all(left is right for left, right in zip(first, second, strict=True))
    assert world.evidence_ids == before_ids


def test_every_active_service_supports_every_template() -> None:
    world = generate_incident_world(n_services=6, incident_seed=105)

    for service_id in range(world.graph.n_services):
        for name, (_, expected_count) in EXPECTED_TEMPLATES.items():
            assert len(collect_probe_evidence(world, name, service_id)) == expected_count


def test_template_and_registry_are_immutable() -> None:
    template = PROBE_TEMPLATES["metrics.quick"]

    with pytest.raises(FrozenInstanceError):
        template.cost_credits = 99  # type: ignore[misc]
    with pytest.raises(TypeError):
        PROBE_TEMPLATES["metrics.quick"] = template  # type: ignore[index]


@pytest.mark.parametrize("service_id", [-1, 8])
def test_probe_rejects_service_outside_the_world(service_id: int) -> None:
    world = generate_incident_world(n_services=8, incident_seed=106)

    with pytest.raises(ValueError, match="service_id"):
        collect_probe_evidence(world, "metrics.quick", service_id)


def test_probe_rejects_boolean_service_id_and_unknown_template() -> None:
    world = generate_incident_world(n_services=8, incident_seed=107)

    with pytest.raises(TypeError, match="service_id"):
        collect_probe_evidence(world, "metrics.quick", False)
    with pytest.raises(KeyError, match="unknown probe template"):
        collect_probe_evidence(world, "traces.quick", 0)


def test_template_rejects_coverage_that_disagrees_with_its_semantics() -> None:
    with pytest.raises(ValueError, match="must cover"):
        ProbeTemplate(
            ProbeTool.LOGS,
            ProbePreset.DETAILED,
            cost_credits=4,
            reading_indices=(0, 1),
        )
