"""Tests for diagnostic tools."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from gymnasium.spaces import Discrete

from rca_sim.contracts import LogEvidenceRecord, MetricCategory, MetricEvidenceRecord
from rca_sim.tools import (
    ACTION_SPACE_SIZE,
    MAX_SERVICE_SLOTS,
    PROBE_TEMPLATES,
    PROBE_TEMPLATE_NAMES,
    STOP_ACTION_INDEX,
    TEMPLATES_PER_SERVICE,
    ProbePreset,
    ProbeResult,
    ProbeStatus,
    ProbeTemplate,
    ProbeTool,
    collect_probe_evidence,
    construct_action_mask,
    decode_probe_action,
    get_probe_template,
    is_stop_action,
    execute_probe_action,
    probe_action_index,
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


def test_action_mapping_constants_match_the_fixed_padded_space() -> None:
    assert PROBE_TEMPLATE_NAMES == (
        "metrics.quick",
        "metrics.detailed",
        "logs.quick",
        "logs.detailed",
    )
    assert TEMPLATES_PER_SERVICE == 4
    assert MAX_SERVICE_SLOTS == 10
    assert STOP_ACTION_INDEX == 40
    assert ACTION_SPACE_SIZE == 41

    action_space = Discrete(ACTION_SPACE_SIZE)
    assert action_space.contains(0)
    assert action_space.contains(STOP_ACTION_INDEX)
    assert not action_space.contains(ACTION_SPACE_SIZE)


@pytest.mark.parametrize(
    ("service_slot", "template_name", "expected_index"),
    [
        (0, "metrics.quick", 0),
        (0, "logs.detailed", 3),
        (3, "logs.quick", 14),
        (9, "metrics.quick", 36),
        (9, "logs.detailed", 39),
    ],
)
def test_probe_action_index_uses_service_major_template_order(
    service_slot: int,
    template_name: str,
    expected_index: int,
) -> None:
    assert probe_action_index(service_slot, template_name) == expected_index


def test_all_forty_probe_indices_round_trip() -> None:
    encoded_indices: list[int] = []

    for service_slot in range(MAX_SERVICE_SLOTS):
        for template_name in PROBE_TEMPLATE_NAMES:
            action_index = probe_action_index(service_slot, template_name)
            decoded_slot, decoded_template = decode_probe_action(action_index)
            encoded_indices.append(action_index)

            assert decoded_slot == service_slot
            assert decoded_template is PROBE_TEMPLATES[template_name]

    assert encoded_indices == list(range(STOP_ACTION_INDEX))


def test_stop_is_fixed_after_every_padded_probe_slot() -> None:
    assert is_stop_action(np.int64(40))
    assert not is_stop_action(0)
    with pytest.raises(ValueError, match="STOP"):
        decode_probe_action(STOP_ACTION_INDEX)


def test_eight_active_services_occupy_the_first_thirty_two_probe_slots() -> None:
    active_probe_indices = {
        probe_action_index(service_slot, template_name)
        for service_slot in range(8)
        for template_name in PROBE_TEMPLATE_NAMES
    }
    padded_probe_indices = set(range(32, STOP_ACTION_INDEX))

    assert active_probe_indices == set(range(32))
    assert len(padded_probe_indices) == 8
    assert active_probe_indices.isdisjoint(padded_probe_indices)


@pytest.mark.parametrize("service_slot", [-1, 10, False, 1.5])
def test_probe_action_index_rejects_invalid_service_slots(
    service_slot: object,
) -> None:
    with pytest.raises((TypeError, ValueError), match="service_slot"):
        probe_action_index(service_slot, "metrics.quick")  # type: ignore[arg-type]


@pytest.mark.parametrize("action_index", [-1, 41, False, 1.5])
def test_action_decoding_rejects_indices_outside_discrete_space(
    action_index: object,
) -> None:
    with pytest.raises((TypeError, ValueError), match="action_index"):
        decode_probe_action(action_index)  # type: ignore[arg-type]


def test_initial_action_mask_enables_active_probes_and_stop_only() -> None:
    mask = construct_action_mask(
        n_services=8,
        remaining_credits=8,
        seen_evidence_ids=(),
    )

    assert mask.dtype == np.bool_
    assert mask.shape == (ACTION_SPACE_SIZE,)
    assert mask[:32].all()
    assert not mask[32:STOP_ACTION_INDEX].any()
    assert mask[STOP_ACTION_INDEX]


def test_action_mask_applies_cost_availability_and_coverage_rules() -> None:
    metrics_detailed = PROBE_TEMPLATES["metrics.detailed"]
    seen = metrics_detailed.evidence_ids(0)
    available_tools = {
        0: frozenset(ProbeTool),
        1: frozenset({ProbeTool.METRICS}),
    }

    mask = construct_action_mask(
        n_services=2,
        remaining_credits=2,
        seen_evidence_ids=seen,
        available_tools=available_tools,
    )

    assert not mask[probe_action_index(0, "metrics.quick")]
    assert not mask[probe_action_index(0, "metrics.detailed")]
    assert mask[probe_action_index(0, "logs.quick")]
    assert not mask[probe_action_index(0, "logs.detailed")]
    assert mask[probe_action_index(1, "metrics.quick")]
    assert mask[probe_action_index(1, "metrics.detailed")]
    assert not mask[probe_action_index(1, "logs.quick")]
    assert mask[STOP_ACTION_INDEX]


def test_quick_then_detailed_keeps_only_unseen_coverage_actionable() -> None:
    quick_ids = PROBE_TEMPLATES["metrics.quick"].evidence_ids(3)

    mask = construct_action_mask(
        n_services=8,
        remaining_credits=2,
        seen_evidence_ids=quick_ids,
    )

    assert not mask[probe_action_index(3, "metrics.quick")]
    assert mask[probe_action_index(3, "metrics.detailed")]


def test_terminated_action_mask_has_no_valid_actions() -> None:
    mask = construct_action_mask(
        n_services=8,
        remaining_credits=8,
        seen_evidence_ids=(),
        terminated=True,
    )

    assert not mask.any()


@pytest.mark.parametrize(
    "action_index",
    [
        probe_action_index(8, "metrics.quick"),
        probe_action_index(0, "logs.detailed"),
    ],
)
def test_executor_rejects_invalid_action_before_reading_world(
    action_index: int,
) -> None:
    world = generate_incident_world(n_services=8, incident_seed=108)

    with pytest.raises(ValueError, match="invalid"):
        execute_probe_action(
            world,
            action_index,
            remaining_credits=1,
            seen_evidence_ids=(),
        )


def test_executor_returns_backend_neutral_result() -> None:
    world = generate_incident_world(n_services=8, incident_seed=109)
    action_index = probe_action_index(3, "metrics.quick")

    result = execute_probe_action(
        world,
        action_index,
        remaining_credits=8,
        seen_evidence_ids=(),
    )
    payload = result.as_dict()

    assert isinstance(result, ProbeResult)
    assert result.status is ProbeStatus.OK
    assert payload["action"] == {
        "tool": "metrics",
        "target_id": "service-3",
        "preset": "quick",
    }
    assert payload["status"] == "ok"
    assert payload["cost"] == {"credits": 1}
    assert payload["coverage"] == {"complete": True}
    assert [item["id"] for item in payload["evidence"]] == [
        "metrics/service-3/cpu/0",
        "metrics/service-3/memory/0",
        "metrics/service-3/latency/0",
    ]
    assert all(item["value"] in (0, 1) for item in payload["evidence"])


def test_executor_masks_exact_repeat_and_does_not_charge_or_mutate() -> None:
    world = generate_incident_world(n_services=8, incident_seed=110)
    action_index = probe_action_index(2, "logs.quick")
    seen = PROBE_TEMPLATES["logs.quick"].evidence_ids(2)

    with pytest.raises(ValueError, match="invalid"):
        execute_probe_action(
            world,
            action_index,
            remaining_credits=8,
            seen_evidence_ids=seen,
        )

    assert world.evidence_ids == generate_incident_world(
        n_services=8,
        incident_seed=110,
    ).evidence_ids
